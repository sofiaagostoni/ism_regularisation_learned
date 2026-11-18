# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
import torch
import deepinv as dinv
from matplotlib import cm
import wandb  # <-- aggiunto

from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity

from opt_functions.Data_manager.generate_measurments import *
from opt_functions.plot_results import *
from opt_functions.Solver_functions import *
from opt_functions.Data_manager.real_data_load import *

from scipy.optimize import least_squares

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

tv = TVLoss()

# ---------------- GENERAZIONE DATI ----------------
hparams = {
    'Nz': 1,
    'pxsize': 40,
    'IS_REAL': False,
    'LOAD_FROM_FILE': True,
    'flux': 30,
    'lam': 0.001,
}
prior = 'drunet'
hparams['IS_3D'] = (hparams['Nz'] > 1)
hparams['real_name'] = '01_tomm20' if hparams['IS_REAL'] else 'tubulin'
hparams['path'] = 'Data/Simul_data/tub_3D.pth' if hparams['IS_3D'] else 'Data/Simul_data/tub_level.pth'

dataset = prepare_ism_data(
    is_real=hparams['IS_REAL'],
    real_name=hparams['real_name'],
    load_path=None if not hparams['LOAD_FROM_FILE'] else hparams['path'],
    phantom_type=hparams['real_name'],
    Nx=256, Ny=256,
    Nz=hparams['Nz'],
    pxsize=hparams['pxsize'],
    flux=hparams['flux'],
    device=device,
    show_plots=True,
    normalization_y=False,
)

noise_image = dataset['noise_image']
kl = KL(back=dataset["back_vec"])

finger_print = dataset["fingerprint"]
physics = dataset["physics"]

# Vettore iniziale
x_0 = dataset['x_init']
x_0 = (x_0 / x_0.max())

for i in range(25):
    max_y_i = torch.max(noise_image[i])
    noise_image[i] = (noise_image[i] / max_y_i) * finger_print[i]

# ---------------- MODELLO (DRUNet fine-tuned) ----------------
if prior == "drunet_finetune":
    drunet = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained=None, device=device)
    checkpoint = torch.load("training_drunet/best_model_checkpoint_drunet.pth", map_location=device)
    drunet.load_state_dict(checkpoint["model_state_dict"])
    
elif prior == "drunet":
    drunet = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)


parameters = {
    "max_iter": 900,
    "tollerance": 1e-12,
    "Lip_reg": dataset["L_th"] * 1e-3,
    "x_init": x_0,
    "physics": dataset["physics"],
    "back": dataset["back_vec"],
    "sigma": 1e-3,
    "ground_truth": dataset["ground_truth"] if not hparams['IS_REAL'] else None,
    "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
    "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
    "single_data_fid": KL_metric,
    "Pnp": drunet,
}


# ---------------- SOLVER (invariato) ----------------
def pnp_ism(y, back, parameters_pnp, device):

    data_fid       = parameters_pnp["data_fid"]
    grad_data_fid  = parameters_pnp["grad_data_fid"]
    single_fid     = parameters_pnp["single_data_fid"]
    tollerance     = parameters_pnp["tollerance"]
    max_iter       = parameters_pnp["max_iter"]
    x_init         = parameters_pnp["x_init"]
    sigma          = parameters_pnp["sigma"]
    L_max          = parameters_pnp["Lip_reg"]
    pnp            = parameters_pnp["Pnp"]
    physics        = parameters_pnp["physics"]

    x_gt = parameters_pnp['ground_truth']

    patience = 10

    funct    = torch.zeros(max_iter, device=device)
    iter_err = torch.zeros(max_iter, device=device)
    norm2    = torch.zeros(max_iter, device=device)
    psnr_vec = torch.zeros(max_iter, device=device)
    ssim_vec = torch.zeros(max_iter, device=device)

    x_k_prec = x_init.to(device)
    y    = y.to(device)
    back = back.to(device)

    tau = 1 / L_max

    best_psnr = -float('inf')
    best_x    = x_k_prec.clone()
    best_k    = 0
    no_improve = 0

    for k in range(max_iter):
        with torch.no_grad():

            x_k_succ = torch.max(
                x_k_prec - tau * grad_data_fid(y, x_k_prec, physics),
                torch.tensor(0.0, device=device),
            )
            x_k_succ = x_k_succ / x_k_succ.max()
            x_k_succ = pnp(x_k_succ, sigma)
            # x_k_succ = torch.clamp(x_k_succ, 0, 1)

            funct[k]    = data_fid(y, x_k_succ, physics)
            norm2[k]    = torch.norm(x_gt - x_k_succ, 'fro')
            iter_err[k] = torch.norm(x_k_prec - x_k_succ, 'fro') / torch.norm(x_k_prec, 'fro')
            psnr_vec[k] = psnr(x_gt / x_gt.max(), x_k_succ / x_k_succ.max())
            ssim_vec[k] = ssim(x_gt / x_gt.max(), x_k_succ / x_k_succ.max())

            if psnr_vec[k] > best_psnr:
                best_psnr  = psnr_vec[k].item()
                best_x     = x_k_succ.clone()
                best_k     = k
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping sul PSNR: massimo = {best_psnr:.4f} dB "
                      f"all'iterazione {best_k} (fermato a k = {k})")
                funct    = funct[0:k + 1]
                iter_err = iter_err[0:k + 1]
                norm2    = norm2[0:k + 1]
                psnr_vec = psnr_vec[0:k + 1]
                ssim_vec = ssim_vec[0:k + 1]
                break

            if iter_err[k] < tollerance:
                print(f"Convergence reached at iteration = {k}")
                funct    = funct[0:k + 1]
                iter_err = iter_err[0:k + 1]
                norm2    = norm2[0:k + 1]
                psnr_vec = psnr_vec[0:k + 1]
                ssim_vec = ssim_vec[0:k + 1]
                break

            x_k_prec = x_k_succ

    return best_x, funct.detach(), iter_err.detach(), norm2.detach(), psnr_vec.detach(), ssim_vec.detach()


# %%
# ==========================================
# 1. CONFIGURAZIONE DELLO SWEEP
# ==========================================
# grid = prodotto cartesiano di tutti i valori -> sostituisce i for annidati.
# NB: 5e-3 compare due volte nella tua lista step: W&B lo tratta come duplicato,
#     quindi verrà eseguito una volta sola. Togli il doppione se non ti serve.
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'best_psnr',   # metrica che lo sweep massimizza
        'goal': 'maximize',
    },
    'parameters': {
        'sigma': {'values': torch.linspace(1e-2, 5e-1, steps=60).tolist()},
        'step':  {'values': [1e-4, 5e-3, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]},
    },
}

sweep_id = wandb.sweep(
    sweep_config,
    entity="ism_regularisation",
    project="drunet-pnp-ism",   # <-- cambia se vuoi un altro progetto
)


# ==========================================
# 2. FUNZIONE DI ESECUZIONE (WRAPPER)
# ==========================================
def run_experiment():
    with wandb.init() as run:

        # 1. Parametri dinamici presi dallo sweep
        sigma = wandb.config.sigma
        step  = wandb.config.step
        run.name = f"drunet_step{step}_sigma{sigma:.5g}"

        parameters['sigma']    = sigma
        parameters['max_iter'] = 900
        parameters['Lip_reg']  = dataset["L_th"] * step
        # >>> ATTENZIONE: nel tuo pnp_ism 'step' NON viene usato (tau = 1/Lip_reg).
        #     Se 'step' deve essere il passo del gradiente, scommenta:
        # parameters['Lip_reg'] = 1.0 / step     # -> tau = 1/L = step

        wandb.config.update({
            "algorithm": prior,
            "max_iter": parameters["max_iter"],
            "tollerance": parameters["tollerance"],
            "flux": hparams["flux"],
            "real_name": hparams["real_name"],
        })

        # 2. Immagini di base (osservazione + ground truth)
        obs = noise_image.sum(0).detach().cpu().squeeze().numpy()
        obs = obs / (obs.max() + 1e-12)
        log_base = {"Observation": wandb.Image(cm.hot(obs), caption="Observation")}
        if parameters['ground_truth'] is not None:
            gt = parameters['ground_truth'].detach().cpu().squeeze().numpy()
            log_base["Ground_Truth"] = wandb.Image(gt / (gt.max() + 1e-12),
                                                    caption="Ground Truth")
        wandb.log(log_base)

        # 3. Solve
        print(f"\n=== Run: sigma={sigma:.5g}, step={step} ===")
        t0 = time.perf_counter()
        x_res, KL_vec, iter_err, norm2, psnr_vec, ssim_vec = pnp_ism(
            noise_image, dataset['back_vec'], parameters, device
        )
        exec_time = time.perf_counter() - t0
        print(f"Tempo di esecuzione: {exec_time:.4f} s | PSNR finale = {psnr_vec[-1].item():.4f}")

        # 4. Curve per-iterazione
        for k in range(len(psnr_vec)):
            wandb.log({
                "psnr": psnr_vec[k].item(),
                "ssim": ssim_vec[k].item(),
                "data_fid": KL_vec[k].item(),
                "rel_err": iter_err[k].item(),
                "norm2": norm2[k].item(),
            })

        # 5. Metriche di sintesi (quella ottimizzata dallo sweep)
        best_idx = int(torch.argmax(psnr_vec).item())
        run.summary["best_psnr"] = psnr_vec[best_idx].item()
        run.summary["best_ssim"] = ssim_vec[best_idx].item()
        run.summary["best_iter"] = best_idx
        run.summary["execution_time_seconds"] = exec_time

        # 6. Ricostruzione migliore
        rec = x_res.detach().cpu().squeeze().numpy()
        rec = rec / (rec.max() + 1e-12)
        wandb.log({"Reconstruction": wandb.Image(cm.hot(rec),
                                                 caption=f"best@{best_idx}")})


# ==========================================
# 3. ESECUZIONE DELL'AGENTE
# ==========================================
# Lancia run_experiment per ogni combinazione (sigma, step) della grid.
wandb.agent(sweep_id, function=run_experiment)
# %%
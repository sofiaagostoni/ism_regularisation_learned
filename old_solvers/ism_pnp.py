# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import deepinv as dinv
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
        
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from scipy.optimize import least_squares

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

## GENERATE DATA ---------
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tv= TVLoss()


hparams = {
    'Nz': 1,
    'pxsize': 40,
    'IS_REAL': False,
    'LOAD_FROM_FILE': True,
    'flux': 40,
    'lam': 0.001
}

# Aggiunta dei parametri dipendenti
hparams['IS_3D'] = (hparams['Nz'] > 1)
hparams['real_name'] = '01_tomm20' if hparams['IS_REAL'] else 'tubulin'
hparams['path'] = 'Data/Simul_data/tub_3D.pth' if hparams['IS_3D'] else 'Data/Simul_data/tub_level.pth'


dataset = prepare_ism_data(
    is_real = hparams['IS_REAL'],
    real_name= hparams['real_name'],
    load_path = None if not hparams['LOAD_FROM_FILE'] else hparams['path'],
    phantom_type= hparams['real_name'],
    Nx = 256, Ny = 256, 
    Nz = hparams['Nz'] , 
    pxsize = hparams['pxsize'], 
    flux = hparams['flux'],
    device = device,
    show_plots = True,
    normalization_y = False
)

noise_image = dataset['noise_image']

kl = KL(back=dataset["back_vec"])

# drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,  pretrained="download", device = device)

# chris_net=dinv.models.DnCNN(depth=5,in_channels=1,out_channels=1, pretrained = None).to(device)
# checkpoint = torch.load('best_model_checkpoint_IIT_flux_40.pth',weights_only=False,map_location=device)
# state_dict = checkpoint['model_state_dict']


# # nuovo dict filtrato e con chiavi rinominate
# new_state_dict = {}
# for k, v in state_dict.items():
#     if k.startswith("physics."):
#         # salta le chiavi di physics
#         continue
#     if k.startswith("network.net."):
#         # rimuovi il prefisso "network."
#         new_k = k[len("network.net."):]
#     else:
#         new_k = k
#     new_state_dict[new_k] = v
# # carica lo state_dict sistemato
# chris_net.load_state_dict(new_state_dict, strict=True)
    
noise_image = dataset["noise_image"]
finger_print = dataset["fingerprint"]
physics = dataset["physics"]

# Initial vector
x_0 = dataset['x_init']
# x_0 = noise_image.sum(0)

x_0 = (x_0 / x_0.max())

    
for i in range(25):
    max_y_i = torch.max(noise_image[i])
    print(f"before normalization {noise_image[i].max()}")
    noise_image[i] = (noise_image[i] / max_y_i) * finger_print[i]
    print(f"after normalization {noise_image[i].max()}")
    

max_gt = dataset["ground_truth"].max()
sigma = 1e-3




#%%


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

    # early stopping sul psnr: quante iterazioni senza miglioramento tollero
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

    # --- tracciamento del miglior psnr ---
    best_psnr = -float('inf')
    best_x    = x_k_prec.clone()
    best_k    = 0
    no_improve = 0
    # -------------------------------------

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

            # --- early stopping sul psnr ---
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
            # -------------------------------

            # convergenza sull'errore relativo (come prima)
            if iter_err[k] < tollerance:
                print(f"Convergence reached at iteration = {k}")
                funct    = funct[0:k + 1]
                iter_err = iter_err[0:k + 1]
                norm2    = norm2[0:k + 1]
                psnr_vec = psnr_vec[0:k + 1]
                ssim_vec = ssim_vec[0:k + 1]
                break

            x_k_prec = x_k_succ

    # restituisco l'iterato col PSNR migliore, non l'ultimo
    return best_x, funct.detach(), iter_err.detach(), norm2.detach(), psnr_vec.detach(), ssim_vec.detach()






# ALGORITHM
# bm3d = dinv.models.BM3D() 
# tgv = dinv.models.TGVDenoiser()
# tv = dinv.models.TVDenoiser()
# med_filter = dinv.models.MedianFilter()
# dncnn = dinv.models.DnCNN(in_channels=1, out_channels=1, depth=20, pretrained="download") # sigma not used
drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
                            pretrained= "download", device=device)
# drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
#                             pretrained= None, device=device)
# checkpoint = torch.load("training_drunet/best_model_checkpoint_drunet.pth",
#                         map_location=device)
# drunet.load_state_dict(checkpoint["model_state_dict"])
data_fid_l2 = dinv.optim.data_fidelity.L2()

parameters = {
    "max_iter": 100,
    "tollerance": 1e-12,
    "Lip_reg": dataset["L_th"]*1e-3, 
    "x_init": x_0,
    "physics": dataset["physics"],
    "back": dataset["back_vec"],
    "sigma": sigma,
    "ground_truth": dataset["ground_truth"] if not hparams['IS_REAL'] else None,
    # "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
    "data_fid": kl.forward_25_3D if hparams['IS_3D'] else data_fid_l2,
    "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
    "single_data_fid": KL_metric if hparams['IS_3D'] else KL_metric,
    "Pnp" : drunet,
}


parameters['Pnp'] = drunet
#
# L_l2 = physics.compute_norm(x_0, tol=1e-4)   # stima ‖AᵀA‖ (se physics è una LinearPhysics deepinv)
L = torch.zeros(25)
for j in range(0,25):
    x_ones = torch.ones_like(x_0.repeat(25,1,1,1))
    norm_H = torch.max(physics(x_ones)[j]) * torch.max(physics.A_adjoint(x_ones)[j])
    norm_y = torch.max(torch.abs(noise_image[j]))
    L[j] = norm_H
L_th_l2 = torch.sum(L)
parameters['Lip_reg'] = L_th_l2

# MULTIPLE SIGMA -------------------
sigma_list = torch.linspace(1e-5, 1e-3, steps=80)
max_iterlist = [500]
step_increment = [1]

# sigma_list = [0.01]

for sigma in sigma_list:
    for maxiter in max_iterlist:
        for step in step_increment:
            # print(f'sigma = {sigma}')
            # print(f'maxiter = {maxiter}')
            # print(f'step = {step}')

            parameters['sigma'] = sigma
            parameters['max_iter'] = maxiter
            
            print('L2 NORM FID')
            parameters['Lip_reg'] = L_th_l2*step
            parameters['data_fid'] = kl.forward_25_3D if hparams['IS_3D'] else data_fid_l2
            x_result_drunet, KL_vec_drunet, iter_drunet, norm2_drunet, psnr_vec, ssim_vec = pnp_ism_l2(
                noise_image, dataset['back_vec'], parameters, device
            )
            print(f'PSNR {psnr_vec[-1].item()}')
            
            # Convergence curve — its own figure
            plt.figure()
            plt.plot(psnr_vec.cpu())
            plt.title(f'psnr  (sigma={sigma}, maxiter={maxiter})')
            plt.show()

            # Result image — its own figure
            plot([x_result_drunet.to("cpu"), noise_image.sum(0).to("cpu"), parameters['ground_truth'].to('cpu')], cmap='hot', figsize = (10,5))
            plt.show()
            
            
                        
            print('KL FID')
            parameters['Lip_reg'] = dataset["L_th"]*1e-3
            parameters['data_fid'] = kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25
            x_result_drunet, KL_vec_drunet, iter_drunet, norm2_drunet, psnr_vec, ssim_vec = pnp_ism(
                noise_image, dataset['back_vec'], parameters, device
            )
            print(f'PSNR {psnr_vec[-1].item()}')

            # Convergence curve — its own figure
            plt.figure()
            plt.plot(psnr_vec.cpu())
            plt.title(f'psnr  (sigma={sigma}, maxiter={maxiter})')
            plt.show()

            # Result image — its own figure
            plot([x_result_drunet.to("cpu"), noise_image.sum(0).to("cpu"), parameters['ground_truth'].to('cpu')], cmap='hot', figsize = (10,5))
            plt.show()
            # gr.ShowImg(x_result_drunet.to("cpu"), hparams['pxsize']*1e-3)

x_res = x_result_drunet*max_gt
# gr.ShowImg(x_res.cpu(), 40*1e-3)
microssim = micro_structural_similarity(parameters['ground_truth'].squeeze().detach().cpu().numpy().astype(np.float32), x_result_drunet.squeeze().detach().cpu().numpy().astype(np.float32))
#%%
# results = {'x_result': x_result_drunet, 'funct': KL_vec_drunet, 'iter_err': iter_drunet,
#                 'diff_fid': None if hparams['IS_REAL'] else KL_vec_drunet,
#                 'norm2': None if hparams['IS_REAL'] else norm2_drunet,
#                 'psnr': None if hparams['IS_REAL'] else psnr_vec,
#                 'ssim': None if hparams['IS_REAL'] else ssim_vec}

# plot_results(results, dataset, hparams['IS_REAL'], hparams['IS_3D'], hparams['pxsize'], x0_sec = 100, y0_sec = 100)


# plot([ dataset['noise_image'].sum(0), x_result_drunet],
#       cmap = 'hot',
#       rescale_mode = 'clip')
# gr.ShowImg(x_result_drunet.to("cpu"), hparams['pxsize']*1e-3)  

# plot_met(KL_vec_drunet.cpu(), iter_drunet.cpu(),)
# %%

"""
Training di DRUNet da zero su phantom ISM (versione rivista).

Cambiamenti rispetto alla versione precedente:
  - sigma diverso per ogni immagine del batch (condizionamento migliore)
  - augmentation: crop casuali + 8 flip/rotazioni
  - training a iterazioni: warmup lineare + cosine, scheduler.step() a ogni batch
  - gradient clipping
  - EMA dei pesi: validazione e checkpoint sul modello EMA
  - rumore di validazione FISSO -> metriche confrontabili tra validazioni
  - log immagini (noisy / out / gt / |err|) + guadagno di PSNR sull'input
  - checkpoint "best" (EMA) + "last" (per riprendere il training)
  - cartella di output unica per run, creata PRIMA di wandb.init
"""

#%%
import os
import time
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from opt_functions.losses import *

import wandb
from matplotlib import cm

import deepinv as dinv
from deepinv.loss.metric import SSIM, PSNR

from ISM.simulation.generate_ism_phantom import *

#%%
# ============================================================================
# 1. SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# ---- dati ----
Nx = 256
pxsizex = 25.0
n_train, n_val = 1000, 300
CACHE = "training_drunet/training_dataset/varied_dataset.pth"

# ---- training (tutto in ITERAZIONI, non epoche) ----
total_iters  = 100_000     # budget totale; scalalo in base a it/s (vedi log)
warmup_iters = 2_000       # warmup lineare del learning rate
val_every    = 1_000       # iterazioni tra due validazioni
log_every    = 20          # iterazioni tra due log di batch su wandb
img_every    = 5_000       # iterazioni tra due log di immagini

batch_size = 32
patch      = 128           # crop casuale; multiplo di 8 (richiesto da DRUNet)
lr         = 2e-4
lr_min     = 1e-6
grad_clip  = 1.0
ema_decay  = 0.999
use_amp    = False         # True = bf16, ~2x piu' veloce; verifica che non
                           # peggiori il PSNR a sigma=0.01 prima di usarlo
val_bs     = 16

# ---- rumore ----
sigma_min, sigma_max = 0.01, 0.1
sigma_test = [0.01, 0.05, 0.1]
sigma_vis  = 0.05

# ---- modello / loss ----
pretrained = None          # None = da zero; "download" = fine-tuning
loss_name  = 'L1'

LOSS_WEIGHTS = {
    'L2':          dict(w_l1=0.0, w_ffl=0.0, w_l2=1.0, w_cldice=0.0),
    'L1':          dict(w_l1=1.0, w_ffl=0.0, w_l2=0.0, w_cldice=0.0),
    'L1_FFL':      dict(w_l1=1.0, w_ffl=1e2, w_l2=0.0, w_cldice=0.0),
    'L2_FFL':      dict(w_l1=0.0, w_ffl=1.0, w_l2=1.0, w_cldice=0.0),
    'FFL':         dict(w_l1=0.0, w_ffl=1.0, w_l2=0.0, w_cldice=0.0),
    'L1_FFL_DICE': dict(w_l1=1.0, w_ffl=1e2, w_l2=0.0, w_cldice=1.0),
}

config = {
    "model": "DRUNet",
    "pretrained": pretrained,
    "Nx": Nx, "pxsizex": pxsizex,
    "n_train": n_train, "n_val": n_val,
    "total_iters": total_iters, "warmup_iters": warmup_iters,
    "batch_size": batch_size, "patch": patch,
    "lr": lr, "lr_min": lr_min,
    "optimizer": "Adam",
    "scheduler": "warmup+cosine (per iteration)",
    "grad_clip": grad_clip, "ema_decay": ema_decay, "amp_bf16": use_amp,
    "augmentation": "random crop + dihedral (8)",
    "noise": "Gaussian, per-image sigma",
    "sigma_min": sigma_min, "sigma_max": sigma_max, "sigma_test": sigma_test,
    "loss": loss_name, **LOSS_WEIGHTS[loss_name],
    "seed": random_seed,
}


run_tag = (f"{'scratch' if pretrained is None else 'finetune'}_{loss_name}_"
           f"{time.strftime('%Y%m%d-%H%M%S')}")
out_dir = os.path.join("training_drunet_newloss", run_tag)
os.makedirs(out_dir, exist_ok=False)

ckpt_path    = os.path.join(out_dir, "best_model.pth")     # pesi EMA migliori
last_path    = os.path.join(out_dir, "last.pth")           # per riprendere
metrics_path = os.path.join(out_dir, "training_metrics.pth")
print(f"Output in: {out_dir}")

USE_WANDB = True
run = wandb.init(
    entity="rp_samplers_ism",
    project="drunet_training",
    name=run_tag,
    config=config,
    save_code=True,
    mode="online" if USE_WANDB else "disabled",
)

#%%
# ============================================================================
# 2. DATASET
# ============================================================================

def to_wandb_img(t, caption=""):
    """tensore (1,H,W) o (H,W) -> wandb.Image RGBA uint8."""
    arr = t.detach().float().cpu().squeeze().numpy()
    arr = np.clip(arr, 0, None)
    arr = arr / (arr.max() + 1e-8)
    rgba = (cm.hot(arr) * 255).astype(np.uint8)
    return wandb.Image(rgba, caption=caption)

if os.path.exists(CACHE):
    print("Carico dataset dalla cache...")
    data = torch.load(CACHE)
else:
    print("Genero il dataset (una tantum)...")
    data = pregenerate_dataset(n_train + n_val, Nx, pxsizex,
                               seed=random_seed, path=CACHE)

data = data.float()
assert data.dim() == 4 and data.shape[1] == 1, f"shape inattesa: {tuple(data.shape)}"
if data.min() < 0 or data.max() > 1.0 + 1e-6:
    raise ValueError(
        f"Dati in [{data.min():.3g}, {data.max():.3g}]: sigma e PSNR assumono [0,1]. "
        "Normalizza il dataset (stessa scala per tutte le immagini).")

training_set   = data[:n_train].to(device)
validation_set = data[n_train:n_train + n_val].to(device)
print(f"Training set:   {tuple(training_set.shape)}")
print(f"Validation set: {tuple(validation_set.shape)}")

wandb.config.update({
    "data_min": float(data.min()),
    "data_max": float(data.max()),
    "data_mean": float(data.mean()),
    "data_zero_frac": float((data == 0).float().mean()),
})

wandb.log({"dataset/examples": [to_wandb_img(training_set[i], f"#{i}")
                                for i in range(8)]}, step=0)


def augment(x, patch):
    """Crop casuale (stessa posizione per tutto il batch) + una delle 8
    trasformazioni diedrali."""
    _, _, H, W = x.shape
    if patch < H or patch < W:
        i = random.randint(0, H - patch)
        j = random.randint(0, W - patch)
        x = x[..., i:i + patch, j:j + patch]
    x = torch.rot90(x, random.randint(0, 3), dims=(-2, -1))
    if random.random() < 0.5:
        x = torch.flip(x, dims=(-1,))
    return x.contiguous()


# ---- rumore di validazione FISSO: generato una volta sola ----
g = torch.Generator(device=device).manual_seed(0)
val_noisy = torch.stack([
    validation_set + s * torch.randn(validation_set.shape, generator=g, device=device)
    for s in sigma_test
])                                                   # (S, N, 1, H, W)

ssim = SSIM()
psnr = PSNR()

psnr_input = np.array([
    psnr(val_noisy[k], validation_set).mean().item() for k in range(len(sigma_test))
])
for k, s in enumerate(sigma_test):
    print(f"Baseline input sigma={s}: PSNR {psnr_input[k]:.2f} dB")
    run.summary[f"input_PSNR_sigma_{s}"] = float(psnr_input[k])

# immagini fisse per la visualizzazione
vis_clean = validation_set[:4]
vis_noisy = vis_clean + sigma_vis * torch.randn(
    vis_clean.shape, generator=g, device=device)

#%%
# ============================================================================
# 3. MODELLO, OTTIMIZZATORE, LOSS
# ============================================================================

model = dinv.models.DRUNet(in_channels=1, out_channels=1,
                           pretrained=pretrained).to(device)
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters),
        CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=lr_min),
    ],
    milestones=[warmup_iters],
)

criterion = ScaleBalancedDenoisingLoss(**LOSS_WEIGHTS[loss_name])

amp_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=use_amp and device.type == "cuda")


@torch.no_grad()
def validate(net):
    """Metriche su rumore fisso, in float32."""
    net.eval()
    S = len(sigma_test)
    ssim_v, psnr_v = np.zeros(S), np.zeros(S)
    for k, s in enumerate(sigma_test):
        for i in range(0, n_val, val_bs):
            x = validation_set[i:i + val_bs]
            y = val_noisy[k, i:i + val_bs]
            out = net(y, sigma=s)
            ssim_v[k] += ssim(out, x).sum().item()
            psnr_v[k] += psnr(out, x).sum().item()
    net.train()
    return ssim_v / n_val, psnr_v / n_val


#%%
# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

loss_hist, SSIM_val_vett, PSNR_val_vett, val_iters = [], [], [], []
best_val_PSNR = -np.inf
running_loss = torch.zeros((), device=device)
n_running = 0

model.train()
start_time = time.time()
t_last = start_time

for it in range(1, total_iters + 1):

    # ---------------- BATCH ----------------
    idx = torch.randint(0, n_train, (batch_size,), device=device)
    clean = augment(training_set[idx], patch)

    sigma = torch.empty(batch_size, device=device).uniform_(sigma_min, sigma_max)
    noisy = clean + sigma.view(-1, 1, 1, 1) * torch.randn_like(clean)

    with amp_ctx():
        out = model(noisy, sigma=sigma)
    loss, parts, weights = criterion(out.float(), clean)   

    # ---------------- STEP ----------------
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()

    if it >= warmup_iters:
        ema_model.update_parameters(model)

    running_loss += loss.detach()
    n_running += 1

    # ---------------- LOG BATCH ----------------
    if it % log_every == 0:
        log_dict = {
            "train/batch_loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for name, w in weights.items():
            if float(w) > 0.0:
                log_dict[f"train/batch_loss_{name}"] = float(w) * parts[name].item()
        wandb.log(log_dict, step=it)

    # ---------------- VALIDAZIONE ----------------
    if it % val_every == 0 or it == total_iters:
        avg_loss = (running_loss / n_running).item()
        running_loss.zero_()
        n_running = 0
        loss_hist.append(avg_loss)

        eval_net = ema_model if it >= warmup_iters else model
        ssim_vals, psnr_vals = validate(eval_net)
        ssim_mean, psnr_mean = float(ssim_vals.mean()), float(psnr_vals.mean())
        SSIM_val_vett.append(ssim_mean)
        PSNR_val_vett.append(psnr_mean)
        val_iters.append(it)

        now = time.time()
        it_per_s = val_every / (now - t_last)
        t_last = now

        log_dict = {
            "train/loss": avg_loss,
            "val/SSIM": ssim_mean,
            "val/PSNR": psnr_mean,
            "time/it_per_s": it_per_s,
        }
        for k, s in enumerate(sigma_test):
            log_dict[f"val/SSIM_sigma_{s}"] = float(ssim_vals[k])
            log_dict[f"val/PSNR_sigma_{s}"] = float(psnr_vals[k])
            log_dict[f"val/PSNR_gain_sigma_{s}"] = float(psnr_vals[k] - psnr_input[k])

        # immagini: noisy / out / gt / |err|
        if it % img_every == 0 or it == total_iters:
            eval_net.eval()
            with torch.no_grad():
                vis_out = eval_net(vis_noisy, sigma=sigma_vis)
            eval_net.train()
            err = (vis_out - vis_clean).abs()
            log_dict["val/images"] = [
                to_wandb_img(t, f"{c} {i}") for i in range(4)
                for t, c in [(vis_noisy[i], "noisy"), (vis_out[i], "out"),
                             (vis_clean[i], "gt"), (err[i], "|err|")]
            ]

        wandb.log(log_dict, step=it)

        eta_min = (total_iters - it) / it_per_s / 60
        print(f"[it {it}/{total_iters}]  loss {avg_loss:.5f}  "
              f"SSIM {ssim_mean:.4f}  PSNR {psnr_mean:.3f}  "
              f"({it_per_s:.1f} it/s, ETA {eta_min:.0f} min)")
        for k, s in enumerate(sigma_test):
            print(f"   sigma={s}: SSIM {ssim_vals[k]:.4f} | PSNR {psnr_vals[k]:.3f} "
                  f"(+{psnr_vals[k] - psnr_input[k]:.2f} dB sull'input)")

        # ---------------- CHECKPOINT ----------------
        net_to_save = ema_model.module if it >= warmup_iters else model
        if psnr_mean > best_val_PSNR:
            best_val_PSNR = psnr_mean
            torch.save({
                "model_state_dict": net_to_save.state_dict(),  # caricabile in DRUNet
                "iteration": it,
                "val_PSNR": psnr_mean,
                "val_SSIM": ssim_mean,
                "config": config,
            }, ckpt_path)
            print(f"   -> nuovo best model salvato (PSNR {psnr_mean:.4f})")
            run.summary["best_PSNR"] = psnr_mean
            run.summary["best_SSIM"] = ssim_mean
            run.summary["best_iter"] = it

        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "iteration": it,
            "best_val_PSNR": best_val_PSNR,
        }, last_path)

#%%
# ============================================================================
# 5. FINE
# ============================================================================

train_time = time.time() - start_time
print(f"Training time: {train_time / 60:.1f} min")
run.summary["train_time_s"] = train_time

torch.save({
    "val_iters": val_iters,
    "loss_train_vett": loss_hist,
    "SSIM_val_vett": SSIM_val_vett,
    "PSNR_val_vett": PSNR_val_vett,
}, metrics_path)

if USE_WANDB:
    artifact = wandb.Artifact(f"drunet-{run.id}", type="model",
                              metadata={"best_PSNR": best_val_PSNR, **config})
    artifact.add_file(ckpt_path)
    run.log_artifact(artifact)

wandb.finish()
print(f"Training complete. Output in {out_dir}")
#%%
"""Run the full grid of experiments. Safe to restart: finished runs are skipped."""
import gc
import itertools
import os
import time
import traceback

import matplotlib.pyplot as plt
import torch
import deepinv as dinv
from tqdm.auto import tqdm

from opt_functions.data_preparation import load_data, make_measurements
from opt_functions.Solver_functions.prior_config import get_denoiser
from opt_functions.Solver_functions.pnp_utils import Config, run_experiment, print_search_plan

import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.utils import load_example

import ISM.analysis.Graph_lib as gr

# device = dinv.utils.get_device()

# x = load_example("CBSD_0010.png", grayscale=True).to(device)

# x = torch.tensor(x, device=device, dtype=torch.float)
# H, W = x.shape[-2:]                                  # center crop to 256 x 256
# assert H >= 256 and W >= 256, f"image is {H}x{W}, smaller than the 256x256 crop"
# x = x[..., (H - 256) // 2:(H - 256) // 2 + 256, (W - 256) // 2:(W - 256) // 2 + 256]
# img_size = x.shape[1:]
# # Set the global random seed from pytorch to ensure reproducibility of the example.
# torch.manual_seed(0)

# torch.save(x, "Castle_gray.pth")

#%%
# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
# Set every grid the method searches; the others are fixed automatically
# (a warning tells you if you set one that is not used):
#
#   prior              algo      searched (backtracking on)   fixed
#   Drunet(_finetune)  PGD       sigma, step, iter            lambda
#   weakly_convex      PROJ_GD   sigma, lambda, iter          step
#   TV_smooth          PROJ_GD   lambda, iter                 sigma, step
#   L1                 PROX_GD   lambda, iter                 sigma, step
#
# With backtracking="none", step is searched too for the explicit priors.
# step_grid multiplies the base step tau = 1 / (L_f * scale),
# scale = kl_step_scale (KL) or l2_step_scale (L2).
# iter_grid is free: one run of max(iter_grid) records every value.

PLOT_SETUP = True      # plot ground truth, PSFs and measurements before running
PSF_CROP = 64          # pixels shown around the PSF center (None = full PSF)

# --- WCRR -------------------------------------------------------------------
cfg = Config(
    prior_name="WCRR",   #WCRR_softplus_ISM
    save_dir="Results/results_gridsearch",
    is_3d=False,
    Nz=1,
    init_choice="adjoint_y",
    kl_step_scale=1e-1,
    l2_step_scale=1,
    sigma_grid=torch.logspace(0.5, 2, 10),     # searched
    lam_grid=torch.logspace(-2,0 , 20),         # searched
    iter_grid=torch.tensor([5000]),           # one value: run length
    backtracking="custom",                    # step adapted -> not searched
)
algo_list = ["PROJ_GD"]
loss_list = [ "L2"]                            # WCRR ignores the loss: keep one entry

# --- Smoothed TV ------------------------------------------------------------
# cfg = Config(prior_name="TV_smooth", tv_eps=1e-2, is_3d=True, Nz=2,
#              save_dir="Results/results_gridsearch", kl_step_scale=1e-3, l2_step_scale=1e1,
#              lam_grid=torch.logspace(-3, 1, 9), iter_grid=torch.tensor([5000]))
# algo_list = ["PROJ_GD"]
# loss_list = ["L2"]

# --- L1 ---------------------------------------------------------------------
# cfg = Config(prior_name="L1", is_3d=True, Nz=2,
#              save_dir="Results/results_gridsearch", kl_step_scale=1e-3, l2_step_scale=1e1,
#              lam_grid=torch.logspace(-4, 0, 9), iter_grid=torch.tensor([5000]))
# algo_list = ["PROX_GD"]
# loss_list = ["L2"]

# --- DRUNet -----------------------------------------------------------------
# cfg = Config(prior_name="Drunet", save_dir="Results/results_gridsearch",
#              kl_step_scale=1e-1, l2_step_scale=1e3,is_3d=False,Nz=1,
#              sigma_grid=torch.logspace(-3, 0, 25),
#              step_grid=torch.logspace(1, 4, 7),
#              iter_grid=torch.linspace(10, 400, 30).round().long())

# algo_list = ["PGD"]
# loss_list = ["L2"]             # Drunet_finetune: one checkpoint per loss

data_list = ["tub_balls"]
flux_list = [5, 10, 20, 40, 60]
fidelity_list = ["L2"]


#%%
# ---------------------------------------------------------------------------
# Plots of the simulation setup
# ---------------------------------------------------------------------------
def load_data(name, cfg):
    """Load cfg.data_dir/<name>.pth and normalize it to [0, 1].
    3D: each plane is normalized separately, then stacked as channels."""
    x = torch.load(os.path.join(cfg.data_dir, f"{name}.pth"), map_location=cfg.device)
    if cfg.is_3d:
        return torch.cat([x[0] / x[0].max() + 1e-5, x[1] / x[1].max() + 1e-5], dim=1)
    else:
        return x / x.max() + 1e-5
    
def plot_setup(x, meas, cfg):
    """Ground truth, PSFs, fingerprint and measurements of one simulation."""
    if cfg.is_3d:
        PSF = meas["PSF"].reshape(25, -1, *meas["PSF"].shape[-2:])     # (25, planes, k, k)
        y_tensor = torch.cat(list(meas['y']), dim=0)    # (25, 1, 256, 256)
        gr.ShowDataset(PSF[:, 0:1].cpu(), normalize=False)
        gr.ShowDataset(PSF[:, 1:2].cpu(), normalize=False)
        gr.ShowImg(x[:, 0:1].to("cpu"), cfg.pxsizex*1e-3)
        gr.ShowImg(x[:, 1:2].to("cpu"), cfg.pxsizex*1e-3)
        gr.ShowDataset(y_tensor.cpu(), normalize=False)
        gr.ShowImg(y_tensor.sum(0).unsqueeze(0).cpu(), cfg.pxsizex*1e-3)
    else:
        PSF = meas["PSF"]
        y_tensor = torch.cat(list(meas['y']), dim=0)    # (25, 1, 256, 256)
        gr.ShowDataset(PSF.cpu(), normalize=False)
        gr.ShowImg(x.unsqueeze(1).to("cpu"), cfg.pxsizex*1e-3)
        gr.ShowDataset(y_tensor.cpu(), normalize=False)
        gr.ShowImg(y_tensor.sum(0).unsqueeze(0).cpu(), cfg.pxsizex*1e-3)


#%%
# Look at one ground truth (any name in cfg.data_dir), linear and log scale
show_name = "circle_phantom"
x_show = load_data(show_name, cfg)[0, 0].cpu()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, im, label in zip(axes, (x_show, x_show.log10()), ("linear", "log10")):
    fig.colorbar(ax.imshow(im, cmap="hot"), ax=ax)
    ax.set_title(f"{show_name} ({label})")
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
plt.show()


#%%
if PLOT_SETUP:
    x = load_data(data_list[0], cfg)
    plot_flux = 5
    # for flux in flux_list[:1]:                 # change to flux_list to see all
    plot_setup(x, make_measurements(x, plot_flux, cfg), cfg)


#%%
# Check what will be searched before starting anything
for algo in algo_list:
    print_search_plan(cfg, algo)
n_exp = len(data_list) * len(flux_list) * len(loss_list) * len(fidelity_list) * len(algo_list)
print(f"\n{n_exp} experiments (data x flux x loss x fidelity x algo)")


#%%
# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def log_failure(msg):
    tqdm.write(msg)
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "failed.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


if __name__ == "__main__":
    outer = tqdm(total=n_exp, desc="experiments")
    for name in data_list:
        x = load_data(name, cfg)
        for flux in flux_list:
            meas = make_measurements(x, flux, cfg)
            for loss in loss_list:
                denoiser = get_denoiser(loss, cfg)
                for fidelity_name, algo in itertools.product(fidelity_list, algo_list):
                    try:
                        run_experiment(cfg, name, flux, loss, fidelity_name, algo=algo,
                                       x=x, meas=meas, denoiser=denoiser)
                    except Exception:
                        log_failure(f"FAILED {name} flux={flux} loss={loss} "
                                    f"fidelity={fidelity_name} algo={algo}\n"
                                    f"{traceback.format_exc()}")
                    finally:
                        outer.update(1)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                del denoiser
    outer.close()


#%%
# ---------------------------------------------------------------------------
# Single experiment: arguments override the cfg grids (one value = no search)
# ---------------------------------------------------------------------------
# cfg.prior_name = "Drunet_fromscratch"
# cfg.kl_step_scale = 1e-1
# res = run_experiment(cfg, "tub_level", 20, "L1", "KL", algo="PGD",
#                      sigmas=[0.001, 0.005, 0.01, 0.1], lambdas=[10.0], iters=[10,20,30,50],
#                      save=False, verbose=True)

# if res["metrics"] is not None:
#     dinv.utils.plotting.plot_curves(res["metrics"])
    
# x_rec = res["x_rec"]
# gr.ShowImg(x_rec.unsqueeze(1).to("cpu"), cfg.pxsizex*1e-3)

# step size chosen by the backtracking

#%%
# reconstruction vs ground truth, one row per plane
# x_rec = res["x_rec"]
# gr.ShowImg(x_rec[:, 0].unsqueeze(1).to("cpu"), cfg.pxsizex*1e-3)
# gr.ShowImg(x_rec[:, 1].unsqueeze(1).to("cpu"), cfg.pxsizex*1e-3)

# %%

#%%
# ---------------------------------------------------------------------------
# Single experiment: arguments override the cfg grids (one value = no search)
# ---------------------------------------------------------------------------
cfg.prior_name = "WCRR"
cfg.kl_step_scale = 1e-2
cfg.l2_step_scale = 1e-1
res = run_experiment(cfg, "circle_phantom", 20, "L2", "KL", algo="Proj_GD",
                     sigmas=[100], lambdas=[10], iters=[2000],
                     save=True, verbose=True)

if res["metrics"] is not None:
    dinv.utils.plotting.plot_curves(res["metrics"])
    
x_rec = res["x_rec"]
gr.ShowImg(x_rec.unsqueeze(1).to("cpu"), cfg.pxsizex*1e-3)
#%%
if res["L_history"]:
    plt.semilogy(1 / torch.tensor(res["L_history"]))
    plt.xlabel("iteration")
    plt.ylabel("step size tau")
    plt.show()
# %%

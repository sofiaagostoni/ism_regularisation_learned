#%%
"""
Interactive cells (VS Code / Jupyter) for the simulated results: 2D tables and figures,
KL vs L2, no regularization / Richardson-Lucy, 3D. Run cell by cell; the first cell loads
the tables. Functions: opt_functions/results_lib/ (formerly compare_results.py).
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
import ISM.analysis.Graph_lib as gr

from opt_functions.results_lib.common import *               # noqa: F401,F403
from opt_functions.results_lib.sim import *                  # noqa: F401,F403
from opt_functions.results_lib.sim_figures import *          # noqa: F401,F403
from opt_functions.results_lib.sim_diagnostics import *      # noqa: F401,F403

#%%
# ===== 2D: tables, reconstructions, diagnostics =====
# split_by_fidelity=False puts KL and L2 runs in the same tables
# search="adaptive": only the current evaluate_methods.py runs ("grid": the old ones, None: all)
# + the single_experiment.py runs (include_single=False to leave them out)
df = load(ROOT, split_by_fidelity=False, search="adaptive")
best = best_per_method(df)
DATA = "tub_balls"                            # or "tub_level" for the 2D results
PLANE = 0                                  # 3D: plane shown in the images (None = max proj.)

#%%
# Heatmap tables: PSNR / SSIM / MicroSSIM, methods x flux
plot_metric_heatmaps(best, DATA)

#%%
# Metrics as a function of the flux; relative_to shows the gain over a baseline
plot_metric_vs_flux(best, DATA)            # e.g. relative_to="WCRR ["

#%%
# Colour-coded table (VS Code interactive window / Jupyter)
display(styled_table(best, DATA))

#%%
print_params(best)

#%%
# Reconstructions + power spectra for one flux (methods=None: all methods)
plot_reconstructions(best, DATA, 20,
                     methods= ["WCRR_ISM [PROJ_GD] KL", "WCHS-free [PROJ_GD] KL", "WCHS-free (SMLM) [PROJ_GD] KL", 'DRUNet-ft (L2) [PGD] KL'],         # e.g. ["WCRR", "TV-smooth", "L1-norm"]
                     plane=PLANE,
                     crop=None,            # e.g. (64, 192, 64, 192) to zoom
                     normalize=False,      # True: each image / its max (as the metrics)
                     shared_scale=True)    # False: one colorbar per panel

#%%
# All images at one flux: rows = images, columns = ground truth + methods
plot_all_images(best, 20)                  # e.g. methods=["WCRR", "TV-smooth"]

#%%
# One method: best parameters and reconstructions
show_method(best, DATA, "WCRR_ISM [PROJ_GD] KL", flux=5, plane=PLANE)
# show_method(best, DATA, "TV-smooth [PROJ_GD] KL")              # all fluxes
# show_method(best, DATA, "L1-norm [PROX_GD] L2", flux=[5, 20])

#%%
# One image by sample + method, to plot yourself (method names: best.method.unique())
img,m = get_image(best, DATA, "WCHS-free [PROJ_GD] KL", flux=20, return_metrics=True)      # kind="gt" / "sum_y"
# with the metrics: img, m = get_image(..., return_metrics=True); m["psnr"], m["ssim"], m["microssim"]
gt = get_image(best, DATA, flux=20, kind="sum_y")
# vmin / vmax must be tensors (ShowImg uses torch.floor on them); same range = comparable
fig, ax = gr.ShowImg(img, 0.04, clabel="photons", vmin=torch.tensor(0.), vmax=gt.max())
# ax.set_title("WCRR_ISM")
plt.show()

#%%
# Convergence of the functional F(x_k) = f + lam R (explicit priors only)
# plot_functional(best, DATA, 40, kind="gap")            # e.g. methods=["WCRR", "TV-smooth"], rates=(1, 2)
plot_functional(best, DATA, 20, kind="gap", methods = ['DRUNet'])   # F(x_k) itself
#%%
# Bounded iterates, finite length, criticality, step sizes (needs refresh_histories.py)
convergence_check(best, DATA, 20, methods=["WCRR_ISM [PROJ_GD] KL", "WCHS-free [PROJ_GD] KL", "WCHS-free (SMLM) [PROJ_GD] KL", 'DRUNet (L2)'])          # e.g. methods=["WCRR", "TV-smooth"]

#%%
# Points tried by the parameter search of one method (red star = chosen)
plot_search(best, DATA, 20, "WCRR_ISM [PROJ_GD] KL")

#%%
# ===== KL vs L2 (evaluate_methods.py; other folders: add them to SOURCES) =====
# on_duplicate="tag": a method in several folders appears once per folder, tagged
# " {source}" after the first one; on_duplicate="<source>": keep only that one
allres = best_per_method(load_merged(on_duplicate="tag"))
print(allres.groupby(["source", "data"]).method.nunique().unstack(0).fillna(0).astype(int))

#%%
# choose here (None = all)
SEL = dict(data=None,                        # e.g. "balls" or ["tub", "balls"]
           priors=["WCRR_ISM"],   # exact prior names, e.g. "L1", "TV_nonsmooth"
           fidelities=["KL", "L2"],
           flux=20,
           sources=None,                     # keys of SOURCES, None = all
           algos="PROJ_GD")                  # PROJ_GD only: MD (KL0) is another algorithm
sel = pick(allres, **SEL)
ft = fidelity_table(sel)
display(ft.style.format("{:.3f}")
        .background_gradient(cmap="RdBu", subset=[c for c in ft.columns if "-" in c[1]],
                             vmin=-2, vmax=2))

#%%
# images of the selection, one image at a time (KL and L2 side by side, own folders)
DATA_FID = "tub"
plot_reconstructions(sel, DATA_FID, 20, normalize=False, shared_scale=True)
plot_metric_heatmaps(sel, DATA_FID)

#%%
# Intensity profile along one line: GT (with the line) + profiles of GT and methods.
# row= / col= pixel index of the line (default: row with the most GT signal);
# returns {label: profile} for your own plots
prof = plot_profile(allres, "tub", 20, row=32, methods=["KL"])
# plot_profile(allres, "tub", 20, methods=["WCRR [PROJ_GD] KL", "WCRR_ISM [PROJ_GD] KL"], col=128)

#%%
# Paper figure: input + GT + methods (full + zoom) and the profile along one line.
# methods as a list keeps its order; zoom=(y0, y1, x0, x1), row= / col= the line
# e.g. methods=["WCRR_ISM [PROJ_GD] KL", "WCRR_ISM [PROJ_GD] L2"],
#      labels={"WCRR_ISM [PROJ_GD] KL": "WCRR (ISM)"}, save="fig.pdf"
plot_comparison_figure(best, DATA, 20,
                       zoom=None, row=None, plane=PLANE, methods = ["KL"])

#%%
# KL vs L2 as a function of the flux: metrics, KL solid / L2 dashed; one figure per
# image; priors= exact prior names
# plot_fidelity_vs_flux(allres, ["tub"], algos=["PROJ_GD", "PROX_GD"])
plot_fidelity_vs_flux(allres, "tub", priors=['TV_smooth'], metrics=("psnr", "ssim"))

#%%
# ===== No regularization (λ = 0) and Richardson-Lucy (Results/unregularized) =====
DATA = "tub_balls"

df0 = load(UNREG_DIR, split_by_fidelity=False, search=None)
df0 = df0[df0["lambda"] == 0]          # λ=0 only (the folder may also get the whiteness-λ runs)

# ProjGD, λ = 0
img, m = get_image(df0, DATA, "TV-smooth [PROJ_GD] KL", flux=20, root=UNREG_DIR, return_metrics=True)
fig, ax = gr.ShowImg(img, 0.04, clabel="photons", vmin=torch.tensor(0.), vmax=img.max())
ax.set_title(f"ProjGD, no reg | PSNR {m['psnr']:.2f}")
plt.show()

#%%
rl = torch.load(f"Results/unregularized/RL_{DATA}_flux20_seed0.pth", weights_only=False)
n = rl["best_iter"]                    # or any count you passed with --iter, e.g. 1000
x = rl["snapshots"][n]
img = torch.as_tensor(to_img(x)) * 20  # tensor (ShowImg needs tensors); × flux: photon scale as get_image
m = rl["metrics"][n]
fig, ax = gr.ShowImg(img, 0.04, clabel="photons", vmin=torch.tensor(0.), vmax=img.max())
ax.set_title(f"Richardson–Lucy, {n} it | PSNR {m['psnr']:.2f}")
plt.show()

#%%
df = load(ROOT, split_by_fidelity=False, search="adaptive")      # full table, not best
DATA = "tub_balls"
img, m = get_image(df, DATA, "TV-smooth [PROJ_GD] KL", flux=20, lam=0.046, return_metrics=True)
gr.ShowImg(img, 0.04, clabel="photons", vmin=torch.tensor(0.), vmax=img.max())

#%%
# PSNR per iteration: ProjGD TV-smooth and ProxGD TV at a chosen lambda, + RL
df_all = pd.concat([load(ROOT, split_by_fidelity=False, search="adaptive").assign(root=ROOT),
                    load(UNREG_DIR, split_by_fidelity=False).assign(root=UNREG_DIR)],
                   ignore_index=True)
plot_psnr_iterations(df_all, DATA, 20,
                     lam=None)        # None = best of each method; 0.046 = same for all;
                                      # {"TV-smooth [PROJ_GD] KL": 0, "TV [PROX_GD] KL": 0.01}

#%%
rl = torch.load(f"Results/unregularized/RL_{DATA}_flux20_seed0.pth", weights_only=False)
k = np.arange(1, len(rl["psnr"]) + 1)
fig, ax = plt.subplots(figsize=(7, 4.5))
# ax.plot(k, rl["psnr_clamp"], label="RL, clamped to [0, 1]")
ax.plot(k, rl["psnr"], label="RL", alpha=0.7)
ax.axvline(rl["best_iter"], color="k", ls="--", lw=1, label=f"RL best (x/max): {rl['best_iter']} it")

pg = glob.glob(f"Results/unregularized/single_*/best_rec_FidelityKL_{DATA}_flux20_TV_smooth_lossnone_PROJ_GD.pth")
if pg:                                    # ProjGD λ=0, same clamped PSNR
    r = torch.load(pg[0], map_location="cpu", weights_only=False)
    p = np.asarray(r["metrics"]["psnr"][0], float)
    # ax.plot(np.arange(1, len(p) + 1), p, label="ProjGD λ=0, clamped")

ax.set_xscale("log")
ax.set_xlabel("iteration")
ax.set_ylabel("PSNR (dB)")
ax.set_title(f"{DATA}, flux 20, no regularization")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
plt.show()

#%%
df_all = pd.concat([load(ROOT, split_by_fidelity=False, search="adaptive").assign(root=ROOT),
                    load(UNREG_DIR, split_by_fidelity=False).assign(root=UNREG_DIR)],
                   ignore_index=True)
plot_psnr_iterations(df_all, DATA, 20, lam=0.274, methods="TV-smooth [PROJ_GD] KL")
plot_psnr_iterations(df_all, DATA, 20, lam=None, metric="mse")
plot_psnr_iterations(df_all, "tub_level", 20, lam={"TV-smooth [PROJ_GD] KL": 0.274, "TV [PROX_GD] KL": 0.274}, metric="mse")


#%%
# ===== 3D =====
# 3D results (run_3d_from_2d.py: parameters of the 2D searches; plane 1 = in focus)
best = best_per_method(load(ROOT, split_by_fidelity=False, search="adaptive"))
DATA_3D = "tub_3D"                         # or "tub_nuc_3D"
print_ranking(best[best.data == DATA_3D])

#%%
# Single images, one figure each: input (sum of detectors), GT and every method, for the
# in-focus plane (1) and then the out-of-focus plane (0); per plane the GT and the methods
# share the colour scale (shared_scale=False: one per image). Titles: metrics of that plane
plot_single_images(best, DATA_3D, 20, methods=["WCRR"])                    # e.g. methods=["WCRR", "TV-smooth"]
# plot_single_images(best, DATA_3D, 20, planes=(1,), crop=(64, 192, 64, 192),
#                    save_dir=f"figures/{DATA_3D}")       # one png per image
# plot_reconstructions(best, DATA_3D, 20)                 # old: all panels in one row

#%%
plot_profile(best, DATA_3D, 20, methods=["WCRR"])
plot_metric_heatmaps(best, DATA_3D)                          # methods x flux
plot_fidelity_vs_flux(best, DATA_3D)                         # KL vs L2 across fluxes

#%%
# 3D vs the 2D result the parameters came from, side by side
t3 = pd.read_csv(os.path.join(ROOT, "3d_from_2d_tub.csv"))
display(t3.pivot_table(index="method", columns=["image", "flux"],
                       values=["psnr_3d", "psnr_2d"]).round(2))

#%%
# One image by sample + method, to plot yourself (method names: best.method.unique());
# 3D: plane=1 in focus (default), plane=0 out of focus
img,m = get_image(best, DATA_3D, "L1-norm [PROX_GD] KL", flux=20, return_metrics=True)      # kind="gt" / "sum_y"
# with the metrics: img, m = get_image(..., return_metrics=True); m["psnr"], m["ssim"], m["microssim"]
gt = get_image(best, DATA_3D, flux=20, kind="sum_y")

fig, ax = gr.ShowImg(gt, 0.04, clabel="photons", vmin=torch.tensor(0.), vmax=gt.max())
plt.show()


#%%
"""
Interactive cells (VS Code / Jupyter) for the real data (run_real.py: no ground truth,
compare by eye). Run cell by cell; the first cell loads the runs.
Functions: opt_functions/results_lib/ (formerly compare_results.py).
"""
import os

from IPython.display import display

from opt_functions.results_lib.real import *                 # noqa: F401,F403
from opt_functions.results_lib.real_figures import *         # noqa: F401,F403
from opt_functions.results_lib.real_diagnostics import *     # noqa: F401,F403

#%%
runs = load_real_runs()                    # one row per Results/real_data/<data>/<tag>.pth
display(runs.drop(columns="path"))
REAL = "02_TUB"

#%%
# Raw data (Σ 25 detectors) with several colormaps, one panel each (REAL_CMAPS names or
# any matplotlib colormap); separate=True: one figure per colormap (save -> raw_<cmap>.png)
# plot_real_raw(REAL, cmaps=["magma", "inferno", "hot", "glow_cyan", "glow_green", "viridis"], separate=True)
# plot_real_raw(REAL, cmaps=["gray", "viridis", "cividis"], zoom=(700, 1000, 700, 1000))
# plot_real_raw(REAL, cmaps=["magma", "glow_cyan"], separate=True, save=f"{REAL}_raw.png")

# The 25 detector images in the 5 x 5 SPAD layout (12 = centre); shared_scale=False:
# each image on its own scale; zoom= a region of every detector
plot_real_detectors(REAL, cmap="hot")
# plot_real_detectors(REAL, cmap="glow_cyan", shared_scale=False, zoom=(700, 1000, 700, 1000))

#%%
# Residual whiteness of the methods you choose (python whiteness_real.py first):
# W vs lambda (RL / DRUNet: vs iterations), hollow circle = min W, W ~ 2 = white.
# metrics="confocal" (detectors summed) / "stack" (25 det.); show_z: bias and spread of Z
wr = load_real_whiteness()                 # re-read after a new whiteness_real.py run
plot_real_whiteness(REAL, methods=["Drunet"],metrics="confocal", wr=wr)
# plot_real_whiteness(REAL, methods="TV_smooth", metrics="confocal", show_z=True)
# plot_real_whiteness(REAL, methods=["WCRR_ISM", "WCHS_free"], mode="highpass", crop=None)

#%%
# Raw (sum of the 25 detectors) + every reconstruction of REAL, in-focus plane.
# methods= / lambdas= to select; crop= the run_real.py --crop (None = full image)
plot_real(runs, REAL, zoom=(700, 1000, 700, 1000))
# plot_real(runs, REAL,zoom=(300, 400, 300, 400))
# plot_real(runs, REAL,crop=200)

# plot_real(runs, REAL, methods="TV_smooth", lambdas=[0.002, 0.00708, 0.02], crop=None)
plot_real(runs, REAL, plane=0)                                  # out-of-focus background
# plot_real(runs, REAL, zoom=(100, 200, 100, 200), cmap="glow_cyan", save="real_03.pdf")

#%%
# Publicity figure: diagonal split raw | reconstruction (as publicity_figures.py)
os.makedirs("publicity", exist_ok=True)
# flip=True: reconstruction upper left (03_tomm20 is dense there)
plot_real_split(runs, REAL, method="TV_smooth", lam=0.00708, title="TOMM20 · mitochondria",
                flip=True, save=f"publicity/{REAL}_split_tvsmooth.png")
# plot_real_split(..., region=(60, 60, 300), cmap="glow_cyan")    # a square detail
# plot_real_split(..., gamma=(0.8, 0.6))                          # lift dim reconstructed structures

#%%
# Convergence curves of the real-data runs (F, criticality, relative step, ||z_k||, step
# size) + summary table; runs saved without histories are listed and skipped
plot_real_convergence(runs, REAL, methods="WCRR")
# plot_real_convergence(runs, REAL, lambdas=[0.00708], crop=200, save="conv_real_03.pdf")

#%%
# Richardson-Lucy (run_real.py --method RL): early-stopped RL vs the regularized runs.
# RL keeps its own rows when filtering by lambda; iters= picks RL iteration counts
plot_real(runs, REAL, methods=["RL"])                                  # RL 5 ... 100 it
plot_real(runs, REAL, methods=["TV_smooth", "RL"], lambdas=[0.00708])   # TV at one lambda + all RL
# plot_real_split(runs, REAL, method="RL", iters=20, flip=True)
# plot_real_convergence(runs, REAL, methods=["RL"], iters=[100])     # KL decrease of RL

#%%
# One TV vs one RL, side by side with the raw data (picks: lambda for TV, iterations for RL)
# PICKS = [("RL", 50), ("TV_smooth", 0.00200), ('WCRR', 50), ('WCRR_ISM', 50)]
# method names as in run_real.py: "WCHS_free", "WCHS_free_smlm", "WCRR_ISM" (no plain "WCHS")
PICKS = [('RL', 100), ('WCRR_ISM', (25, 50)), ('WCHS_free', (25, 50)),
         ('WCHS_free_smlm', (25, 50)), ('TV_smooth', 0.002), ('L1', 0.003)]

# plot_real(runs, REAL, picks=PICKS, zoom=(100, 200, 100, 200))
plot_real(runs, REAL, picks=PICKS, inset=(600, 900, 600, 900))

#%%
# ONE result with a zoom on a crop: pick = (method, lambda) / ("RL", iterations) / "raw";
# zoom=(y0, y1, x0, x1) after trim. layout="side" (full + zoom next to it), "inset"
# (zoom in a corner), "zoom" (the crop only); zoom_norm="zoom" renormalises the crop
img, crop_img = plot_real_single(runs, REAL, PICKS[0], zoom=(100, 200, 100, 200))
# plot_real_single(runs, REAL, ("RL", 20), zoom=(100, 200, 100, 200), layout="inset")
# plot_real_single(runs, REAL, "raw", zoom=(100, 200, 100, 200), zoom_norm="zoom")
# plot_real_single(runs, REAL, ("WCRR_ISM", 50), zoom=(300, 400, 300, 400), plane=0,
#                  layout="zoom", save=f"{REAL}_wcrr_zoom.pdf")

#%%
# Same, with a crop of the data in the corner of every panel (inset=(y0, y1, x0, x1),
# after trim) and the profile along a line (profile=(y0, x0, y1, x1)) in a row below.
# inset_loc="lower left" / "upper right"; profile_width=3 averages 3 px across the line
_, prof = plot_real(runs, REAL, picks=PICKS,
                    inset=(100, 200, 100, 200),
                    profile=(130, 100, 130, 160),
                    profile_width=1, profile_norm="max")
# prof: {label: profile} for your own plots

#%%
# PnP DRUNet (run_real.py --method DRUNet --flux auto --sigma ...): one file per
# iteration count. All of them, then one picked: ("DRUNet", n_it), ("DRUNet", (sigma,
# n_it)) or ("DRUNet", (sigma, step, n_it))
plot_real(runs, REAL, methods=["DRUNet"])
# plot_real(runs, REAL, picks=[("RL", 100), ("TV_smooth", 0.00200), ("DRUNet", (0.01, 100))],
#           inset=(100, 160, 100, 160), profile=(130, 100, 130, 160))

#%%
# Publicity figure, two diagonals: raw | RL | TV (flip=True: TV upper left, raw lower right)
plot_real_bands(runs, REAL, picks=PICKS, title="TOMM20 · mitochondria",
                save=f"publicity/{REAL}_bands_rl_tv.png")
# plot_real_bands(runs, REAL, picks=PICKS, flip=True, band=0.25, cmap="glow_cyan")

# %%

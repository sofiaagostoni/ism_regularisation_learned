#%%
"""
Richardson-Lucy (no regularizer) on the SIMULATED 2D images, same measurement as
single_experiment.py / evaluate_methods.py (same image, flux, seed -> same noise), same
forward model and KL with background as the regularized methods (run_real.richardson_lucy):
    x <- x / A^T 1 * A^T( y / (A x + eta) ),   x_0 = A^T y / A^T 1

One run of max(--iter) iterations per image; the PSNR is computed at EVERY iteration
(semi-convergence curve), the image is kept at every --iter count.

    CUDA_VISIBLE_DEVICES=0 python rl_simulated.py                         # tub, tub_balls, tub_level, flux 20
    python rl_simulated.py --images tub --iter 10 50 100 500 2000
    python rl_simulated.py --flux 5 10 20

PSNR as in the rest of the code: "psnr" = reconstruction / its max (pnp_utils.psnr, the
reported metric), "psnr_clamp" = clamped to [0, 1] (deepinv's per-iteration metrics["psnr"],
the curve in the ProjGD figures of single_experiment.py).

Saved in SAVE_DIR/RL_<image>_flux<flux>_seed<seed>.pth (+ .png):
    psnr, psnr_clamp       per iteration (index k = after k+1 iterations)
    best_iter, best_psnr, best_ssim, x_best
    snapshots {n: x}, metrics {n: {psnr, ssim, microssim}} for the --iter counts
    history (cost, step_norm, iterate_norm, criticality, as run_real.py)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from opt_functions.data_preparation import load_data, make_measurements
from opt_functions.Solver_functions.pnp_utils import compute_final_metrics, psnr, psnr_fn
from run_real import richardson_lucy
from single_experiment import method_config, to_img

SAVE_DIR = "Results/unregularized"
IMAGES = ["tub", "tub_balls", "tub_level"]
ITERS = [10, 20, 50, 100, 200, 500, 1000]


def run_one(image, flux, seed, iters, save_dir, overwrite=False):
    path = os.path.join(save_dir, f"RL_{image}_flux{flux:g}_seed{seed}.pth")
    if os.path.exists(path) and not overwrite:
        print(f"{image} flux {flux:g}: exists, skipped ({path}; --overwrite to rerun)")
        return
    # Config of any explicit-prior method: only the simulation settings are used here
    cfg, _, _ = method_config("TV_smooth", seed, save_dir)
    x = load_data(image, cfg)
    meas = make_measurements(x, flux, cfg)
    n_max = max(iters)
    # every iteration count: one CPU copy per iteration (256 x 256 floats, 256 kB each)
    recs, hist, times = richardson_lucy(meas, range(1, n_max + 1), history=True)
    x_cpu = x.detach().float().cpu()
    p = np.array([psnr(recs[n], x_cpu) for n in range(1, n_max + 1)])
    p_clamp = np.array([psnr_fn(recs[n].clamp(0, 1), x_cpu).item()
                        for n in range(1, n_max + 1)])
    best = int(np.nanargmax(p)) + 1
    snaps = {n: recs[n] for n in sorted(set(iters) | {best})}
    mets = {n: compute_final_metrics(x_cpu, s) for n, s in snaps.items()}
    del recs

    res = {"method": "RL", "algo": "RL", "fidelity": "KL", "image": image, "flux": flux,
           "seed": seed, "lambda": 0.0, "iters": sorted(iters), "n_max": n_max,
           "psnr": p, "psnr_clamp": p_clamp, "best_iter": best, "best_psnr": float(p[best - 1]),
           "best_ssim": mets[best]["ssim"], "x_best": snaps[best], "x_gt": x_cpu,
           "snapshots": snaps, "metrics": mets, "history": hist, "time_s": times[n_max]}
    os.makedirs(save_dir, exist_ok=True)
    torch.save(res, path)

    print(f"\n{image}  flux {flux:g}  seed {seed}: Richardson-Lucy, {n_max} it in "
          f"{times[n_max]:.1f} s")
    for n in sorted(snaps):
        m = mets[n]
        mark = "  <- best" if n == best else ""
        print(f"  {n:>6} it: PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.4f}  "
              f"MicroSSIM {m['microssim']:.4f}{mark}")
    plot(res, meas["sum_y"], path.replace(".pth", ".png"))
    print(f"-> {path}")


def plot(res, sum_y, path):
    last = res["n_max"]
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), layout="constrained")
    panels = [("ground truth", res["x_gt"]), ("sum of the 25 detectors", sum_y),
              (f"RL best: {res['best_iter']} it, PSNR {res['best_psnr']:.2f}", res["x_best"]),
              (f"RL {last} it, PSNR {res['metrics'][last]['psnr']:.2f}", res["snapshots"][last])]
    for a, (t, im) in zip(axes[:4], panels):
        a.imshow(to_img(im), cmap="hot", vmin=0, vmax=1)
        a.set_title(t, fontsize=10)
        a.axis("off")
    k = np.arange(1, last + 1)
    ax = axes[4]
    ax.plot(k, res["psnr"], label="x / max(x)")
    ax.plot(k, res["psnr_clamp"], label="clamped to [0, 1]")
    ax.axvline(res["best_iter"], color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("iteration")
    ax.set_title("PSNR per iteration", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.suptitle(f"Richardson-Lucy | {res['image']} | flux {res['flux']:g} | seed {res['seed']}")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--images", nargs="+", default=IMAGES)
    ap.add_argument("--flux", nargs="+", type=float, default=[20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iter", nargs="+", type=int, default=ITERS,
                    help="iteration counts whose image is kept (the run goes to the max)")
    ap.add_argument("--save-dir", default=SAVE_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for image in args.images:
        for f in args.flux:
            flux = int(f) if f == int(f) else f
            run_one(image, flux, args.seed, args.iter, args.save_dir, args.overwrite)


if __name__ == "__main__":
    main()

#%%
"""
PSNR per iteration of the unregularized runs, Richardson-Lucy vs ProjGD (lambda = 0),
one panel per image (results of rl_simulated.py and single_experiment.py --lam 0).
Both curves: iterate clamped to [0, 1] (deepinv's per-iteration PSNR), so comparable.

    python plot_unregularized.py                       # tub, tub_balls, tub_level, flux 20
    python plot_unregularized.py --images tub --flux 5
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SAVE_DIR = "Results/unregularized"


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False) if path else None


def projgd_path(save_dir, image, flux):
    """Most recent ProjGD lambda = 0 result of this image and flux (None if not run)."""
    tag = f"best_rec_FidelityKL_{image}_flux{flux:g}_TV_smooth_lossnone_PROJ_GD.pth"
    paths = glob.glob(os.path.join(save_dir, "single_*", tag))
    return max(paths, key=os.path.getmtime) if paths else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=["tub", "tub_balls", "tub_level"])
    ap.add_argument("--flux", type=float, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=SAVE_DIR)
    args = ap.parse_args()
    flux = int(args.flux) if args.flux == int(args.flux) else args.flux

    fig, axes = plt.subplots(1, len(args.images), figsize=(6 * len(args.images), 4.5),
                             squeeze=False, layout="constrained")
    for ax, image in zip(axes[0], args.images):
        rl_file = os.path.join(args.save_dir, f"RL_{image}_flux{flux:g}_seed{args.seed}.pth")
        rl = load(rl_file if os.path.exists(rl_file) else None)
        pg = load(projgd_path(args.save_dir, image, flux))
        if rl is not None:
            k = np.arange(1, len(rl["psnr_clamp"]) + 1)
            ax.plot(k, rl["psnr_clamp"], label=f"Richardson-Lucy (best {rl['best_iter']} it)")
        else:
            print(f"{image}: no RL result ({rl_file})")
        if pg is not None and (pg.get("metrics") or {}).get("psnr"):
            p = np.asarray(pg["metrics"]["psnr"][0], float)
            ax.plot(np.arange(1, len(p) + 1), p,
                    label=f"ProjGD λ=0 ({pg.get('iter_run')} it, final {pg['psnr']:.2f})")
        else:
            print(f"{image}: no ProjGD lambda = 0 result in {args.save_dir}/single_*")
        ax.set_xscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("PSNR (dB), iterate clamped to [0, 1]")
        ax.set_title(f"{image}, flux {flux:g}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    out = os.path.join(args.save_dir, f"unregularized_psnr_flux{flux:g}.png")
    os.makedirs(args.save_dir, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()

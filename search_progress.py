#%%
"""
Progress of the adaptive parameter searches of evaluate_methods.py.

    python search_progress.py                     # running searches (partial_*.pth)
    python search_progress.py --all               # also the finished ones (best_rec_*.pth)
    python search_progress.py --filter tub_flux10 --plot

For each search: the points evaluated, in order, and the best PSNR so far.
--plot: PSNR vs lambda (1 parameter) or the evaluated (lambda, sigma) points (2 parameters).
"""
import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import torch

SAVE_DIR = "Results/evaluation"


def searches(save_dir, finished, pattern):
    files = glob.glob(os.path.join(save_dir, "*", "partial_*.pth"))
    if finished:
        files += glob.glob(os.path.join(save_dir, "*", "best_rec_*.pth"))
    files = [f for f in sorted(files) if pattern in os.path.basename(f)]
    for f in files:
        state = torch.load(f, map_location="cpu", weights_only=False)
        evals = state.get("evaluations") or []
        if evals:
            running = os.path.basename(f).startswith("partial_")
            tag = os.path.basename(f).split("_", 1)[1].removeprefix("rec_")[:-4]
            yield tag, running, evals, os.path.getmtime(f)


def print_search(tag, running, evals, mtime):
    import time
    status = (f"RUNNING, last update {time.strftime('%H:%M:%S', time.localtime(mtime))}"
              if running else "finished")
    print(f"\n=== {tag}  [{status}, {len(evals)} runs] ===")
    best = -math.inf
    for k, e in enumerate(evals):
        mark = ""
        if e["psnr"] > best:
            best, mark = e["psnr"], "  <- best so far"
        print(f"{k:3d}  lambda={e['lambda']:<10.4g} sigma={e['sigma']:<10.4g} "
              f"PSNR={e['psnr']:7.2f}{mark}")


def plot_search(tag, evals):
    lam = [e["lambda"] for e in evals]
    sig = [e["sigma"] for e in evals]
    p = [e["psnr"] if math.isfinite(e["psnr"]) else float("nan") for e in evals]
    best = max(range(len(p)), key=lambda k: p[k] if p[k] == p[k] else -math.inf)
    fig, ax = plt.subplots(figsize=(6, 4.2), layout="constrained")
    if len(set(sig)) == 1:                     # 1 parameter: PSNR vs lambda, order labelled
        ax.plot(lam, p, "o", color="0.6")
        for k, (x, y) in enumerate(zip(lam, p)):
            ax.annotate(str(k), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.plot(lam[best], p[best], "r*", ms=14)
        ax.set_ylabel("PSNR (dB)")
    else:                                      # 2 parameters: points coloured by PSNR
        h = ax.scatter(lam, sig, c=p, cmap="viridis", s=40)
        ax.plot(lam[best], sig[best], "r*", ms=14)
        ax.set_yscale("log")
        ax.set_ylabel("sigma")
        fig.colorbar(h, ax=ax, label="PSNR (dB)")
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_title(f"{tag}\nbest PSNR {p[best]:.2f} (run {best})", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.show()


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=SAVE_DIR)
    parser.add_argument("--all", action="store_true", help="include finished searches")
    parser.add_argument("--filter", default="", help="substring of the result name")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    found = False
    for tag, running, evals, mtime in searches(args.dir, args.all, args.filter):
        found = True
        print_search(tag, running, evals, mtime)
        if args.plot:
            plot_search(tag, evals)
    if not found:
        print("no search found" + ("" if args.all else " running (--all for finished ones)"))

#%%
"""
Clean confocal-whiteness figure for REAL data (no ground truth: no PSNR), same style as
whiteness_poisson/plot_whiteness_counts_clean.py: W of the confocal residual (25 detectors
summed, whiteness_real.py) vs lambda, masked and whole standardization; ring = minimum of
each curve (the lambda that criterion picks); legend with the lambdas only, no titles.

Data: Results/real_data/whiteness_real.csv (whiteness_real.py). Several rows with the same
lambda (reruns): the most recent one is kept. Rows with a crop are left out.

    python plot_whiteness_real_clean.py                              # 02_TUB, zebrafish; TV_smooth
    python plot_whiteness_real_clean.py --data 02_TUB --method L1 --modes standard

Output: Results/real_data/whiteness/<data>_<method>_confocal_<mode>_clean.png / .pdf
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = "Results/real_data/whiteness_real.csv"
OUT_DIR = "Results/real_data/whiteness"
DATA = ["02_TUB", "zebrafish"]

INK, INK2, MUTED, GRID = "#1f1f1e", "#52514e", "#8a8984", "#e6e5e0"
CRITERIA = [  # (label, column, colour, linestyle, marker)
    ("W masked", "W_confocal_masked_{m}", "#2a78d6", "-", "o"),
    ("W whole", "W_confocal_whole_{m}", "#eb6834", "--", "s"),
]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": GRID,
    "legend.fontsize": 8.5,
})


def rows(df, data, method):
    g = df[(df["data"] == data) & (df["method"] == method) & df["lambda"].notna()]
    if "crop" in g:
        g = g[g["crop"].isna()]
    g = g.sort_values("mtime").groupby("lambda", as_index=False).last()   # newest per lambda
    return g.sort_values("lambda")


def plot(df, data, method, mode):
    g = rows(df, data, method)
    if len(g) < 2:
        print(f"{data} {method}: {len(g)} lambda values in {CSV} (skipped)")
        return
    lam = g["lambda"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(2.0, color=MUTED, ls=(0, (4, 3)), lw=1.1, label="white noise (W = 2)")
    picks = []
    for label, col, color, ls, mk in CRITERIA:
        W = g[col.format(m=mode)].to_numpy(float)
        j = int(np.nanargmin(W))
        picks.append((label, lam[j], W[j]))
        ax.plot(lam, W, ls=ls, lw=1.8, color=color, marker=mk, ms=4,
                mfc=color if ls == "-" else "white", mew=1.0, zorder=3,
                label=f"{label}  (min: λ = {lam[j]:.2g})")
        ax.plot(lam[j], W[j], "o", ms=12, mfc="none", mec=color, mew=1.8, zorder=4)
    ax.set_xscale("log")
    W_all = g[[c.format(m=mode) for _, c, *_ in CRITERIA]].to_numpy(float)
    if np.nanmax(W_all) / np.nanmin(W_all) > 10:
        ax.set_yscale("log")
    ax.set_xlabel("λ")
    ax.set_ylabel("confocal residual whiteness W")
    handles, labels = ax.get_legend_handles_labels()
    ring = plt.Line2D([], [], ls="", marker="o", ms=10, mfc="none", mec=INK2, mew=1.6)
    ax.legend(handles[1:] + [handles[0], ring], labels[1:] + [labels[0], "min W (λ picked)"],
              loc="upper left")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{data}_{method}_confocal_{mode}_clean")
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    print(f"saved {out}.png / .pdf")
    for label, l, w in picks:
        print(f"  min {label:9s} λ = {l:.3g}  (W = {w:.3f})")
    capped = g[g["iterations_run"] >= g["max_iter"]]["lambda"].tolist()
    if capped:
        print(f"  note: stopped at max_iter (not converged) for λ = {[float(f'{v:.3g}') for v in capped]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=DATA)
    ap.add_argument("--method", default="TV_smooth")
    ap.add_argument("--modes", nargs="+", default=["standard", "highpass"])
    a = ap.parse_args()
    df = pd.read_csv(CSV)
    for data in a.data:
        for mode in a.modes:
            plot(df, data, a.method, mode)


if __name__ == "__main__":
    main()

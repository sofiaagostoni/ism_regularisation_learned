#%%
"""
Figures of DRUNet, DRUNet-ft, WCRR and WCRR_ISM on simulated and real data, saved to OUT_DIR.

Simulated (SIM_DATA, flux 20, KL): the best-PSNR run of each method (adaptive search or a
single_experiment.py run), compare_results.plot_single_images: one png per image (input =
sum of the 25 detectors, ground truth, each method; same colour range for GT and methods).
    OUT_DIR/<data>/<data>_flux20_plane1_<label>.png

Real (REAL_DATA, run_real.py results): no ground truth, so no "best": every saved setting
of each method in one figure (compare_results.plot_real, raw data first), to choose by eye.
--flux 1 runs of the learned priors are left out (x must be in ~[0, 1] for them).
    OUT_DIR/<data>/<data>_<method>_all.png

    python plot_best_methods.py
    python plot_best_methods.py --sim tub_balls --real zebrafish
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import torch

import compare_results as cr

OUT_DIR = "figures/best_methods"
SIM_DATA = ["tub_level", "tub_balls"]
REAL_DATA = ["02_TUB", "zebrafish"]
SIM_METHODS = ["DRUNet [PGD] KL", "DRUNet-ft", "WCRR [PROJ_GD] KL", "WCRR_ISM [PROJ_GD] KL"]
REAL_METHODS = ["DRUNet", "DRUNet_ft", "WCRR", "WCRR_ISM"]
LEARNED = {"WCRR", "WCRR_ISM", "DRUNet", "DRUNet_ft"}      # need x in ~[0, 1]: flux != 1


def real_runs(data_list):
    """compare_results.load_real_runs, only for these data folders (faster)."""
    rows = []
    for data in data_list:
        folder = os.path.join(cr.REAL_ROOT, data)
        for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if f.endswith(".pth"):
                r = torch.load(os.path.join(folder, f), map_location="cpu", weights_only=False)
                rows.append({**{k: r.get(k) for k in cr._REAL_FIELDS},
                             "path": os.path.join(folder, f)})
    df = pd.DataFrame(rows, columns=list(cr._REAL_FIELDS) + ["path"])
    if len(df):
        df["converged"] = (df.iterations_run < df.max_iter).astype(object)
        df.loc[(df.method == "RL") | df.method.isin(cr.PNP_METHODS), "converged"] = None
        df = df.sort_values(["data", "method", "crop", "lambda", "sigma", "iterations_run"],
                            na_position="first")
    return df.reset_index(drop=True)


def sim(data_list):
    df = cr.load(cr.ROOT, split_by_fidelity=False, search="adaptive")
    best = cr.best_per_method(df)
    for data in data_list:
        g = best[(best.data == data) & (best.flux == 20)]
        found = [m for m in SIM_METHODS if g.method.str.contains(m, regex=False).any()]
        missing = [m for m in SIM_METHODS if m not in found]
        if missing:
            print(f"{data}: not saved: {missing}")
        if not found:
            continue
        print(f"\n{data}, flux 20: best run of " + ", ".join(found))
        sub = g[g.method.apply(lambda m: any(s in m for s in found))]
        for r in sub.itertuples():
            print(f"  {r.method:28s} PSNR {r.psnr:.2f}  {r.params}")
        out = os.path.join(OUT_DIR, data)
        os.makedirs(out, exist_ok=True)
        cr.plot_single_images(best, data, 20, methods=found, planes=(cr.REG_PLANE,),
                              save_dir=out)
        print(f"saved in {out}/")


def real(data_list):
    runs = real_runs(data_list)
    for data in data_list:
        out = os.path.join(OUT_DIR, data)
        os.makedirs(out, exist_ok=True)
        for m in REAL_METHODS:
            sel = runs[(runs.data == data) & (runs.method == m)]
            if m in LEARNED:
                sel = sel[sel.flux != 1]
            if sel.empty:
                print(f"{data}: no {m} run")
                continue
            path = os.path.join(out, f"{data}_{m}_all.png")
            cr.plot_real(sel, data, methods=[m], ncols=4 if len(sel) < 12 else 5, save=path)
            print(f"{data}: {m}, {len(sel)} settings -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", nargs="*", default=SIM_DATA)
    ap.add_argument("--real", nargs="*", default=REAL_DATA)
    a = ap.parse_args()
    if a.sim:
        sim(a.sim)
    if a.real:
        real(a.real)


if __name__ == "__main__":
    main()

#%%
"""
Residual whiteness of every real-data reconstruction of run_real.py (Results/real_data).

For each saved result x (both planes) the expected photon counts of the forward model
used by run_real.py are rebuilt (same loader, crop, PSF, background and alpha):
    lambda_hat_j = flux * (A_j x + eta_j)        (flux = 1: x in counts; flux = alpha)
and compared with the raw counts y_j of the 25 detectors:
    'masked' : Z = (y - lambda_hat T) / sqrt(lambda_hat V) on y > 0     (standardize_unbiased_masked)
    'whole'  : Z = (y - lambda_hat) / sqrt(lambda_hat)                   (standardize)
Two whiteness metrics, each with the two standardizations and two modes:
  stack    : W = M_eff * whiteness_measure(Z) on the (25, N, N) stack, as in the simulations
             (whiteness_poisson / whiteness_experiment.py): correlations between pixels AND
             detectors (3D FFT).
  confocal : the 25 detectors summed first (a confocal image: sum_j y_j ~ Poisson(sum_j
             lambda_hat_j)), then the same W on the (N, N) residual image (2D FFT): only
             correlations between pixels.
~2 for white noise, larger = structure left in the residual (under-fitting). Modes:
'standard' (all frequencies) and 'highpass' (cutoff 0.10).
Zstd = std of Z (masked: on y > 0): ~1 if the Poisson model with lambda_hat is right
(< 1: residual too small, over-fitting).
`trim` px are removed from each border of Z first (wrap-around of the circular FFT model,
as the trim of compare_results.plot_real).

The whiteness always uses the Poisson model with the measurement background, also for
runs fitted with another fidelity (L2, KL0): it measures how Poisson-white the residual is.

    python whiteness_real.py                          # every result, CPU (default)
    python whiteness_real.py --data 02_TUB 03_H3      # some datasets
    python whiteness_real.py --device cuda            # on the GPU
    python whiteness_real.py --plot-only              # only the figures, from the csv

Safe to restart: results already in the csv (same file, same modification time) are
skipped. Output: Results/real_data/whiteness_real.csv (one row per result) and
Results/real_data/whiteness/<data>_<mode>.png (W vs lambda / iterations, per method; rows:
stack masked / whole, confocal masked / whole).
"""
import argparse
import contextlib
import io
import os

import pandas as pd
import torch

from opt_functions.data_preparation import batch_measurements, real_measurements_3D
from opt_functions.Solver_functions.white_opt_pnp import (standardize,
                                                         standardize_unbiased_masked,
                                                         whiteness_measure)
from run_real import load_real                     # same loader and centre crop as the runs

REAL_ROOT = "Results/real_data"
OUT_CSV = os.path.join(REAL_ROOT, "whiteness_real.csv")
FIG_DIR = os.path.join(REAL_ROOT, "whiteness")
TRIM = 12
MASKS = {"masked": standardize_unbiased_masked, "whole": standardize}
MODES = {"standard": 0.10, "highpass": 0.10}       # mode -> cutoff (unused by 'standard')
FIELDS = ("data", "method", "algo", "fidelity", "lambda", "sigma", "step_scale", "flux",
          "tv_eps", "crop", "max_iter", "iterations_run", "bt_s")
METRICS = ("stack", "confocal")
VERSION = 2                     # rows of an older version (other metrics) are recomputed


# ---------------------------------------------------------------------------
# Forward model of a (data, crop), as run_real.py
# ---------------------------------------------------------------------------

def build_model(data, crop, device):
    """Raw counts y (25, N, N), batched A and the background eta (1, 25, 1, 1), in units
    of x at flux 1 (A and eta do not depend on alpha: lambda_hat = flux * (A x + eta))."""
    counts, PSF, _ = load_real(data, None if crop is None or pd.isna(crop) else int(crop))
    meas = batch_measurements(real_measurements_3D(counts, PSF, 1.0, device))
    y = counts[:, 0].float().to(device)
    eta = torch.as_tensor(meas["eta_vec"]).reshape(1, -1, 1, 1).to(device)
    return y, meas["physics"], eta


def whiteness_2d(Z, mode="standard", cutoff=0.10):
    """whiteness_measure for one (N, N) image: the same W = M_eff * sum|F|^4 / (sum|F|^2)^2
    with the 2D FFT; 'highpass' zeroes the same central block (+- cutoff / 2 per axis)."""
    mag_sq = torch.abs(torch.fft.fftshift(torch.fft.fft2(Z))) ** 2
    M_eff = Z.numel()
    if mode == "highpass":
        s0, s1 = mag_sq.shape
        c0, c1 = s0 // 2, s1 // 2
        r0, r1 = max(1, int(s0 * cutoff / 2)), max(1, int(s1 * cutoff / 2))
        mag_sq[c0 - r0:c0 + r0, c1 - r1:c1 + r1] = 0.0
        M_eff -= (2 * r0) * (2 * r1)
    return float(M_eff * (mag_sq ** 2).sum() / mag_sq.sum() ** 2)


def _crop(t, trim):
    return t[..., trim:-trim, trim:-trim] if trim else t


def whiteness_of(x, flux, model, trim):
    """All metrics of one reconstruction: W_<metric>_<mask>_<mode>, Zmean_ / Zstd_<metric>_<mask>."""
    y, physics, eta = model
    out = {}
    with torch.no_grad():
        lam = flux * (physics.A(x.float().to(y.device)).clamp_min(0) + eta)[0]   # (25, N, N)
        for metric in METRICS:
            # confocal: the 25 detectors summed (sum of Poisson = Poisson of the summed mean)
            yy, ll = (y, lam) if metric == "stack" else (y.sum(0), lam.sum(0))
            for m, std_fn in MASKS.items():
                Z = _crop(std_fn(yy, ll), trim)
                sup = Z[_crop(yy, trim) > 0] if m == "masked" else Z
                out[f"Zmean_{metric}_{m}"] = float(sup.mean())
                out[f"Zstd_{metric}_{m}"] = float(sup.std())
                for mode, cutoff in MODES.items():
                    if metric == "stack":
                        with contextlib.redirect_stdout(io.StringIO()):     # it prints Z.shape
                            wh, M_eff = whiteness_measure(Z, mode=mode, cutoff=cutoff)
                        w = float(M_eff * wh)
                    else:
                        w = whiteness_2d(Z, mode, cutoff)
                    out[f"W_{metric}_{m}_{mode}"] = w
    return out


# ---------------------------------------------------------------------------
# All results
# ---------------------------------------------------------------------------

def result_files(datasets=None):
    files = []
    for data in sorted(os.listdir(REAL_ROOT)):
        folder = os.path.join(REAL_ROOT, data)
        if not os.path.isdir(folder) or folder == FIG_DIR or (datasets and data not in datasets):
            continue
        files += [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".pth")]
    return files


def compute(datasets, device, trim, overwrite):
    done = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) and not overwrite else pd.DataFrame()
    if len(done):
        done = done[done.get("version", pd.Series(0, index=done.index)) == VERSION]
    known = (set(zip(done.path, done.mtime.round(3), done.trim)) if len(done) else set())
    rows = done.to_dict("records") if len(done) else []
    todo = [f for f in result_files(datasets)
            if (f, round(os.path.getmtime(f), 3), trim) not in known]
    print(f"{len(todo)} results to evaluate ({len(rows)} already in {OUT_CSV})")
    models, key_prev = {}, None
    for k, path in enumerate(todo, 1):
        r = torch.load(path, map_location="cpu", weights_only=False)
        key = (r["data"], r.get("crop"))
        if key not in models:
            if key_prev is not None and key_prev[0] != key[0]:
                models.clear()                     # one dataset in memory at a time
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            models[key] = build_model(*key, device)
        key_prev = key
        x = r["x"]
        row = {f: r.get(f) for f in FIELDS}
        row.update(path=path, mtime=round(os.path.getmtime(path), 3), trim=trim,
                   version=VERSION, finite=bool(torch.isfinite(x).all()))
        if row["finite"]:
            row.update(whiteness_of(x, float(r["flux"]), models[key], trim))
        rows.append(row)
        print(f"[{k}/{len(todo)}] {os.path.relpath(path, REAL_ROOT)}  "
              + ("  ".join(f"{mt} W {row[f'W_{mt}_masked_standard']:.4g} "
                           f"Zstd {row[f'Zstd_{mt}_masked']:.3f}" for mt in METRICS)
                 if row["finite"] else "not finite"),
              flush=True)
        if k % 10 == 0 or k == len(todo):          # save as we go (safe to restart)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures and table
# ---------------------------------------------------------------------------

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]                                # categorical slots 1-8, fixed order
ITER_METHODS = {"RL", "DRUNet", "DRUNet_ft", "DRUNet_scratch"}   # x axis = iterations


def _series(g):
    """One line per (method, sigma, step, crop), x = lambda or iterations."""
    out = []
    keys = ["method", "sigma", "step_scale", "crop"]
    g = g.copy()
    for k in keys:
        g[k] = g[k].astype(object).where(g[k].notna(), None)
    for key, s in g.groupby(keys, dropna=False, sort=True):
        method, sigma, step, crop = key
        it = method in ITER_METHODS
        s = s.sort_values("iterations_run" if it else "lambda")
        lab = method + (f" σ={sigma:g}" if sigma is not None and not pd.isna(sigma) else "") \
            + (f" step={step:g}" if step is not None and not pd.isna(step) and it else "") \
            + (f" crop{int(crop)}" if crop is not None and not pd.isna(crop) else "")
        out.append((lab, it, s))
    return out


def plot(df, mode="standard"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    df = df[df.finite]
    rows = [(mt, mask) for mt in METRICS for mask in MASKS]       # stack / confocal x masked / whole
    for data, g in df.groupby("data"):
        series = _series(g)
        fig, axes = plt.subplots(len(rows), 2, figsize=(13, 4 * len(rows)), sharey="row",
                                 layout="constrained", squeeze=False)
        methods = sorted({lab.split(" ")[0] for lab, _, _ in series})
        color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(methods)}
        styles = ["-", "--", ":", "-."]
        style = {}                                 # the n-th line of a method: n-th style
        for lab, _, _ in series:
            m = lab.split(" ")[0]
            style[lab] = styles[sum(1 for l in style if l.split(" ")[0] == m) % len(styles)]
        for r_i, (metric, mask) in enumerate(rows):
            key = f"W_{metric}_{mask}_{mode}"
            for col, it_axis in enumerate((False, True)):
                ax = axes[r_i, col]
                for lab, it, s in series:
                    if it != it_axis:
                        continue
                    x = s.iterations_run if it else s["lambda"]
                    ax.plot(x, s[key], marker="o", ms=4, lw=1.6, ls=style[lab],
                            color=color[lab.split(" ")[0]], label=lab)
                ax.axhline(2, color="#6b6b66", ls="--", lw=1, label="white noise (≈2)")
                ax.set_xscale("log")
                vals = g[key].dropna()
                if len(vals) and vals.max() > 20:
                    ax.set_yscale("log")
                ax.grid(True, color="#e4e3dd", lw=0.6)
                ax.set_axisbelow(True)
                ax.set_xlabel("iterations" if it_axis else "λ")
                if col == 0:
                    ax.set_ylabel(f"W {metric} · {mask}")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=7, loc="best")
                what = ("stack: 25 det. × pixels (3D)" if metric == "stack"
                        else "confocal: Σ 25 det., pixels (2D)")
                ax.set_title(("RL / PnP (vs iterations)" if it_axis else "regularized (vs λ)")
                             + f" · {what} · {mask}", fontsize=9)
        fig.suptitle(f"{data} · residual whiteness of the real reconstructions "
                     f"({mode}, trim {int(g.trim.iloc[0])} px)", fontsize=12)
        path = os.path.join(FIG_DIR, f"{data}_{mode}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")


def print_table(df, mode="standard"):
    cols = ["data", "method", "lambda", "sigma", "flux", "crop", "iterations_run"]
    for mt in METRICS:
        cols += [f"W_{mt}_masked_{mode}", f"W_{mt}_whole_{mode}", f"Zstd_{mt}_masked"]
    short = {c: c.replace(f"_{mode}", "").replace("confocal", "conf") for c in cols}
    with pd.option_context("display.width", 300, "display.max_rows", 2000,
                           "display.float_format", "{:.4g}".format):
        print(df[df.finite][cols].rename(columns=short)
              .sort_values(["data", "method", "lambda", "iterations_run"]).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=None, help="only these datasets")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--threads", type=int, default=16, help="CPU threads for torch")
    p.add_argument("--trim", type=int, default=TRIM, help="px removed from each border of Z")
    p.add_argument("--overwrite", action="store_true", help="recompute everything")
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.plot_only:
        df = pd.read_csv(OUT_CSV)
    else:
        df = compute(a.data, torch.device(a.device), a.trim, a.overwrite)
    if a.data:
        df = df[df.data.isin(a.data)]
    if df.empty:
        print("no results")
        return
    for mode in MODES:
        plot(df, mode)
    print_table(df)


if __name__ == "__main__":
    main()

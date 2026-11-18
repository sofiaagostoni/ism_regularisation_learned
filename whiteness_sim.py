#%%
"""
Residual whiteness of the saved SIMULATED reconstructions of evaluate_methods.py /
single_experiment.py / run_3d_from_2d.py (Results/evaluation_stop/<run>/best_rec_*.pth),
for the tubulin images (default: every image whose name starts with "tub": tub, tub_level,
tub_balls, tub_3D, tub_nuc_3D).

Each result keeps x_rec (the best reconstruction of its search), x_gt, the flux and the
simulation settings (seed, Nx, Nz, pxsizex): the measurement is simulated again as in the
evaluation (load_data + make_measurements, same seed, default device: the noise depends
on the device RNG, as compare_results.sum_measurements). In counts:
    y_j = counts of detector j,   lambda_hat_j = flux * (A_j x_rec + eta_j)
and the oracle lambda_true_j = flux * (A_j x_gt + eta_j). Same metrics as whiteness_real.py:
  stack    : W on the (25, N, N) residual stack (3D FFT: pixels and detectors)
  confocal : W on the residual of the detector sum (2D FFT: pixels only)
for 'masked' / 'whole' standardization and 'standard' / 'highpass' mode; W ~ 2 for white
noise. Zmean / Zstd: mean and std of Z (masked: on y > 0). W_true_*: the same for the true
mean (what W would be for the exact solution). No border trim by default (the simulated
data use the same circular model as the reconstruction).
Note: the reconstructions of the simulations are clamped to [0, 1] (cfg.x_max = 1).

    python whiteness_sim.py                        # every tub* result
    python whiteness_sim.py --images tub tub_3D    # some images
    python whiteness_sim.py --plot-only            # only the figures, from the csv

Safe to restart (results already in the csv are skipped). Output: Results/whiteness_sim/
whiteness_sim.csv (one row per result) and <image>_<mode>.png (W vs PSNR, per flux).
"""
import argparse
import contextlib
import io
import os
import re

import pandas as pd
import torch

from opt_functions.data_preparation import load_data, make_measurements
from opt_functions.Solver_functions.pnp_utils import Config, default_device
from opt_functions.Solver_functions.white_opt_pnp import whiteness_measure
from whiteness_real import MASKS, MODES, whiteness_2d

ROOT = "Results/evaluation_stop"
OUT_DIR = "Results/whiteness_sim"
OUT_CSV = os.path.join(OUT_DIR, "whiteness_sim.csv")
METRICS = ("stack", "confocal")
VERSION = 1
_NAME = re.compile(r"best_rec_Fidelity[A-Z0-9]+_(?P<name>.+)_flux(?P<flux>[0-9.]+)_")
FIELDS = ("run_id", "name", "flux", "fidelity", "prior", "algo", "loss", "lambda", "sigma",
          "step_scale", "iter", "iter_run", "psnr", "ssim", "microssim")


# ---------------------------------------------------------------------------
# Measurement (as in the evaluation) and whiteness
# ---------------------------------------------------------------------------

def result_files(images=None, prefix="tub"):
    for run in sorted(os.listdir(ROOT)):
        folder = os.path.join(ROOT, run)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            m = _NAME.match(f)
            if not (m and f.endswith(".pth")):
                continue
            name = m.group("name")
            if (images and name in images) or (not images and name.startswith(prefix)):
                yield os.path.join(folder, f)


def simulate(name, flux, c, device):
    """Counts y (25, N, N), the 25 detector operators and the ground truth of the run."""
    Nz = c.get("Nz", 1)
    cfg = Config(device=device, Nx=c.get("Nx", 256), Nz=Nz, pxsizex=c.get("pxsizex", 40),
                 is_3d=Nz > 1, seed=c.get("seed", 0))
    x = load_data(name, cfg)
    meas = make_measurements(x, flux, cfg)
    y = torch.cat(list(meas["y"]), dim=0)[:, 0] * flux          # intensity -> counts
    return y.round(), meas["physics"].physics_list, x


def expected_counts(x, physics_list, flux):
    """lambda_j = flux * (A_j x + eta_j), (25, N, N)."""
    return torch.cat([p.A(x) + p.noise_model.bkg for p in physics_list], dim=0)[:, 0] * flux


def _crop(t, trim):
    return t[..., trim:-trim, trim:-trim] if trim else t


def whiteness_all(y, lam, trim=0, prefix=""):
    out = {}
    for metric in METRICS:
        yy, ll = (y, lam) if metric == "stack" else (y.sum(0), lam.sum(0))
        for m, std_fn in MASKS.items():
            Z = _crop(std_fn(yy, ll), trim)
            sup = Z[_crop(yy, trim) > 0] if m == "masked" else Z
            out[f"{prefix}Zmean_{metric}_{m}"] = float(sup.mean())
            out[f"{prefix}Zstd_{metric}_{m}"] = float(sup.std())
            for mode, cutoff in MODES.items():
                if metric == "stack":
                    with contextlib.redirect_stdout(io.StringIO()):     # it prints Z.shape
                        wh, M_eff = whiteness_measure(Z, mode=mode, cutoff=cutoff)
                    w = float(M_eff * wh)
                else:
                    w = whiteness_2d(Z, mode, cutoff)
                out[f"{prefix}W_{metric}_{m}_{mode}"] = w
    return out


def compute(images, device, trim, overwrite):
    os.makedirs(OUT_DIR, exist_ok=True)
    done = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) and not overwrite else pd.DataFrame()
    if len(done):
        done = done[(done.version == VERSION) & (done.trim == trim)]
    known = set(zip(done.path, done.mtime.round(3))) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    todo = [f for f in result_files(images) if (f, round(os.path.getmtime(f), 3)) not in known]
    print(f"{len(todo)} results to evaluate ({len(rows)} already in {OUT_CSV}), device {device}")
    sims = {}
    for k, path in enumerate(todo, 1):
        r = torch.load(path, map_location="cpu", weights_only=False)
        c = r.get("config") or {}
        key = (r["name"], r["flux"], c.get("seed", 0), c.get("Nz", 1))
        if key not in sims:
            y, phys, x = simulate(r["name"], r["flux"], c, device)
            with torch.no_grad():                  # oracle: residual w.r.t. the true mean
                oracle = whiteness_all(y, expected_counts(x, phys, r["flux"]), trim, "true_")
            sims[key] = (y, phys, x, oracle)
        y, phys, x, oracle = sims[key]
        row = {f: r.get(f) for f in FIELDS}
        row["run_id"] = r.get("run_id") or os.path.basename(os.path.dirname(path))
        row.update(path=path, mtime=round(os.path.getmtime(path), 3), trim=trim, version=VERSION,
                   gt_match=bool(torch.allclose(torch.as_tensor(r["x_gt"]).float(),
                                                x.float().cpu(), atol=1e-6)))
        x_rec = r.get("x_rec")
        row["finite"] = x_rec is not None and bool(torch.isfinite(torch.as_tensor(x_rec)).all())
        if row["finite"]:
            with torch.no_grad():
                lam = expected_counts(torch.as_tensor(x_rec).float().to(y.device), phys, r["flux"])
                row.update(whiteness_all(y, lam, trim))
            row.update(oracle)
        rows.append(row)
        print(f"[{k}/{len(todo)}] {r['name']} flux {r['flux']:g} {r.get('prior')} "
              f"{r.get('fidelity')} {r.get('algo')}  PSNR {row['psnr']:.2f}  "
              + ("  ".join(f"{mt} W {row[f'W_{mt}_masked_standard']:.4g} "
                           f"(oracle {row[f'true_W_{mt}_masked_standard']:.4g})" for mt in METRICS)
                 if row["finite"] else "no reconstruction")
              + ("" if row["gt_match"] else "  [ground truth differs from load_data!]"),
              flush=True)
        if k % 10 == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        if len(sims) > 8:                          # keep a few measurements in memory
            sims.pop(next(iter(sims)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures and table
# ---------------------------------------------------------------------------

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]                                # categorical slots 1-8, fixed order
MARKERS = {"KL": "o", "L2": "s", "KL0": "^"}


def method_label(r):
    return f"{r.prior} [{r.algo}] {r.fidelity}"


def plot(df, mode="standard"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = df[df.finite].copy()
    df["method"] = [method_label(r) for r in df.itertuples()]
    for name, g in df.groupby("name"):
        fluxes = sorted(g.flux.unique())
        rows = [(mt, "masked") for mt in METRICS]
        fig, axes = plt.subplots(len(rows), len(fluxes), figsize=(4.6 * len(fluxes), 4.2 * len(rows)),
                                 squeeze=False, layout="constrained")
        priors = sorted(g.prior.unique())
        color = {p: COLORS[i % len(COLORS)] for i, p in enumerate(priors)}
        for i, (metric, mask) in enumerate(rows):
            key = f"W_{metric}_{mask}_{mode}"
            for j, fl in enumerate(fluxes):
                ax = axes[i, j]
                s = g[g.flux == fl]
                for r in s.itertuples():
                    ax.scatter(r.psnr, getattr(r, key), s=46, color=color[r.prior],
                               marker=MARKERS.get(r.fidelity, "D"), edgecolors="white",
                               linewidths=0.8, label=r.method, zorder=3)
                if len(s):
                    ax.axhline(s[f"true_{key}"].iloc[0], color="#6b6b66", ls="--", lw=1.1,
                               label="oracle (true mean)")
                ax.axhline(2, color="#b9b8b2", ls=":", lw=1, label="white noise (≈2)")
                if len(s) and s[key].max() > 20:
                    ax.set_yscale("log")
                ax.grid(True, color="#e4e3dd", lw=0.6)
                ax.set_axisbelow(True)
                ax.set_xlabel("PSNR (dB)")
                if j == 0:
                    ax.set_ylabel(f"W {metric} · {mask} ({mode})")
                ax.set_title(f"flux {fl:g}", fontsize=10)
        h, l = axes[0, 0].get_legend_handles_labels()
        for ax in axes.flat:                       # one legend for the figure, all methods
            hh, ll = ax.get_legend_handles_labels()
            for a, b in zip(hh, ll):
                if b not in l:
                    h.append(a)
                    l.append(b)
        fig.legend(h, l, loc="outside right center", fontsize=7, frameon=False)
        fig.suptitle(f"{name} · residual whiteness of the best reconstruction of each method "
                     f"(colour = prior, marker = fidelity)", fontsize=11)
        path = os.path.join(OUT_DIR, f"{name}_{mode}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")


def print_table(df, mode="standard"):
    df = df[df.finite].copy()
    df["method"] = [method_label(r) for r in df.itertuples()]
    cols = ["name", "flux", "method", "lambda", "sigma", "psnr"]
    for mt in METRICS:
        cols += [f"W_{mt}_masked_{mode}", f"true_W_{mt}_masked_{mode}", f"Zmean_{mt}_masked"]
    short = {c: c.replace(f"_masked_{mode}", "").replace("_masked", "").replace("confocal", "conf")
             for c in cols}
    with pd.option_context("display.width", 300, "display.max_rows", 2000,
                           "display.float_format", "{:.4g}".format):
        print(df[cols].rename(columns=short).sort_values(["name", "flux", "psnr"],
                                                         ascending=[True, True, False])
              .to_string(index=False))
    if (~df.gt_match).any():
        print("\nWARNING: ground truth of these results differs from load_data (data changed?):")
        print(df[~df.gt_match][["name", "flux", "method", "path"]].to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", default=None, help="default: every tub* image")
    p.add_argument("--device", default=None,
                   help="default: as the evaluation (cuda if available): the simulated noise "
                        "depends on the device RNG, so keep it for the same measurement")
    p.add_argument("--trim", type=int, default=0, help="px removed from each border of Z")
    p.add_argument("--overwrite", action="store_true", help="recompute everything")
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if a.plot_only:
        df = pd.read_csv(OUT_CSV)
    else:
        device = torch.device(a.device) if a.device else default_device()
        df = compute(a.images, device, a.trim, a.overwrite)
    if a.images:
        df = df[df.name.isin(a.images)]
    if df.empty:
        print("no results")
        return
    for mode in MODES:
        plot(df, mode)
    print_table(df)


if __name__ == "__main__":
    main()

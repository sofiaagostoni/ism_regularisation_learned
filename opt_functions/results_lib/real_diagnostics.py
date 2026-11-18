"""
Diagnostics of the real-data runs: convergence curves (as sim_diagnostics.convergence_check,
without PSNR) and the residual whiteness (whiteness_real.py -> whiteness_real.csv).
"""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .real import (REAL_ROOT, PNP_METHODS, _not_converged, _select_real, _real_label,
                   _theme)


def plot_real_convergence(runs, data, methods=None, lambdas=None, crop="any", iters=None,
                          tail=0.1, save=None):
    """
    Convergence curves of run_real.py reconstructions (as convergence_check for the
    simulations, without PSNR: no ground truth). Needs the histories that run_real.py
    saves by default (runs made before that, or with --no-history, are listed and skipped).

    Panels (iteration k, iterates z_k unclamped):
      (F(z_k) - F*) / |F*|        F = f + lam R in float64, F* = min of each curve
      ||G(z_k)|| / ||G(z_0)||     G(z) = z - P_+(z - grad F(z)) (smooth priors; prox-gradient
                                  map for non-smooth ones): 0 iff z is critical
      ||z_{k+1} - z_k|| / ||z_k|| the early-stop quantity, its tolerance dashed
      ||z_k||                     bounded iterates
      step size tau_k = 1 / L_k   (backtracking; flat at 1 / bt_s = capped)
    One colour per run (by lambda); dashed = stopped at max_iter (not converged).
    Prints a summary table and returns it; tail = fraction of the last iterations for
    "tail length" and "at step cap %".
    """
    sel = _select_real(runs, data, methods, lambdas, crop, iters)
    curves, skipped, loaded = [], [], {}
    for i, r in sel.iterrows():
        res = torch.load(r.path, map_location="cpu", weights_only=False)
        if not res.get("history") or "cost" not in res["history"]:
            skipped.append(os.path.basename(r.path))
        else:
            loaded[i] = res
    dup = sel.loc[list(loaded)].duplicated(["method", "lambda", "sigma"], keep=False)
    for i, res in loaded.items():
        r, h = sel.loc[i], res["history"]
        L = res.get("L_history")
        curves.append(dict(label=_real_label(r, detail=bool(dup[i])).replace("\n", ", "),
                           row=r, h={k: np.asarray(v, dtype=np.float64) for k, v in h.items()},
                           tau=1.0 / np.asarray(L, dtype=np.float64) if L else None,
                           cap=1.0 / r.bt_s if r.bt_s and not pd.isna(r.bt_s) else None,
                           stop_tol=res.get("stop_tol")))
    if skipped:
        print("no convergence history (rerun with the current run_real.py):\n  "
              + "\n  ".join(skipped))
    if not curves:
        print(f"no convergence histories for {data}")
        return None

    cmap = plt.get_cmap("viridis")
    # colour order: by lambda; runs without one (Richardson-Lucy) after, by iteration count
    order = np.argsort([c["row"]["lambda"] if c["row"]["lambda"] is not None
                        and not pd.isna(c["row"]["lambda"])
                        else 1e30 + c["row"]["iterations_run"] for c in curves])
    colors = {int(j): cmap(t) for j, t in zip(order, np.linspace(0, 0.9, len(curves)))}
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), layout="constrained")
    ax = dict(zip(["F", "crit", "step", "norm", "tau", "legend"], axes.flat))
    rows = []
    for j, c in enumerate(curves):
        h, r = c["h"], c["row"]
        F, s, n, g = h["cost"], h["step_norm"], h["iterate_norm"], h["criticality"]
        k = np.arange(1, len(F) + 1)
        kw = dict(color=colors[j], lw=1.4, label=c["label"],
                  ls="--" if _not_converged(r) else "-")
        gap = (F - F.min()) / abs(F.min())
        ax["F"].plot(k, np.where(gap > 0, gap, np.nan), **kw)
        ax["crit"].plot(np.arange(len(g)), g / g[0], **kw)
        rel = s / np.r_[n[0], n[:-1]]                # ||z_{k+1} - z_k|| / ||z_k||
        ax["step"].plot(k, rel, **kw)
        ax["norm"].plot(k, n, **kw)
        n_tail = max(1, int(tail * len(s)))
        at_cap = np.nan
        if c["tau"] is not None:
            ax["tau"].plot(np.arange(1, len(c["tau"]) + 1), c["tau"], **kw)
            if c["cap"]:
                at_cap = float(np.isclose(c["tau"][-n_tail:], c["cap"], rtol=1e-6).mean())
        rows.append({"run": c["label"], "iterations": len(F), "converged": r.converged,
                     "last rel. step": rel[-1], "crit last / first": g[-1] / g[0],
                     "max ||z_k||": n.max(), "length": s.sum(),
                     f"tail length (last {tail:.0%})": s[-n_tail:].sum(),
                     f"at step cap (last {tail:.0%})": at_cap,
                     "F increases": int((np.diff(F) > 1e-12 * abs(F[0])).sum()),
                     "time [min]": r.time_s / 60 if r.time_s else np.nan})

    titles = {"F": r"$(F(z_k) - F^*) / |F^*|$",
              "crit": r"$\|G(z_k)\| / \|G(z_0)\|$,  $G(z) = z - P_+(z - \nabla F(z))$",
              "step": r"$\|z_{k+1} - z_k\| / \|z_k\|$", "norm": r"$\|z_k\|$",
              "tau": r"step size $\tau_k = 1/L_k$"}
    for key, a in ax.items():
        if key == "legend":
            continue
        if key in ("F", "crit", "step", "tau"):
            a.set_yscale("log")
        a.set_title(titles[key], fontsize=10)
        a.set_xlabel("iteration k")
        a.grid(alpha=0.3, which="both")
    for tol in {c["stop_tol"] for c in curves if c["stop_tol"] is not None}:
        ax["step"].axhline(tol, color="k", ls="--", lw=1)
        ax["step"].annotate(f"stop tol {tol:g}", (0.01, tol), xycoords=("axes fraction", "data"),
                            fontsize=8, va="bottom")
    for cap in {c["cap"] for c in curves if c["cap"]}:
        ax["tau"].axhline(cap, color="grey", ls=":", lw=1)
    handles, lbls = ax["F"].get_legend_handles_labels()
    ax["legend"].axis("off")
    ax["legend"].legend(handles, lbls, fontsize=9, frameon=False, loc="center left",
                        title="dashed: stopped at max_iter")
    first = sel.iloc[0]
    crop_txt = f" | crop {int(first.crop)}" if not pd.isna(first.crop) else ""
    fig.suptitle(f"Convergence | {data} (real data, flux {first.flux:g}){crop_txt}")
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()

    table = pd.DataFrame(rows).set_index("run")
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.float_format", "{:.3g}".format):
        print(table)
    return table


# ---------------------------------------------------------------------------
# Residual whiteness of the real runs (whiteness_real.py -> whiteness_real.csv)
# ---------------------------------------------------------------------------

WHITENESS_REAL_CSV = os.path.join(REAL_ROOT, "whiteness_real.csv")
_ITER_METHODS = {"RL"} | PNP_METHODS                # x axis = iterations, not lambda


def load_real_whiteness(path=WHITENESS_REAL_CSV):
    """whiteness_real.csv (one row per real-data result), finite results only."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run `python whiteness_real.py` first")
    df = pd.read_csv(path)
    return df[df.finite.astype(bool)].reset_index(drop=True)


def plot_real_whiteness(data, methods=None, metrics=("confocal", "stack"), mask="masked",
                        mode="standard", crop="any", show_z=False, wr=None, theme=None,
                        save=None):
    """
    Residual whiteness W of the chosen methods of `data`: W vs lambda (regularized methods)
    and vs iterations (RL, DRUNet), one line per (method, sigma, crop), the minimum of each
    line marked (hollow circle). W ~ 2 = white residual (dashed line).

    methods : exact method name(s), e.g. "TV_smooth", ["TV_smooth", "L1", "RL"]; None = all
    metrics : "confocal" (detectors summed, pixels only) and / or "stack" (25 det. x pixels)
    mask    : "masked" / "whole";  mode: "standard" / "highpass"
    crop    : None = full-image runs, an int = that run_real.py --crop, "any" = all
    show_z  : also Zmean (bias of the residual, ~0) and Zstd (~1 if the Poisson model fits)
    wr      : the table of load_real_whiteness() (default: read the csv)
    Returns the rows shown.
    """
    th = _theme(theme)
    wr = load_real_whiteness() if wr is None else wr
    g = wr[wr.data == data].copy()
    if methods is not None:
        g = g[g.method.isin([methods] if isinstance(methods, str) else methods)]
    if crop != "any":
        g = g[g.crop.isna()] if crop is None else g[g.crop == crop]
    if g.empty:
        print(f"no whiteness results for {data} {methods} (crop {crop}): "
              f"methods there: {sorted(wr[wr.data == data].method.unique())}")
        return g
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    rows = [("W", mt, f"W_{mt}_{mask}_{mode}") for mt in metrics]
    if show_z:
        rows += [(z, mt, f"{z}_{mt}_{mask}") for z in ("Zmean", "Zstd") for mt in metrics]
    is_it = g.method.isin(_ITER_METHODS)
    cols = [c for c, has in (("lambda", (~is_it).any()), ("iterations", is_it.any())) if has]

    # one line per (method, sigma, step, crop); colours by method in the order given
    order = ([methods] if isinstance(methods, str) else list(methods)) if methods is not None \
        else sorted(g.method.unique())
    color = {m: th["colors"][i % len(th["colors"])] for i, m in enumerate(order)}
    for k in ("sigma", "step_scale", "crop"):
        if k not in g:
            g[k] = np.nan
    lines = []
    for key, s in g.groupby(["method", "sigma", "step_scale", "crop"], dropna=False, sort=False):
        m, sig, step, cr = key
        lab = m + (f" σ={sig:g}" if not pd.isna(sig) else "") \
            + (f" step={step:g}" if not pd.isna(step) and m in _ITER_METHODS else "") \
            + (f" crop{int(cr)}" if not pd.isna(cr) else "")
        xcol = "iterations_run" if m in _ITER_METHODS else "lambda"
        lines.append((lab, m, xcol, s.sort_values(xcol)))
    styles, n_of = ["-", "--", ":", "-."], {}

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(6.2 * len(cols), 3.3 * len(rows)),
                             squeeze=False, sharex="col", layout="constrained")
    for i, (kind, mt, key) in enumerate(rows):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            for lab, m, xcol, s in lines:
                if (xcol == "iterations_run") != (col == "iterations") or key not in s:
                    continue
                ls = styles[n_of.setdefault(lab, sum(1 for l in n_of if l.split(" ")[0] == m))
                            % len(styles)]
                x, v = s[xcol].to_numpy(float), s[key].to_numpy(float)
                ax.plot(x, v, marker="o", ms=4, lw=1.7, ls=ls, color=color.get(m, th["ink"]),
                        label=lab)
                if kind == "W" and np.isfinite(v).any():
                    b = int(np.nanargmin(v))
                    ax.plot(x[b], v[b], "o", ms=11, mfc="none", mec=color.get(m, th["ink"]),
                            mew=1.8)
            ref = {"W": 2.0, "Zmean": 0.0, "Zstd": 1.0}[kind]
            ax.axhline(ref, color=th["raw"], ls="--", lw=1)
            ax.set_xscale("log")
            vals = g[key].dropna() if key in g else pd.Series(dtype=float)
            if kind == "W" and len(vals) and vals.max() > 20:
                ax.set_yscale("log")
            ax.grid(True, color=th["grid"], lw=0.6)
            ax.set_axisbelow(True)
            if j == 0:
                what = {"confocal": "confocal (Σ det.)", "stack": "stack (25 det.)"}[mt]
                ax.set_ylabel(f"{kind} · {what}")
            if i == len(rows) - 1:
                ax.set_xlabel("λ" if col == "lambda" else "iterations")
            if i == 0:
                ax.set_title("regularized methods (vs λ)" if col == "lambda"
                             else "RL / PnP (vs iterations)", fontsize=10)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{data} · residual whiteness ({mask}, {mode}) · hollow circle = min W, "
                 f"dashed = white residual", fontsize=11)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()
    show = ["method", "lambda", "sigma", "crop", "iterations_run"] + [k for _, _, k in rows]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format):
        print(g[[c for c in show if c in g]].sort_values(["method", "lambda", "iterations_run"])
              .to_string(index=False))
    return g

"""
Diagnostics of the simulated runs: PSNR per iteration (+ Richardson-Lucy), the functional
F(x_k), the convergence check, and the points tried by the parameter search.
"""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .common import _as_list, _match_method, _method_colors
from .sim import ROOT, UNREG_DIR, _subset, _select, load_rec


# ---------------------------------------------------------------------------
# PSNR per iteration
# ---------------------------------------------------------------------------

def plot_psnr_iterations(df, data, flux, methods=("TV-smooth [PROJ_GD] KL", "TV [PROX_GD] KL"),
                         lam=None, rl=True, root=ROOT, ax=None, show=True, metric="psnr"):
    """
    PSNR per iteration (iterate clamped to [0, 1], deepinv's metrics["psnr"]) of some methods
    at a chosen lambda, as for the Richardson-Lucy iterations; RL is added on top.
    metric  : "psnr" (dB) or "mse" = mean squared error of the clamped iterate vs the
              ground truth, both in [0, 1] units: MSE = 10^(-PSNR / 10) exactly (deepinv
              PSNR with max_pixel = 1), log scale; best = min MSE.

    df      : the FULL table (load(...), not best_per_method), so every lambda run is there.
              To include the lambda = 0 runs of Results/unregularized:
                  df_all = pd.concat([df.assign(root=ROOT),
                                      load(UNREG_DIR, split_by_fidelity=False).assign(root=UNREG_DIR)])
    methods : method name(s) or unambiguous substring(s); one string = one method
    lam     : one lambda for all methods, a dict {method: lambda} (missing = best PSNR),
              or None = the best-PSNR run of each method. 1% tolerance, as get_image.
              Only lambdas that were run are on disk (search best or single_experiment.py).
    rl      : also the RL curve (rl_simulated.py, PSNR clamped as the others), or a path
    Returns the axis.
    """
    if metric not in ("psnr", "mse"):
        raise ValueError(f"metric must be 'psnr' or 'mse', not {metric!r}")
    methods = _as_list(methods)                    # a single string: not its characters
    g = _subset(df, data, None)
    g = g[g.flux == flux]
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300"]

    def draw(psnr_curve, label, color, ls="-"):
        """One curve in the chosen metric, its best iteration marked."""
        p = np.asarray(psnr_curve, float)
        if metric == "mse":
            v, k, fmt = 10 ** (-p / 10), int(np.nanargmax(p)), "{:.3g}"
            best = "min"
        else:
            v, k, fmt = p, int(np.nanargmax(p)), "{:.2f} dB"
            best = "max"
        ax.plot(np.arange(1, len(v) + 1), v, color=color, lw=1.8, ls=ls,
                label=f"{label}  ({best} {fmt.format(v[k])} at {k + 1}, "
                      f"last {fmt.format(v[-1])} at {len(v)})")
        ax.plot(k + 1, v[k], "o", ms=7, color=color, mec="white", mew=1.2)
    for method, color in zip(methods, colors):
        try:
            name = _match_method(g, method)
        except ValueError:
            print(f"{method}: not saved for {data}, flux {flux} (skipped)")
            continue
        rows = g[g.method == name]
        l = lam.get(method, lam.get(name)) if isinstance(lam, dict) else lam
        if l is not None:
            rows = rows[np.isclose(rows["lambda"].astype(float), float(l), rtol=1e-2, atol=1e-12)]
            if rows.empty:
                have = sorted(g[g.method == name]["lambda"].dropna().astype(float).unique())
                print(f"{name}: no run with lambda={l:g} for {data}, flux {flux}; "
                      f"available: {[float(f'{v:.4g}') for v in have]}")
                continue
        r = rows.loc[rows.psnr.idxmax()]
        res = load_rec(r, root)
        hist = (res.get("metrics") or {}).get("psnr")
        if not hist:
            print(f"{name} (lambda={r['lambda']:.4g}): no per-iteration PSNR saved")
            continue
        draw(hist[0], f"{name.replace(' KL', '')}, λ = {r['lambda']:.3g}", color)
    if rl:
        path = rl if isinstance(rl, str) else os.path.join(
            UNREG_DIR, f"RL_{data}_flux{flux:g}_seed0.pth")
        if os.path.exists(path):
            d = torch.load(path, map_location="cpu", weights_only=False)
            draw(d["psnr_clamp"], "Richardson–Lucy, no reg", "#1f1f1e", ls="--")
        else:
            print(f"no RL result {path} (python rl_simulated.py --images {data})")
    ax.set_xscale("log")
    ax.set_xlabel("iteration")
    if metric == "mse":
        ax.set_yscale("log")
        ax.set_ylabel("MSE vs ground truth, iterate clamped to [0, 1]")
    else:
        ax.set_ylabel("PSNR (dB), iterate clamped to [0, 1]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    if show:
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Convergence: functional F(x_k) = f + lam R per iteration
# ---------------------------------------------------------------------------

def plot_functional(df, data, flux, fidelity=None, methods=None, root=ROOT,
                    kind="gap", rates=(1,), save=None):
    """
    Functional F(x_k) saved by the explicit-prior iterations (metrics["cost"]),
    one curve per method.

    kind    : "gap" = (F(x_k) - F*) / |F*| on log-log axes, F* = min of each curve
              (the last iterates are then biased towards 0: F* is not the true minimum);
              "F"   = F(x_k) itself (each method has its own lambda, R and fidelity:
              compare the shapes, not the levels)
    rates   : "gap" only: reference slopes C / k^p for every p, e.g. (1, 2)
    methods : substrings of method names, None = all methods with a cost history
    Curves from files saved before the float64 fix (no "cost_def") are dashed:
    float32 smooth part (data term only for L1 / PROX_GD), floor ~1e-7 relative.
    """
    g = _select(df, data, flux, fidelity, methods)
    curves = {}
    for r in g.itertuples():
        res = load_rec(r, root)
        cost = (res.get("metrics") or {}).get("cost")
        if not cost or not len(cost[0]):
            continue                    # PnP / RED: no explicit functional
        curves[r.method] = (np.asarray(cost[0], dtype=np.float64),
                            res.get("cost_def") == "F_fp64")
    if not curves:
        raise ValueError(f"no cost history for data={data}, flux={flux}")

    colors = _method_colors(list(curves))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    k_max, gap_max = 1, 0.0
    for method, (F, new) in curves.items():
        k = np.arange(1, len(F) + 1)    # cost[k-1] = F after k iterations
        y = F if kind == "F" else (F - F.min()) / abs(F.min())
        if kind != "F":
            y = np.where(y > 0, y, np.nan)      # log axis: drop the exact minimum
            gap_max = max(gap_max, np.nanmax(y[:10]))
        k_max = max(k_max, len(F))
        ax.plot(k, y, color=colors[method], lw=1.6, ls="-" if new else "--",
                label=method + ("" if new else "  (old fp32)"))

    ax.set_xscale("log")
    ax.set_xlabel("iteration k")
    if kind == "F":
        ax.set_ylabel(r"$F(x_k)$")
    else:
        ax.set_yscale("log")
        ax.set_ylabel(r"$(F(x_k) - F^*) / |F^*|$")
        kk = np.array([1.0, k_max])
        for p in rates:                 # anchored at the largest gap of the first iterates
            ax.plot(kk, gap_max * kk ** -p, color="0.5", lw=1, ls=":")
            ax.annotate(rf"$k^{{-{p}}}$", (kk[-1], gap_max * kk[-1] ** -p), color="0.4",
                        fontsize=8, ha="left", va="center")
    ax.grid(alpha=0.3, which="both")
    fig.legend(fontsize=8, frameon=False, loc="outside right center")
    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    ax.set_title(f"{data} | flux {flux:g}{fid}")
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()
    return curves


def convergence_check(df, data, flux, fidelity=None, methods=None, root=ROOT,
                      tail=0.1, save=None):
    """
    Empirical check of a KL-type convergence theorem (bounded iterates -> finite length,
    convergence to a critical point) on the iterates z_k (unclamped) of each method.
    Needs the histories written with log_cost (refresh_histories.py / new runs).

    Panels (linear iteration axis):
      PSNR(z_k)               deepinv PSNR of z_k clamped to [0, 1], NOT divided by its
                              max (the tables divide by the max: values differ)
      (F - F*)/|F*|           F* = min of each curve
      ||G(z_k)|| / ||G(z_0)|| G(z) = z - P_{>=0}(z - grad F(z)), projected gradient: the
                              constrained version of ||grad F|| (z >= 0), 0 iff critical
      ||z_{k+1} - z_k|| / ||z_k|| (the quantity of the early stop, its tolerance dashed);
      ||z_k|| (bounded); step size tau_k = 1 / L_k (a small step norm
      with tau_k at the fallback 1 / L_max means a stalled backtracking, not convergence).
    The table also gives the length sum ||z_{k+1} - z_k||.
    Prints a summary table and returns it; tail = fraction of the last iterations used
    for "tail length" and "fallback %".
    """
    g = _select(df, data, flux, fidelity, methods)
    runs, skipped = {}, []
    for r in g.itertuples():
        res = load_rec(r, root)
        m = res.get("metrics") or {}
        keys = ("psnr", "cost", "step_norm", "iterate_norm", "criticality")
        if not all(k in m for k in keys):
            skipped.append(r.method)
            continue
        runs[r.method] = {k: np.asarray(m[k][0], dtype=np.float64) for k in keys}
        L = res.get("L_history")
        runs[r.method]["tau"] = 1.0 / np.asarray(L, dtype=np.float64) if L else None
        runs[r.method]["stop_tol"] = (res.get("config") or {}).get("stop_tol")
    if skipped:
        print("no convergence history (run refresh_histories.py):\n  " + "\n  ".join(skipped))
    if not runs:
        raise ValueError(f"no convergence histories for data={data}, flux={flux}")

    colors = _method_colors(list(runs))
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), layout="constrained")
    ax = dict(zip(["psnr", "F", "crit", "step", "norm", "tau"], axes.flat))
    rows = []
    for method, h in runs.items():
        F, s, n, c, tau = h["cost"], h["step_norm"], h["iterate_norm"], h["criticality"], h["tau"]
        k = np.arange(1, len(F) + 1)
        kw = dict(color=colors[method], lw=1.4, label=method)
        ax["psnr"].plot(np.arange(len(h["psnr"])), h["psnr"], **kw)   # k = 0 .. N
        gap = (F - F.min()) / abs(F.min())
        ax["F"].plot(k, np.where(gap > 0, gap, np.nan), **kw)
        ax["crit"].plot(np.arange(len(c)), c / c[0], **kw)   # at z_k, k = 0 .. N-1
        # relative step of the early stop: ||z_{k+1} - z_k|| / ||z_k||, with ||z_k|| =
        # iterate_norm[k-1]; ||z_0|| is not saved: ||z_1|| is used for the first step
        rel = s / np.r_[n[0], n[:-1]]
        ax["step"].plot(k, rel, **kw)
        length = np.cumsum(s)
        ax["norm"].plot(k, n, **kw)                       # ||z_k||, k >= 1
        n_tail = max(1, int(tail * len(s)))
        fallback = np.nan
        if tau is not None:
            ax["tau"].plot(np.arange(1, len(tau) + 1), tau, **kw)
            fallback = float(np.isclose(tau[-n_tail:], tau.min(), rtol=1e-6).mean())
        rows.append({"method": method, "max ||z_k||": n.max(), "length": length[-1],
                     f"tail length (last {tail:.0%})": s[-n_tail:].sum(),
                     "last rel. step": rel[-1], "crit last / first": c[-1] / c[0],
                     f"fallback steps (last {tail:.0%})": fallback,
                     "F increases": int((np.diff(F) > 1e-12 * abs(F[0])).sum())})

    labels = {"psnr": r"PSNR ($z_k$ clamped to [0, 1])", "F": r"$(F(z_k) - F^*) / |F^*|$",
              "crit": r"$\|G(z_k)\| / \|G(z_0)\|$,  $G(z) = z - P_+(z - \nabla F(z))$",
              "step": r"$\|z_{k+1} - z_k\| / \|z_k\|$", "norm": r"$\|z_k\|$",
              "tau": r"step size $\tau_k = 1/L_k$"}
    for key, a in ax.items():
        if key in ("F", "crit", "step", "tau"):
            a.set_yscale("log")
        a.set_title(labels[key], fontsize=10)
        a.set_xlabel("iteration k")
        a.grid(alpha=0.3, which="both")
    for tol in {h["stop_tol"] for h in runs.values() if h["stop_tol"] is not None}:
        ax["step"].axhline(tol, color="k", ls="--", lw=1)
        ax["step"].annotate(f"stop tol {tol:g}", (0.01, tol), xycoords=("axes fraction", "data"),
                            fontsize=8, va="bottom")
    handles, lbls = ax["F"].get_legend_handles_labels()
    fig.legend(handles, lbls, fontsize=8, frameon=False, loc="outside right center")
    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    fig.suptitle(f"Convergence check | {data} | flux {flux:g}{fid}")
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()

    table = pd.DataFrame(rows).set_index("method")
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.float_format", "{:.3g}".format):
        print(table)
    return table


# ---------------------------------------------------------------------------
# Parameter search
# ---------------------------------------------------------------------------

def search_points(res):
    """Every point evaluated by the search of one result, as a DataFrame with the
    searched parameters and psnr. Adaptive search: res["evaluations"]; grid search:
    the finite entries of res["psnr_map"] (axes res["grid_axes"])."""
    if res.get("evaluations"):
        return pd.DataFrame(res["evaluations"])
    pm = res.get("psnr_map")
    if pm is None:
        raise ValueError("this result has neither 'evaluations' nor 'psnr_map'")
    axes = res.get("grid_axes", ("sigma", "step", "iter", "lambda"))
    grids = {"sigma": res["sigma_grid"], "step": res["step_grid"],
             "iter": res["iter_grid"], "lambda": res["lambda_grid"]}
    idx = torch.nonzero(torch.isfinite(pm), as_tuple=False)
    cols = {a: torch.as_tensor(grids[a]).flatten()[idx[:, i]].double().numpy()
            for i, a in enumerate(axes)}
    cols["psnr"] = pm[torch.isfinite(pm)].double().numpy()
    return pd.DataFrame(cols)


def plot_search(df, data, flux, method, fidelity=None, root=ROOT, save=None):
    """
    Every point tried by the parameter search of one method: PSNR against each searched
    parameter (log axis; iter linear), coloured by evaluation order (adaptive search),
    the chosen point as a red star. Diverged runs are not shown.

    method : full method name or an unambiguous substring, e.g. "WCRR_ISM [PROJ_GD] KL"
    Works for the adaptive search (evaluate_methods.py) and the old grid search.
    """
    g = _subset(df, data, fidelity)
    name = _match_method(g, method)
    g = g[(g.method == name) & (g.flux == flux)]
    if g.empty:
        raise ValueError(f"no results for {name}, data={data}, flux={flux}")
    r = g.sort_values("psnr", ascending=False).iloc[0]
    res = load_rec(r, root)
    ev = search_points(res)
    ev = ev[np.isfinite(ev.psnr)]
    chosen = {"sigma": res.get("sigma"), "lambda": res.get("lambda"),
              "step": res.get("step_scale"), "iter": res.get("iter")}
    params = [p for p in ("sigma", "lambda", "step", "iter") if p in ev and ev[p].nunique() > 1]
    if not params:
        raise ValueError(f"{name}: only one point was evaluated")
    adaptive = bool(res.get("evaluations"))

    fig, axes = plt.subplots(1, len(params), figsize=(4.5 * len(params) + 1, 4),
                             layout="constrained", squeeze=False)
    for ax, p in zip(axes[0], params):
        sc = ax.scatter(ev[p], ev.psnr, c=np.arange(len(ev)) if adaptive else "tab:blue",
                        cmap="viridis" if adaptive else None, s=18)
        if chosen[p] is not None:
            ax.scatter([chosen[p]], [res["psnr"]], color="r", s=70, marker="*", label="chosen")
        if p != "iter":
            ax.set_xscale("log")
        ax.set_xlabel(p)
        ax.set_ylabel("PSNR")
        ax.grid(alpha=0.3)
    if adaptive:
        fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8).set_label("evaluation order")
    axes[0, 0].legend(fontsize=8)
    hit = [a for a, h in (res.get("hit_bound") or {}).items() if h]
    fig.suptitle(f"Parameter search | {name} | {data} | flux {flux:g} | {len(ev)} runs"
                 + (f" | at search limit: {', '.join(hit)}" if hit else ""))
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()
    return ev

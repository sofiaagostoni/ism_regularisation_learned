"""
Simulated results (evaluate_methods.py / single_experiment.py): loading the tables
(summary_all.csv + config.json of each run, several folders with load_merged), filters,
text tables, colour-coded tables (heatmaps, metrics vs flux, styled_table), and the
result files of one row (load_rec, sum_measurements, get_image).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from opt_functions.Solver_functions.prior_config import REG_PLANE
from .common import (PRIOR_NAMES, EXPLICIT, PRIORS_WITHOUT_SIGMA, PRIORS_USING_LOSS,
                     METRIC_LABELS, METRIC_FMT, _fmt, _as_list, _plane_label, to_img, _crop,
                     _match_method, _method_colors)


ROOT = "Results/evaluation_stop"            # evaluate_methods.py; old grid search: "Results/results_gridsearch"


UNREG_DIR = "Results/unregularized"          # rl_simulated.py / single_experiment.py --lam 0


# ---------------------------------------------------------------------------
# Loading and text tables
# ---------------------------------------------------------------------------

def run_config(root, run_id):
    """config.json of a run ({} if missing)."""
    path = os.path.join(root, str(run_id), "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


SINGLE_PREFIX = "single_"     # run folders of single_experiment.py (chosen parameters)


def run_search(root, run_id):
    """Search type of a run: 'single' (single_experiment.py, folder single_<hash>), else
    from its config.json: 'adaptive' or 'grid'."""
    if str(run_id).startswith(SINGLE_PREFIX):
        return "single"
    return run_config(root, run_id).get("search", "grid")


def outdated_reason(cfg, fidelity):
    """Why a run (its config.json, fidelity) no longer matches the current model / solver,
    "" if it does:
      "bkg"  KL / L2 run with the old background model (L2 without background, KL with
             a background flux times too small); current runs have "fid_bkg"
      "bt"   explicit prior with the old float32 backtracking test (stalls near
             convergence); current runs have "bt_test" in config.json"""
    reasons = []
    if fidelity in ("KL", "L2") and not cfg.get("fid_bkg"):
        reasons.append("bkg")
    if (cfg.get("backtracking") == "custom" and cfg.get("prior_name") in EXPLICIT
            and not cfg.get("bt_test")):
        reasons.append("bt")
    return "+".join(reasons)


def load(root=ROOT, run_ids=None, exclude_legacy=False, split_by_fidelity=True, search=None,
         outdated=False, include_single=True):
    """split_by_fidelity=False: compare KL and L2 runs in the same tables
    (the fidelity is then shown in the method name).
    search: 'adaptive' = only the runs of evaluate_methods.py (adaptive search),
    'grid' = only the fixed-grid runs, None = all.
    include_single: with search set, also keep the single_experiment.py runs (same method
    names, so best_per_method picks the best among the search and the single runs).
    Column "search" = 'adaptive' / 'grid' / 'single' of each row.
    outdated: runs that no longer match the current model / solver (outdated_reason) are
    dropped (False) or kept with " {old:<reason>}" appended to the method name (True);
    column "outdated" = the reason ("" for current runs)."""
    df = pd.read_csv(os.path.join(root, "summary_all.csv"))
    for col in ("step_scale", "stepsize", "prior"):        # missing in old files
        if col not in df:
            df[col] = "" if col == "prior" else float("nan")
    if run_ids is not None:
        df = df[df.run_id.isin(run_ids)]
    if exclude_legacy:
        df = df[~df.run_id.astype(str).str.startswith("legacy_")]
    kind = {r: run_search(root, r) for r in df.run_id.unique()}
    df["search"] = df.run_id.map(kind)
    if search is not None:
        allowed = {search} | ({"single"} if include_single else set())
        df = df[df.search.isin(allowed)]
        if df.empty:
            raise ValueError(f"no finished {search}-search results in {root}")
    # raw values are needed to rebuild the result file names
    df["fidelity_raw"] = df.fidelity.astype(str)
    df["loss_raw"] = df.loss.astype(str)
    df["fidelity"] = df.fidelity.astype(str).str.upper()
    cfgs = {r: run_config(root, r) for r in df.run_id.unique()}
    df["outdated"] = [outdated_reason(cfgs[r], f) for r, f in zip(df.run_id, df.fidelity)]
    if not outdated:
        df = df[df.outdated == ""].copy()
    if df.empty:
        raise ValueError(f"no current results in {root} (outdated=True keeps the runs with "
                         "the old background model / backtracking test)")
    df["loss"] = df.loss.fillna("none").astype(str)
    df["fid"] = df.fidelity               # KL / L2 / KL0
    df["method"] = df.apply(method_name, axis=1)
    if not split_by_fidelity:
        df["method"] = df.method + " " + df.fidelity
        df["fidelity"] = "all"
    old = df.outdated != ""
    df.loc[old, "method"] = df.loc[old, "method"] + " {old:" + df.loc[old, "outdated"] + "}"
    df["params"] = df.apply(params_str, axis=1)
    return df


def method_name(r):
    name = PRIOR_NAMES.get(r.prior, r.prior)
    if r.prior in PRIORS_USING_LOSS:
        name += f" ({r.loss})"
    return f"{name} [{r.algo}]"


def params_str(r):
    """The parameters searched for this method."""
    parts = []
    if r.prior not in PRIORS_WITHOUT_SIGMA:
        parts.append(f"sigma={_fmt(r.sigma, '.3g')}")
    if r.prior in EXPLICIT or (pd.notna(r["lambda"]) and r["lambda"] != 1):
        parts.append(f"lambda={_fmt(r['lambda'], '.3g')}")
    if pd.notna(r.step_scale) and (r.prior not in EXPLICIT or r.step_scale != 1):
        parts.append(f"step x{_fmt(r.step_scale, '.3g')}")
    parts.append(f"iter={_fmt(r.iter, '.0f')}")
    if pd.notna(r.stepsize):
        parts.append(f"tau={_fmt(r.stepsize, '.2e')}")
    return "  ".join(parts)


def best_per_method(df):
    """If a method was run with several grids, keep its best PSNR per (data, fidelity, flux)."""
    keys = ["data", "fidelity", "flux", "method"]
    return df.loc[df.groupby(keys).psnr.idxmax()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Several results folders together (same run_experiment and file format)
# ---------------------------------------------------------------------------

#   eval    evaluate_methods.py (KL, KL0 and L2 runs)
# Add other folders here to merge them (the method names get " {source}" when a method
# is in more than one folder, see load_merged).
SOURCES = {"eval": "Results/evaluation_stop"}


def load_merged(sources=SOURCES, on_duplicate="tag", search="adaptive", outdated=False,
                include_single=True):
    """
    All results of several folders in one table (KL and L2 together: method names end
    with the fidelity, e.g. "WCRR [PROJ_GD] KL"). Extra columns: source, root (folder of
    the result files, used by load_rec), fid (KL / L2 / KL0).

    on_duplicate: what to do when a method exists in more than one source
        "tag"      keep all; the method name gets " {source}" for every source but the
                   first in `sources` (e.g. "WCRR [PROJ_GD] KL {other}")
        "<source>" keep only that source's row where both have the same
                   (data, flux, method); the other source fills the remaining images
    """
    parts = []
    for name, root in sources.items():
        if not os.path.exists(os.path.join(root, "summary_all.csv")):
            print(f"skip source {name}: no {root}/summary_all.csv")
            continue
        try:
            d = load(root, split_by_fidelity=False, search=search, outdated=outdated,
                     include_single=include_single)
        except ValueError as e:
            print(f"skip source {name}: {e}")
            continue
        d["source"], d["root"] = name, root
        parts.append(d)
    if not parts:
        raise ValueError("no results in any source")
    df = pd.concat(parts, ignore_index=True)
    first = next(iter(sources))
    shared = {m for m, g in df.groupby("method") if g.source.nunique() > 1}
    if on_duplicate == "tag":
        dup = df.method.isin(shared) & (df.source != first)
        df.loc[dup, "method"] = df.loc[dup, "method"] + " {" + df.loc[dup, "source"] + "}"
    elif on_duplicate in sources:
        keys = ["data", "flux", "method"]
        has_pref = df[df.source == on_duplicate].set_index(keys).index
        df = df[(df.source == on_duplicate) | ~df.set_index(keys).index.isin(has_pref)]
    else:
        raise ValueError(f"on_duplicate must be 'tag' or one of {list(sources)}")
    if shared:
        print(f"methods in more than one source ({on_duplicate}): {sorted(shared)}")
    return df.reset_index(drop=True)


def pick(df, data=None, priors=None, fidelities=None, flux=None, sources=None, algos=None):
    """Filter a table. Every argument is optional and takes one value or a list:
    data ("tub", ...), priors ("WCRR", "TV_smooth", "TV_nonsmooth", "L1", ... exact prior
    names), fidelities ("KL", "L2", "KL0"), flux, sources (keys of SOURCES), algos."""
    m = pd.Series(True, index=df.index)
    fid = df["fid"] if "fid" in df else df.fidelity_raw.str.upper()
    for col, vals in [("data", data), ("prior", priors), ("flux", flux), ("algo", algos)]:
        if vals is not None:
            m &= df[col].isin(_as_list(vals))
    if fidelities is not None:
        m &= fid.isin([f.upper() for f in _as_list(fidelities)])
    if sources is not None and "source" in df:
        m &= df.source.isin(_as_list(sources))
    out = df[m]
    if out.empty:
        raise ValueError("pick(): nothing left with these filters")
    return out


def fidelity_table(df, fidelities=("KL", "L2"), metrics=("psnr", "ssim", "microssim"),
                   ref=None):
    """
    One row per (data, flux, prior, algo, source), one column per (metric, fidelity),
    plus the difference of each fidelity to ref (default: the first one), e.g.
    "psnr L2-KL" > 0 = L2 better. Rows where a fidelity is missing are kept (NaN).
    """
    fidelities = [f.upper() for f in fidelities]
    ref = (ref or fidelities[0]).upper()
    d = df.copy()
    d["fid"] = d["fid"] if "fid" in d else d.fidelity_raw.str.upper()
    d = d[d.fid.isin(fidelities)]
    if "source" not in d:
        d["source"] = "-"
    keys = ["data", "flux", "prior", "algo", "source"]
    # the algo differs across fidelities for MD (KL0 + MD_BURG): compare by prior only then
    if d.groupby(["data", "flux", "prior", "source"]).algo.nunique().max() > 1:
        keys.remove("algo")
    d = d.loc[d.groupby(keys + ["fid"]).psnr.idxmax()]
    t = d.pivot_table(index=keys, columns="fid", values=list(metrics), aggfunc="first")
    for mt in metrics:
        for f in fidelities:
            if f != ref and (mt, f) in t and (mt, ref) in t:
                t[(mt, f"{f}-{ref}")] = t[(mt, f)] - t[(mt, ref)]
    order = [(mt, c) for mt in metrics
             for c in fidelities + [f"{f}-{ref}" for f in fidelities if f != ref]
             if (mt, c) in t]
    t = t[order]
    t.index = t.index.set_names(keys)
    return t.rename(index=lambda v: PRIOR_NAMES.get(v, v), level="prior")


def print_ranking(df):
    """One table per (data, fidelity, flux): methods ranked by PSNR."""
    cols = ["method", "psnr", "ssim", "microssim", "params", "search", "run_id"]
    for (data, fid, flux), g in df.groupby(["data", "fidelity", "flux"]):
        g = g.sort_values("psnr", ascending=False)[cols]
        print(f"\n=== {data} | fidelity {fid} | flux {flux} ===")
        print(g.to_string(index=False, formatters={
            "psnr": "{:.2f}".format, "ssim": "{:.4f}".format, "microssim": "{:.4f}".format}))


def print_comparison(df, metric="psnr"):
    """Methods x flux table of one metric, per (data, fidelity); best value marked with *."""
    fmt = "{:.2f}" if metric == "psnr" else "{:.4f}"
    for (data, fid), g in df.groupby(["data", "fidelity"]):
        t = g.pivot_table(index="method", columns="flux", values=metric, aggfunc="max")
        t["mean"] = t.mean(axis=1)
        t = t.sort_values("mean", ascending=False)
        shown = t.copy().astype(object)
        for c in t.columns:
            best = t[c].max()
            shown[c] = [("-" if pd.isna(v) else fmt.format(v) + ("*" if v == best else " "))
                        for v in t[c]]
        print(f"\n=== {metric.upper()} | {data} | fidelity {fid}   (* = best per column) ===")
        print(shown.to_string())


def print_params(df):
    """Best parameters per method and flux, per (data, fidelity)."""
    for (data, fid), g in df.groupby(["data", "fidelity"]):
        t = g.pivot_table(index="method", columns="flux", values="params", aggfunc="first")
        print(f"\n=== best parameters | {data} | fidelity {fid} ===")
        for method, row in t.iterrows():
            print(f"\n{method}")
            for flux, p in row.items():
                if isinstance(p, str):
                    print(f"  flux {flux:>4}: {p}")


# ---------------------------------------------------------------------------
# Visual tables
# ---------------------------------------------------------------------------


def _subset(df, data, fidelity):
    g = df[df.data == data]
    if fidelity is not None:
        g = g[g.fidelity == fidelity.upper()]
    if g.empty:
        raise ValueError(f"no results for data={data}, fidelity={fidelity}")
    return g


def metric_tables(df, data, fidelity=None, metrics=("psnr", "ssim", "microssim")):
    """{metric: methods x flux table (+ mean column)}, rows ordered by mean PSNR.
    Metrics with no values at all (e.g. MicroSSIM not installed) are dropped."""
    g = _subset(df, data, fidelity)
    tables = {}
    for m in metrics:
        t = g.pivot_table(index="method", columns="flux", values=m, aggfunc="max")
        if t.isna().all().all():
            continue
        t["mean"] = t.mean(axis=1)
        tables[m] = t
    order = tables[next(iter(tables))]["mean"].sort_values(ascending=False).index
    return {m: t.reindex(order) for m, t in tables.items()}


def plot_metric_heatmaps(df, data, fidelity=None, metrics=("psnr", "ssim", "microssim"),
                         save=None):
    """
    One heatmap per metric: methods (rows) x flux (columns).
    Colour = rank within each column (green = best, red = worst), because the
    metrics grow with the flux and a global colour scale would only show that.
    The best value of each column is boxed and bold.
    """
    tables = {m: t.drop(columns="mean")
              for m, t in metric_tables(df, data, fidelity, metrics).items()}
    first = next(iter(tables.values()))
    n_rows, n_cols = first.shape

    fig, axes = plt.subplots(1, len(tables), sharey=True, layout="constrained",
                             figsize=(3.3 + len(tables) * (1.2 + 0.9 * n_cols),
                                      0.48 * n_rows + 1.8), squeeze=False)
    for ax, (m, t) in zip(axes[0], tables.items()):
        v = t.values.astype(float)
        lo, hi = np.nanmin(v, axis=0), np.nanmax(v, axis=0)
        tied = ~(hi > lo)                                    # all values equal: neutral
        rank = np.where(np.isnan(v), np.nan,
                        np.where(tied, 0.5, (v - lo) / np.where(tied, 1.0, hi - lo)))
        ax.imshow(np.ma.masked_invalid(rank), cmap="RdYlGn", vmin=-0.15, vmax=1.15,
                  aspect="auto")
        for r in range(n_rows):
            for c in range(n_cols):
                if np.isnan(v[r, c]):
                    ax.text(c, r, "-", ha="center", va="center", fontsize=8, color="0.5")
                    continue
                best = v[r, c] == hi[c] and not tied[c]
                ax.text(c, r, METRIC_FMT[m].format(v[r, c]), ha="center", va="center",
                        fontsize=8.5, fontweight="bold" if best else "normal")
                if best:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=2, ec="k"))
        ax.set_xticks(range(n_cols), [f"flux {float(f):g}" for f in t.columns], fontsize=9)
        ax.set_title(METRIC_LABELS[m], fontsize=11, fontweight="bold")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0, 0].set_yticks(range(n_rows), list(first.index), fontsize=9)
    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    fig.suptitle(f"{data}{fid}   (colour = rank within each column, box = best)", fontsize=11)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()


def plot_metric_vs_flux(df, data, fidelity=None, metrics=("psnr", "ssim", "microssim"),
                        relative_to=None, save=None):
    """
    Each metric as a function of the flux, one line per method.
    relative_to: substring of a method name, e.g. "DRUNet [": plot the
    difference to that method instead (easier to read: the gaps are small).
    """
    tables = metric_tables(df, data, fidelity, metrics)
    methods = list(next(iter(tables.values())).index)
    ref = None
    if relative_to is not None:
        ref = next((m for m in methods if relative_to in m), None)
        if ref is None:
            raise ValueError(f"no method matches '{relative_to}': {methods}")
    colors = _method_colors(methods)
    markers = "osD^v<>Ph*X"
    fig, axes = plt.subplots(1, len(tables), figsize=(4.6 * len(tables) + 2.5, 4),
                             layout="constrained", squeeze=False)
    for ax, (m, t) in zip(axes[0], tables.items()):
        flux = [c for c in t.columns if c != "mean"]
        if ref is not None:
            t = t[flux] - t.loc[ref, flux]
            ax.axhline(0, color="k", lw=1)
        for k, method in enumerate(methods):
            ax.plot(flux, t.loc[method, flux], marker=markers[k % len(markers)],
                    color=colors[method], lw=1.8, ms=5,
                    ls="--" if method.endswith(" L2") else "-", label=method)
        ax.set_xscale("log")
        ax.set_xticks(flux, [str(f) for f in flux])
        ax.minorticks_off()
        ax.set_xlabel("flux")
        ax.set_title(METRIC_LABELS[m] if ref is None else f"\u0394 {METRIC_LABELS[m]}",
                     fontweight="bold")
        ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", fontsize=8, frameon=False)
    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    fig.suptitle(f"{data}{fid}" + (f"   (difference to {ref})" if ref else ""))
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()


def plot_fidelity_vs_flux(df, data, priors=None, metrics=("psnr", "ssim", "microssim"),
                          algos=None, sources=None, save=None):
    """
    KL vs L2 as a function of the flux, for one image (data = one name; a list: one figure
    per image): each metric vs flux, one colour per prior, KL solid / L2 dashed.
    df     : best_per_method(load_merged(...)) or best_per_method(load(..., split_by_fidelity=False))
    priors : exact prior names, e.g. ["WCRR", "TV_smooth"] (None = all with KL or L2)
    algos  : e.g. "PROJ_GD" (None = all); sources: keys of SOURCES (None = all)
    Fluxes where one fidelity is missing are shown for the other one only.
    """
    for d in _as_list(data):
        try:
            sel = pick(df, data=d, priors=priors, fidelities=["KL", "L2"], algos=algos,
                       sources=sources)
        except ValueError:
            print(f"plot_fidelity_vs_flux: no KL / L2 results for {d} yet, skipped")
            continue
        t = fidelity_table(sel, fidelities=("KL", "L2"), metrics=metrics).reset_index()
        t["label"] = t.prior + " [" + t.algo + "]" if "algo" in t else t.prior
        if "source" in t and t.source.nunique() > 1:
            t["label"] = t.label + " {" + t.source + "}"
        labels = list(dict.fromkeys(t.label))
        colors = dict(zip(labels, plt.cm.tab10.colors))
        fluxes = sorted(t.flux.unique())

        fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics) + 2.5, 3.8),
                                 layout="constrained", squeeze=False)
        for j, mt in enumerate(metrics):
            ax = axes[0, j]
            for lab in labels:
                g = t[t.label == lab].sort_values("flux")
                for fid, ls, mk in (("KL", "-", "o"), ("L2", "--", "s")):
                    if (mt, fid) in g:
                        ax.plot(g.flux, g[(mt, fid)], ls=ls, marker=mk, ms=5, lw=1.8,
                                color=colors[lab])
            ax.set_title(METRIC_LABELS.get(mt, mt), fontweight="bold")
            ax.grid(alpha=0.3)
            ax.set_xscale("log")
            ax.set_xticks(fluxes, [f"{f:g}" for f in fluxes])
            ax.minorticks_off()
            ax.set_xlabel("flux")

        from matplotlib.lines import Line2D
        handles = ([Line2D([], [], color=colors[lab], lw=2.5, label=lab) for lab in labels]
                   + [Line2D([], [], color="k", ls="-", marker="o", label="KL"),
                      Line2D([], [], color="k", ls="--", marker="s", label="L2")])
        fig.legend(handles=handles, loc="outside right center", fontsize=8, frameon=False)
        fig.suptitle(f"KL vs L2 | {d}")
        if save:
            root_, ext_ = os.path.splitext(save)
            fig.savefig(f"{root_}_{d}{ext_ or '.png'}" if len(_as_list(data)) > 1 else save,
                        dpi=200, bbox_inches="tight")
        plt.show()


def styled_table(df, data, fidelity=None, metrics=("psnr", "ssim", "microssim")):
    """
    Colour-coded table for the VS Code interactive window / Jupyter:
        display(styled_table(best, "tub_level"))
    Export: .to_html("table.html") or .to_latex(convert_css=True) for a paper.
    """
    tables = metric_tables(df, data, fidelity, metrics)
    big = pd.concat({METRIC_LABELS[m]: t for m, t in tables.items()}, axis=1)
    big.columns = big.columns.set_names(["", "flux"])
    big.index.name = None
    fmt = {col: METRIC_FMT[m] for m in tables for col in big.columns
           if col[0] == METRIC_LABELS[m]}
    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    return (big.style
            .format(fmt, na_rep="-")
            .background_gradient(cmap="RdYlGn", axis=0, low=0.15, high=0.15)
            .highlight_max(axis=0, props="font-weight: bold; border: 2px solid black;")
            .set_caption(f"{data}{fid} (colour = rank within each column, bold = best)")
            .set_table_styles([{"selector": "th", "props": "text-align: center;"},
                               {"selector": "caption", "props": "font-size: 1.1em;"}]))


# ---------------------------------------------------------------------------
# Result files of one row: reconstruction, re-simulated measurement, one image
# ---------------------------------------------------------------------------

def load_rec(row, root=ROOT):
    """Full result dict (.pth) of one row of the table (same tag as run_experiment).
    Rows of load_merged() carry their own results folder (column 'root'), used instead."""
    row_root = getattr(row, "root", None)
    if isinstance(row_root, str) and row_root:
        root = row_root
    tag = (f"Fidelity{row.fidelity_raw}_{row.data}_flux{row.flux}_{row.prior}"
           f"_loss{row.loss_raw}_{row.algo}")
    path = os.path.join(root, str(row.run_id), f"best_rec_{tag}.pth")
    return torch.load(path, map_location="cpu", weights_only=False)


def _select(df, data, flux, fidelity, methods):
    g = df[(df.data == data) & (df.flux == flux)]
    if fidelity is not None:
        g = g[g.fidelity == fidelity.upper()]
    if methods is not None:
        methods = [methods] if isinstance(methods, str) else methods
        g = g[g.method.apply(lambda m: any(s in m for s in methods))]
    if g.empty:
        raise ValueError(f"no results for data={data}, flux={flux}, fidelity={fidelity}")
    return g.sort_values("psnr", ascending=False)


_SUM_Y_CACHE = {}


def sum_measurements(res):
    """
    Sum of the 25 detector images of the measurement a result was computed from.
    The measurement is not saved in the .pth: it is simulated again with the same
    seed (same noise realization), and cached. Returns (sum_y, metrics vs ground truth).
    """
    from opt_functions.data_preparation import load_data, make_measurements
    from opt_functions.Solver_functions.pnp_utils import Config, compute_final_metrics
    c = res.get("config") or {}
    key = (res["name"], res["flux"], c.get("seed", 0))
    if key not in _SUM_Y_CACHE:
        Nz = c.get("Nz", 1)
        # default device, as in the evaluation: the noise depends on the device RNG
        cfg = Config(Nx=c.get("Nx", 256), Nz=Nz, pxsizex=c.get("pxsizex", 40),
                     is_3d=Nz > 1, seed=key[2])
        x = load_data(res["name"], cfg)
        sum_y = make_measurements(x, res["flux"], cfg)["sum_y"].cpu()
        _SUM_Y_CACHE[key] = sum_y, compute_final_metrics(x[:, REG_PLANE:REG_PLANE + 1] if x.shape[1] > 1 else x, sum_y)
    return _SUM_Y_CACHE[key]


def get_image(df, data, method=None, flux=None, fidelity=None, kind="rec", root=ROOT,
              plane=REG_PLANE, crop=None, normalize=False, scale_by_flux=True, verbose=True,
              return_metrics=False, lam=None, sigma=None):
    """
    One 2D image as a torch tensor (H, W), ready for gr.ShowImg.
    return_metrics=True: (image, {"psnr", "ssim", "microssim"}) instead, metrics vs the
    ground truth on the full image (as in the tables; not affected by crop / plane).

    data          : sample name, e.g. "tub_level"
    method        : full method name or an unambiguous substring ("WCRR_ISM [PROJ_GD] KL");
                    not needed for kind="gt" / "sum_y" (any method of that sample is used)
    flux          : needed if the sample has several fluxes
    kind          : "rec"   reconstruction of the method (best PSNR if several runs)
                    "gt"    ground truth
                    "sum_y" sum of the 25 detector images (re-simulated, same seed)
    plane         : 3D only: axial plane (None = max projection)
    crop          : (y0, y1, x0, x1) in pixels
    normalize     : divide by the max (as for the metrics)
    scale_by_flux : multiply by the flux, as show_method (rec, gt and sum_y are all in
                    intensity units, y = counts / flux): sum_y then = photon counts
    lam, sigma    : the run with this lambda / sigma (1% tolerance, so 0.1778 finds
                    0.17782794) instead of the best PSNR. Only values that were actually
                    run are on disk: the search's best one, or single_experiment.py runs.
                    Pass the full table (load(...)), not best_per_method(...), which keeps
                    one row per method.
    """
    g = _subset(df, data, fidelity)
    if method is not None:
        g = g[g.method == _match_method(g, method)]
    elif kind == "rec":
        raise ValueError("kind='rec' needs a method; one of:\n  " + "\n  ".join(g.method.unique()))
    if flux is not None:
        g = g[g.flux == flux]
    elif g.flux.nunique() > 1:
        raise ValueError(f"several fluxes for {data}: choose flux= one of {sorted(g.flux.unique())}")
    if g.empty:
        raise ValueError(f"no results for data={data}, method={method}, flux={flux}")
    for col, val in (("lambda", lam), ("sigma", sigma)):
        if val is None:
            continue
        hit = g[np.isclose(g[col].astype(float), float(val), rtol=1e-2, atol=1e-12)]
        if hit.empty:
            have = sorted(g[col].dropna().astype(float).unique())
            raise ValueError(f"no run with {col}={val:g} for {data}, {method}, flux {flux}; "
                             f"available: {[float(f'{v:.4g}') for v in have]}")
        g = hit
    r = g.loc[g.psnr.idxmax()]
    res = load_rec(r, root)

    if kind == "rec":
        x, label = res["x_rec"], r.method
        m = {k: float(res[k]) for k in ("psnr", "ssim", "microssim")}
    elif kind == "gt":
        x, label = res["x_gt"], "ground truth"
        m = {"psnr": float("inf"), "ssim": 1.0, "microssim": 1.0}
    elif kind == "sum_y":
        x, m = sum_measurements(res)
        m, label = dict(m), "sum of detectors"
    else:
        raise ValueError(f"kind must be 'rec', 'gt' or 'sum_y', not {kind!r}")
    if kind != "gt":
        label += (f" | PSNR {m['psnr']:.2f} dB, SSIM {m['ssim']:.3f}, "
                  f"MicroSSIM {m['microssim']:.3f}")

    img = _crop(to_img(x, normalize, plane), crop)
    if scale_by_flux and not normalize:
        img = img * r.flux
    if verbose:
        params = f"  ({params_str(r)})" if kind == "rec" else ""
        print(f"{data} | flux {r.flux:g} | {label}{_plane_label(plane)}{params}")
    img = torch.as_tensor(img)
    return (img, m) if return_metrics else img

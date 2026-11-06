#%%
"""
Print and compare the grid-search results (DRUNet, DRUNet-ft, WCRR, TV-smooth, L1, ...).
Reads summary_all.csv, plus the best_rec_*.pth files for the images.

3D data: images show one axial plane (plane=0 by default, the regularized one);
plane=None shows the max projection over the planes.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ISM.analysis.Graph_lib as gr


ROOT = "Results/evaluation_stop"            # evaluate_methods.py; old grid search: "Results/results_gridsearch"

# Display names; unknown priors are shown with their own name
PRIOR_NAMES = {
    "Drunet": "DRUNet",
    "Drunet_finetune": "DRUNet-ft",
    "weakly_convex": "WCRR",
    "TV_smooth": "TV-smooth",
    "L1": "L1-norm",               # not "L1": that is also a training-loss name
    # names used by old (legacy) result files
    "drunet": "DRUNet-ft",
    "drunet_gauss": "DRUNet",
}
EXPLICIT = {"WCRR", "WCRR_ISM", "WCRR_softplus_ISM", "weakly_convex",
            "weakly_convex_ISM", "TV_smooth", "L1",
            "WCRR_allpass", "WCRR_ISM_allpass", "WCRR_softplus_ISM_allpass"}
PRIORS_WITHOUT_SIGMA = {"TV_smooth", "L1"}          # sigma is not used
PRIORS_USING_LOSS = {"Drunet_finetune", "drunet"}   # one checkpoint per training loss


# ---------------------------------------------------------------------------
# Loading and text tables
# ---------------------------------------------------------------------------

def run_search(root, run_id):
    """Search type of a run, from its config.json: 'adaptive' or 'grid'."""
    path = os.path.join(root, str(run_id), "config.json")
    if not os.path.exists(path):
        return "grid"
    with open(path) as f:
        return json.load(f).get("search", "grid")


def load(root=ROOT, run_ids=None, exclude_legacy=False, split_by_fidelity=True, search=None):
    """split_by_fidelity=False: compare KL and L2 runs in the same tables
    (the fidelity is then shown in the method name).
    search: 'adaptive' = only the runs of evaluate_methods.py (adaptive search),
    'grid' = only the fixed-grid runs, None = all."""
    df = pd.read_csv(os.path.join(root, "summary_all.csv"))
    for col in ("step_scale", "stepsize", "prior"):        # missing in old files
        if col not in df:
            df[col] = "" if col == "prior" else float("nan")
    if run_ids is not None:
        df = df[df.run_id.isin(run_ids)]
    if exclude_legacy:
        df = df[~df.run_id.astype(str).str.startswith("legacy_")]
    if search is not None:
        kind = {r: run_search(root, r) for r in df.run_id.unique()}
        df = df[df.run_id.map(kind) == search]
        if df.empty:
            raise ValueError(f"no finished {search}-search results in {root}")
    # raw values are needed to rebuild the result file names
    df["fidelity_raw"] = df.fidelity.astype(str)
    df["loss_raw"] = df.loss.astype(str)
    df["fidelity"] = df.fidelity.astype(str).str.upper()
    df["loss"] = df.loss.fillna("none").astype(str)
    df["method"] = df.apply(method_name, axis=1)
    if not split_by_fidelity:
        df["method"] = df.method + " " + df.fidelity
        df["fidelity"] = "all"
    df["params"] = df.apply(params_str, axis=1)
    return df


def method_name(r):
    name = PRIOR_NAMES.get(r.prior, r.prior)
    if r.prior in PRIORS_USING_LOSS:
        name += f" ({r.loss})"
    return f"{name} [{r.algo}]"


def _fmt(v, f):
    return "-" if pd.isna(v) else format(v, f)


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


def print_ranking(df):
    """One table per (data, fidelity, flux): methods ranked by PSNR."""
    cols = ["method", "psnr", "ssim", "microssim", "params", "run_id"]
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

METRIC_LABELS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "microssim": "MicroSSIM"}
METRIC_FMT = {"psnr": "{:.2f}", "ssim": "{:.3f}", "microssim": "{:.3f}"}


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


def _method_colors(methods):
    """One colour family per prior; tab10 for the others."""
    fams = {"DRUNet-ft": plt.cm.Blues, "DRUNet": plt.cm.Greys, "WCRR": plt.cm.Reds,
            "TV-smooth": plt.cm.Greens, "L1-norm": plt.cm.Purples}
    groups, colors = {}, {}
    for m in methods:
        fam = next((f for f in fams if m.startswith(f)), None)   # "DRUNet-ft" before "DRUNet"
        groups.setdefault(fam, []).append(m)
    other = iter(plt.cm.tab10.colors)
    for fam, ms in groups.items():
        for k, m in enumerate(ms):
            colors[m] = (fams[fam](0.45 + 0.5 * (k + 1) / (len(ms) + 0.5)) if fam
                         else next(other))
    return colors


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
# Reconstructions and Fourier spectra
# ---------------------------------------------------------------------------

def load_rec(row, root=ROOT):
    """Full result dict (.pth) of one row of the table (same tag as run_experiment)."""
    tag = (f"Fidelity{row.fidelity_raw}_{row.data}_flux{row.flux}_{row.prior}"
           f"_loss{row.loss_raw}_{row.algo}")
    path = os.path.join(root, str(row.run_id), f"best_rec_{tag}.pth")
    return torch.load(path, map_location="cpu", weights_only=False)


def to_img(x, normalize=False, plane=0):
    """2D image from a (B, C, H, W) tensor.
    plane: channel to show for 3D data (0 = regularized plane); None = max projection.
    normalize=True divides by the max, as done for the metrics."""
    img = torch.as_tensor(x).detach().float().cpu()
    while img.ndim > 3:                       # drop the batch dimension
        img = img[0]
    if img.ndim == 3:                         # (C, H, W)
        img = img.amax(dim=0) if plane is None else img[min(plane, img.shape[0] - 1)]
    if normalize:
        img = img / img.max().clamp_min(1e-12)
    return img.numpy()


def _plane_label(plane):
    return "" if plane is None else f" | plane {plane}"


def edge_window(shape, alpha=0.2):
    """2D Tukey window: tapers the borders before the FFT, which removes the
    cross-shaped artefacts caused by the non-periodic image edges."""
    def tukey(n):
        w = np.ones(n)
        m = int(alpha * n / 2)
        if m > 0:
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
            w[:m], w[-m:] = ramp, ramp[::-1]
        return w
    return np.outer(tukey(shape[0]), tukey(shape[1]))


def _fft(img, window=True):
    if window:
        img = (img - img.mean()) * edge_window(img.shape) + img.mean()
    return np.fft.fftshift(np.fft.fft2(img))


def fft_img(img, window=True):
    """log(1 + |FFT|), zero frequency in the centre."""
    return np.log1p(np.abs(_fft(img, window)))


def radial_spectrum(img, window=True):
    """Radially averaged power spectrum: (frequency in cycles/pixel, power)."""
    P = np.abs(_fft(img, window)) ** 2
    ny, nx = img.shape
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - ny // 2, xx - nx // 2).astype(int)
    power = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    n = min(ny, nx) // 2
    return np.arange(n) / min(ny, nx), power[:n]


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


def _crop(img, crop):
    if crop is None:
        return img
    y0, y1, x0, x1 = crop
    return img[y0:y1, x0:x1]


def plot_reconstructions(df, data, flux, fidelity=None, methods=None, root=ROOT,
                         plane=0, pxsize_um=0.04, crop=None, spectrum=True, window=True,
                         normalize=False, shared_scale=True, vmax=None,
                         cmap="hot", save=None):
    """|
    Ground truth + best reconstruction of each method (row 1), their FFTs
    (row 2) and, optionally, the radially averaged power spectra.

    df           : output of best_per_method(load(...))
    methods      : substring or list of substrings to keep, e.g. ["WCRR", "TV-smooth"]; None = all
    plane        : 3D only: axial plane to show (None = max projection)
    pxsize_um    : pixel size in um, for the axes (FFT axes in cycles/um)
    crop         : (y0, y1, x0, x1) in pixels, to zoom (the FFT uses the full image)
    window       : taper the image borders before the FFT
    normalize    : True = each image divided by its max (as for PSNR/SSIM)
    shared_scale : True = same colour scale for all panels; False = one per panel
    vmax         : upper limit of the shared image scale (default: max over panels)
    save         : file name to save the figure (png/pdf)
    """
    rows = list(_select(df, data, flux, fidelity, methods).itertuples())
    results = [load_rec(r, root) for r in rows]

    full = [to_img(results[0]["x_gt"], normalize, plane)] + \
           [to_img(res["x_rec"], normalize, plane) for res in results]
    titles = ["Ground truth"] + [
        f"{r.method}\nPSNR {r.psnr:.2f}  SSIM {r.ssim:.3f}\nMicroSSIM {_fmt(r.microssim, '.3f')}"
        for r in rows]
    ffts = [fft_img(im, window) for im in full]

    ny, nx = full[0].shape
    shown = [_crop(im, crop) for im in full]
    if crop is not None:
        y0, y1, x0, x1 = crop
        ext = [x0 * pxsize_um, x1 * pxsize_um, y1 * pxsize_um, y0 * pxsize_um]
    else:
        ext = [0, nx * pxsize_um, ny * pxsize_um, 0]
    f_max = 1 / (2 * pxsize_um)
    f_ext = [-f_max, f_max, -f_max, f_max]

    img_vmax = vmax if vmax is not None else max(float(im.max()) for im in shown)
    f_lo, f_hi = np.percentile(ffts[0], [1, 99.9])    # FFT scale from the ground truth

    n = len(full)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 7.8), squeeze=False, layout="constrained")
    for j, (im, F, title) in enumerate(zip(shown, ffts, titles)):
        if shared_scale:
            h_img = axes[0, j].imshow(im, cmap=cmap, vmin=0, vmax=img_vmax, extent=ext)
            h_fft = axes[1, j].imshow(F, cmap="viridis", vmin=f_lo, vmax=f_hi, extent=f_ext)
        else:
            h_img = axes[0, j].imshow(im, cmap=cmap, vmin=0, vmax=float(im.max()), extent=ext)
            h_fft = axes[1, j].imshow(F, cmap="viridis", extent=f_ext,
                                      vmin=np.percentile(F, 1), vmax=np.percentile(F, 99.9))
            fig.colorbar(h_img, ax=axes[0, j], shrink=0.75, pad=0.01)
            fig.colorbar(h_fft, ax=axes[1, j], shrink=0.75, pad=0.01)
        axes[0, j].set_title(title, fontsize=9)
        axes[1, j].set_title("log(1 + |FFT|)", fontsize=9)
        axes[0, j].set_xlabel("um", fontsize=8)
        axes[1, j].set_xlabel("cycles/um", fontsize=8)
        for a in axes[:, j]:
            a.tick_params(labelsize=7)

    if shared_scale:      # one colorbar per row
        label = "intensity (normalized)" if normalize else "intensity"
        fig.colorbar(h_img, ax=axes[0, :].tolist(), shrink=0.8, pad=0.01).set_label(label, fontsize=9)
        fig.colorbar(h_fft, ax=axes[1, :].tolist(), shrink=0.8, pad=0.01).set_label(
            "log(1 + |FFT|)", fontsize=9)

    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    fig.suptitle(f"{data} | flux {flux}{fid}{_plane_label(plane)}", fontsize=12)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")

    if spectrum:
        fig2, ax = plt.subplots(figsize=(6.5, 4.5))
        for im, lab in zip(full, ["Ground truth"] + [r.method for r in rows]):
            f, P = radial_spectrum(im, window)
            gt = lab == "Ground truth"
            ax.semilogy(f[1:] / pxsize_um, P[1:], label=lab, lw=2.5 if gt else 1.3,
                        color="k" if gt else None)
        ax.set_xlabel("spatial frequency (cycles/um)")
        ax.set_ylabel("radially averaged power")
        ax.set_title(f"Power spectrum | {data} | flux {flux}{fid}{_plane_label(plane)}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig2.tight_layout()
        if save:
            root_, ext_ = os.path.splitext(save)
            fig2.savefig(f"{root_}_spectrum{ext_ or '.png'}", dpi=200, bbox_inches="tight")
    plt.show()


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
        _SUM_Y_CACHE[key] = sum_y, compute_final_metrics(x[:, :1], sum_y)
    return _SUM_Y_CACHE[key]


def plot_all_images(df, flux, fidelity=None, methods=None, root=ROOT, plane=0, crop=None,
                    normalize=True, show_sum=True, cmap="hot", save=None):
    """
    Every image tested at one flux: one row per image, ground truth (+ sum of the
    25 measurements) + best reconstruction of each method; columns ordered by mean
    PSNR over the images.

    df        : output of best_per_method(load(...))
    methods   : substring or list of substrings to keep; None = all
    normalize : True = each image divided by its max (as for PSNR/SSIM)
    show_sum  : add the sum of the 25 detector images (simulated again, same seed);
                always divided by its max, with its own scale if normalize=False
    Colour scale shared within each row.
    """
    g = df[df.flux == flux]
    if fidelity is not None:
        g = g[g.fidelity == fidelity.upper()]
    if methods is not None:
        methods = [methods] if isinstance(methods, str) else methods
        g = g[g.method.apply(lambda m: any(s in m for s in methods))]
    if g.empty:
        raise ValueError(f"no results for flux={flux}, fidelity={fidelity}")
    order = list(g.groupby("method").psnr.mean().sort_values(ascending=False).index)
    images = sorted(g.data.unique())

    sum_label = "Sum of 25 detectors"
    n_cols = len(order) + 1 + show_sum
    fig, axes = plt.subplots(len(images), n_cols, squeeze=False, layout="constrained",
                             figsize=(2.6 * n_cols, 2.6 * len(images) + 0.6))
    for i, data in enumerate(images):
        rows = {r.method: r for r in g[g.data == data].itertuples()}
        recs = {m: load_rec(r, root) for m, r in rows.items()}
        first = next(iter(recs.values()))
        panels = [("Ground truth", to_img(first["x_gt"], normalize, plane))]
        metrics = {}
        if show_sum:
            sum_y, metrics[sum_label] = sum_measurements(first)
            panels.append((sum_label, to_img(sum_y, True, plane)))
        panels += [(m, to_img(recs[m]["x_rec"], normalize, plane)
                    if m in recs and recs[m]["x_rec"] is not None else None) for m in order]
        metrics.update({m: {"psnr": r.psnr, "ssim": r.ssim} for m, r in rows.items()})
        own_scale = {sum_label} if not normalize else set()
        vmax = max(float(im.max()) for m, im in panels if im is not None and m not in own_scale)
        for j, (m, im) in enumerate(panels):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if im is None:
                ax.text(0.5, 0.5, "-", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.imshow(_crop(im, crop), cmap=cmap, vmin=0,
                          vmax=float(im.max()) if m in own_scale else vmax)
            title = m if i == 0 else ""
            if m in metrics:
                title += f"\nPSNR {metrics[m]['psnr']:.2f}  SSIM {metrics[m]['ssim']:.3f}"
            ax.set_title(title.strip("\n"), fontsize=8)
        axes[i, 0].set_ylabel(data, fontsize=10)

    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    fig.suptitle(f"All images | flux {flux}{fid}{_plane_label(plane)}", fontsize=12)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# One method: parameters, metrics and reconstructions
# ---------------------------------------------------------------------------

def _match_method(df, method):
    """Exact method name, or a substring that matches exactly one method."""
    names = list(df.method.unique())
    if method in names:
        return method
    hits = [m for m in names if method in m]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"'{method}' matches {len(hits)} methods; choose one of:\n  "
                     + "\n  ".join(hits or names))


def show_method(df, data, method, flux=None, fidelity=None, root=ROOT,
                plane=0, pxsize_um=0.04, crop=None, normalize=False,
                scale_by_flux=True, save=None):
    """
    For one method: print its best parameters and metrics for each flux, and
    show each reconstruction in its own figure.

    method        : full method name or an unambiguous substring, e.g. "WCRR" or "TV-smooth"
    flux          : one flux, a list of fluxes, or None = all fluxes
    plane         : 3D only: axial plane to show (None = max projection)
    crop          : (y0, y1, x0, x1) in pixels, to zoom on a region
    normalize     : True = each image divided by its max (as for the metrics)
    scale_by_flux : multiply the image by the flux (intensity in photons)
    save          : file name; one file per flux is written, e.g. rec_flux5.png
    """
    g = _subset(df, data, fidelity)
    name = _match_method(g, method)
    g = g[g.method == name]
    if flux is not None:
        g = g[g.flux.isin(np.atleast_1d(flux))]
    if g.empty:
        raise ValueError(f"no results for {name}, data={data}, flux={flux}")
    g = g.loc[g.groupby("flux").psnr.idxmax()].sort_values("flux")   # best per flux

    # ---- parameters and metrics ----
    priors = set(g.prior)
    cols = ["flux", "psnr", "ssim", "microssim"]
    if not priors <= PRIORS_WITHOUT_SIGMA:
        cols.append("sigma")
    if priors & EXPLICIT:
        cols.append("lambda")
    if not (g.step_scale.fillna(1) == 1).all():
        cols.append("step_scale")
    cols += ["iter", "stepsize", "run_id"]
    cols = [c for c in cols if c in g and g[c].notna().any()]
    fmts = {"psnr": ".2f", "ssim": ".4f", "microssim": ".4f", "sigma": ".3g",
            "lambda": ".3g", "step_scale": ".3g", "iter": ".0f", "stepsize": ".2e"}
    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    print(f"\n=== {name} | {data}{fid} ===")
    print(g[cols].rename(columns={"stepsize": "tau"}).to_string(
        index=False,
        formatters={("tau" if c == "stepsize" else c): (lambda v, f=f: _fmt(v, f))
                    for c, f in fmts.items() if c in cols}))

    # ---- images: one figure per flux ----
    for r in g.itertuples():
        rec = _crop(to_img(load_rec(r, root)["x_rec"], normalize, plane), crop)
        if scale_by_flux:
            rec = rec * r.flux
        gr.ShowImg(rec, pxsize_um)
        plt.gcf().suptitle(f"{name} | flux {r.flux:g}{_plane_label(plane)}", fontsize=10)
        if save:
            root_, ext_ = os.path.splitext(save)
            plt.gcf().savefig(f"{root_}_flux{r.flux:g}{ext_ or '.png'}", dpi=200,
                              bbox_inches="tight")
        plt.show()
    return g


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



#%%
if __name__ == "__main__":
    # split_by_fidelity=False puts KL and L2 runs in the same tables
    # search="adaptive": only the current evaluate_methods.py runs ("grid": the old ones, None: all)
    df = load(ROOT, split_by_fidelity=False, search="adaptive")
    best = best_per_method(df)
    DATA = "circle_phantom"                            # or "tub_level" for the 2D results
    PLANE = 0                                  # 3D: plane shown in the images (None = max proj.)

#%%
    # Heatmap tables: PSNR / SSIM / MicroSSIM, methods x flux
    plot_metric_heatmaps(best, DATA)

#%%
    # Metrics as a function of the flux; relative_to shows the gain over a baseline
    plot_metric_vs_flux(best, DATA)            # e.g. relative_to="WCRR ["

#%%
    # Colour-coded table (VS Code interactive window / Jupyter)
    from IPython.display import display
    display(styled_table(best, DATA))

#%%
    print_params(best)

#%%
    # Reconstructions + FFT for one flux (methods=None: all methods)
    plot_reconstructions(best, DATA, 40,
                         methods=None,         # e.g. ["WCRR", "TV-smooth", "L1-norm"]
                         plane=PLANE,
                         crop=None,            # e.g. (64, 192, 64, 192) to zoom
                         normalize=False,      # True: each image / its max (as the metrics)
                         shared_scale=True)    # False: one colorbar per panel

#%%
    # All images at one flux: rows = images, columns = ground truth + methods
    plot_all_images(best, 20, methods=["WCRR_ISM", "TV-smooth"])                  # e.g. methods=["WCRR", "TV-smooth"]

#%%
    # One method: best parameters and reconstructions
    show_method(best, DATA, "WCRR_ISM [PROJ_GD] KL", flux=5, plane=PLANE)
    # show_method(best, DATA, "TV-smooth [PROJ_GD] KL")              # all fluxes
    # show_method(best, DATA, "L1-norm [PROX_GD] L2", flux=[5, 20])

#%%
    # Convergence of the functional F(x_k) = f + lam R (explicit priors only)
    # plot_functional(best, DATA, 40, kind="gap")            # e.g. methods=["WCRR", "TV-smooth"], rates=(1, 2)
    plot_functional(best, DATA, 40, kind="gap", methods = ['WCRR'])   # F(x_k) itself
#%%
    # Bounded iterates, finite length, criticality, step sizes (needs refresh_histories.py)
    convergence_check(best, DATA, 20, methods=["WCRR", "TV-smooth"])          # e.g. methods=["WCRR", "TV-smooth"]
# %%

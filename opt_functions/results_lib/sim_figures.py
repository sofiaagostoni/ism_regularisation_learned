"""
Figures of the simulated results: reconstructions and power spectra, intensity profiles,
the paper comparison figure, all images at one flux, one method, single images (3D planes).
"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ISM.analysis.Graph_lib as gr

from opt_functions.Solver_functions.prior_config import REG_PLANE
from .common import (PRIORS_WITHOUT_SIGMA, EXPLICIT, _fmt, _plane_label, to_img, _crop,
                     _match_method)
from .sim import ROOT, _subset, _select, load_rec, sum_measurements


# ---------------------------------------------------------------------------
# Reconstructions and Fourier spectra
# ---------------------------------------------------------------------------


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


def radial_spectrum(img, window=True):
    """Radially averaged power spectrum: (frequency in cycles/pixel, power)."""
    P = np.abs(_fft(img, window)) ** 2
    ny, nx = img.shape
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - ny // 2, xx - nx // 2).astype(int)
    power = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    n = min(ny, nx) // 2
    return np.arange(n) / min(ny, nx), power[:n]


def plot_reconstructions(df, data, flux, fidelity=None, methods=None, root=ROOT,
                         plane=REG_PLANE, pxsize_um=0.04, crop=None, spectrum=True, window=True,
                         normalize=False, shared_scale=True, vmax=None,
                         cmap="hot", save=None):
    """
    Ground truth + best reconstruction of each method (one row) and, optionally, the
    radially averaged power spectra.

    df           : output of best_per_method(load(...))
    methods      : substring or list of substrings to keep, e.g. ["WCRR", "TV-smooth"]; None = all
    plane        : 3D only: axial plane to show (None = max projection)
    pxsize_um    : pixel size in um, for the axes (spectrum in cycles/um)
    crop         : (y0, y1, x0, x1) in pixels, to zoom (the spectrum uses the full image)
    window       : taper the image borders before the FFT of the spectrum
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

    ny, nx = full[0].shape
    shown = [_crop(im, crop) for im in full]
    if crop is not None:
        y0, y1, x0, x1 = crop
        ext = [x0 * pxsize_um, x1 * pxsize_um, y1 * pxsize_um, y0 * pxsize_um]
    else:
        ext = [0, nx * pxsize_um, ny * pxsize_um, 0]

    img_vmax = vmax if vmax is not None else max(float(im.max()) for im in shown)

    n = len(full)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.2), squeeze=False, layout="constrained")
    for j, (im, title) in enumerate(zip(shown, titles)):
        if shared_scale:
            h_img = axes[0, j].imshow(im, cmap=cmap, vmin=0, vmax=img_vmax, extent=ext)
        else:
            h_img = axes[0, j].imshow(im, cmap=cmap, vmin=0, vmax=float(im.max()), extent=ext)
            fig.colorbar(h_img, ax=axes[0, j], shrink=0.75, pad=0.01)
        axes[0, j].set_title(title, fontsize=9)
        axes[0, j].set_xlabel("um", fontsize=8)
        axes[0, j].tick_params(labelsize=7)

    if shared_scale:      # one colorbar for the row
        label = "intensity (normalized)" if normalize else "intensity"
        fig.colorbar(h_img, ax=axes[0, :].tolist(), shrink=0.8, pad=0.01).set_label(label, fontsize=9)

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


def _short_labels(names):
    """Method names without the words they all share, + the shared part (for the title).
    ["TV-smooth [PROJ_GD] KL", "TV-smooth [PROJ_GD] L2"] -> (["KL", "L2"], "TV-smooth [PROJ_GD]")"""
    words = [n.split() for n in names]
    common = [w for w in words[0] if all(w in ws for ws in words[1:])] if len(names) > 1 else []
    short = [" ".join(w for w in ws if w not in common) or n for ws, n in zip(words, names)]
    return short, " ".join(common)


def plot_profile(df, data, flux, methods=None, fidelity=None, row=None, col=None,
                 root=ROOT, plane=REG_PLANE, normalize=True, pxsize_um=0.04, crop=None,
                 cmap="hot", save=None):
    """
    Intensity profile along one line of the image, ground truth vs methods: the ground
    truth with the line drawn on it (left) and the profiles (right; PSNR / SSIM of each
    method in the legend).

    df        : output of best_per_method(load(...)) or of pick(...)
    methods   : substring or list of substrings (as plot_reconstructions), e.g.
                ["TV-smooth [PROJ_GD] KL", "TV-smooth [PROJ_GD] L2"] or "TV-smooth [PROJ_GD]"
                with a table that has both fidelities; None = all methods of (data, flux)
    row / col : pixel index of a horizontal (row) or vertical (col) line; neither: the row
                with the most ground-truth signal (largest row sum, inside crop)
    Methods are shown in the order of `methods` (a list), else by PSNR.
    normalize : each image divided by its max, as for PSNR / SSIM (False: intensity units)
    crop      : (y0, y1, x0, x1) in pixels: zoom (images and profile); row / col stay
                full-image indices
    """
    g = _select(df, data, flux, fidelity, methods)
    if isinstance(methods, (list, tuple)):        # keep the order asked for
        first = lambda m: next(k for k, s in enumerate(methods) if s in m)
        g = g.assign(_o=g.method.map(first)).sort_values(["_o", "psnr"], ascending=[True, False])
    rows = list(g.itertuples())
    results = [load_rec(r, root) for r in rows]
    is_3d = torch.as_tensor(results[0]["x_gt"]).ndim == 4 and results[0]["x_gt"].shape[1] > 1
    gt = to_img(results[0]["x_gt"], normalize, plane)
    recs = [to_img(res["x_rec"], normalize, plane) for res in results]
    ny, nx = gt.shape
    y0, y1, x0, x1 = crop if crop is not None else (0, ny, 0, nx)
    if row is None and col is None:
        row = y0 + int(np.argmax(gt[y0:y1, x0:x1].sum(axis=1)))
    if row is not None and not y0 <= row < y1 or col is not None and not x0 <= col < x1:
        raise ValueError(f"the line (row={row}, col={col}) is outside crop={crop}")

    def line(img):                             # profile inside the crop
        return img[row, x0:x1] if row is not None else img[y0:y1, col]

    pos = (np.arange(x0, x1) if row is not None else np.arange(y0, y1)) * pxsize_um
    ext = [x0 * pxsize_um, x1 * pxsize_um, y1 * pxsize_um, y0 * pxsize_um]
    labels, common = _short_labels([r.method for r in rows])
    colors = plt.cm.tab10.colors              # distinct lines (KL / L2 of one prior too)

    fig, (ax_gt, ax) = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained",
                                    gridspec_kw={"width_ratios": [1, 2]})
    h_img = ax_gt.imshow(_crop(gt, crop), cmap=cmap, vmin=0, vmax=float(_crop(gt, crop).max()),
                         extent=ext)
    if row is not None:
        ax_gt.axhline(row * pxsize_um, color="c", lw=1.2)
    else:
        ax_gt.axvline(col * pxsize_um, color="c", lw=1.2)
    ax_gt.set_title("Ground truth", fontsize=9)
    ax_gt.set_xlabel("um", fontsize=8)
    fig.colorbar(h_img, ax=ax_gt, shrink=0.8, pad=0.01).set_label(
        "intensity (normalized)" if normalize else "intensity", fontsize=9)

    ax.plot(pos, line(gt), color="k", lw=2.2, label="GT")
    for k, (im, lab, r) in enumerate(zip(recs, labels, rows)):
        ax.plot(pos, line(im), lw=1.3, color=colors[k % 10],
                label=f"{lab}  (PSNR {r.psnr:.2f}, SSIM {r.ssim:.3f})")
    where = f"row {row}" if row is not None else f"column {col}"
    ax.set_title(f"profile, {where} (cyan line)", fontsize=9)
    ax.set_xlabel("um")
    ax.set_ylabel("intensity (normalized)" if normalize else "intensity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    for a in fig.axes:
        a.tick_params(labelsize=7)

    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    vs = " vs ".join(labels) if common else ""
    fig.suptitle(" | ".join(s for s in (common, data, f"flux {flux:g}", vs) if s) + fid
                 + (_plane_label(plane) if is_3d else ""), fontsize=12)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()
    return {lab: line(im) for lab, im in zip(["GT"] + labels, [gt] + recs)}


def _brightest_window(img, size):
    """(y0, y1, x0, x1) of the size x size window with the largest sum of img."""
    ny, nx = img.shape
    size = min(size, ny, nx)
    c = np.pad(img.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]
    y0, x0 = np.unravel_index(np.argmax(s), s.shape)
    return int(y0), int(y0 + size), int(x0), int(x0 + size)


def plot_comparison_figure(df, data, flux, methods=None, fidelity=None, root=ROOT,
                           plane=REG_PLANE, row=None, col=None, zoom=None, zoom_size=None,
                           show_input=True, labels=None, normalize=True, pxsize_um=0.04,
                           profile_range=None, cmap="hot", title=None, save=None):
    """
    Paper-style comparison figure:
      row 1  input (sum of the 25 detectors) + ground truth + each method, full image,
             with the zoom box (lime) and the profile line (cyan); PSNR / SSIM in the titles
      row 2  the zoomed region of every panel
      row 3  intensity profile along the line, all curves (GT black, input light grey)

    df            : best_per_method(load(...)), pick(...) or load_merged(...)
    methods       : substring or list of substrings; a list keeps its order (e.g. weakest
                    to strongest), else ordered by increasing PSNR; None = all
    row / col     : pixel index of a horizontal / vertical profile line; neither = the row
                    with the most ground-truth signal inside the zoom box
    zoom          : (y0, y1, x0, x1) in pixels; None = the zoom_size square (default
                    image size / 3) with the most ground-truth signal
    show_input    : add the sum of the 25 detectors (re-simulated, same seed)
    labels        : {method name: display name}; default = method names without the
                    words they all share (those go in the title)
    normalize     : each image divided by its max, as for PSNR / SSIM
    profile_range : (start, stop) in pixels along the line; None = full line
    Returns {label: profile}.
    """
    g = _select(df, data, flux, fidelity, methods)
    if isinstance(methods, (list, tuple)):
        first = lambda m: next(k for k, s in enumerate(methods) if s in m)
        g = g.assign(_o=g.method.map(first)).sort_values(["_o", "psnr"])
    else:
        g = g.sort_values("psnr")
    rows = list(g.itertuples())
    results = [load_rec(r, root) for r in rows]
    is_3d = torch.as_tensor(results[0]["x_gt"]).ndim == 4 and results[0]["x_gt"].shape[1] > 1

    short, common = _short_labels([r.method for r in rows])
    if labels:
        short = [labels.get(r.method, s) for r, s in zip(rows, short)]
    gt = to_img(results[0]["x_gt"], normalize, plane)
    panels = []                                # (label, image, subtitle, own colour scale)
    if show_input:
        sum_y, m_in = sum_measurements(results[0])
        panels.append(("input (sum of detectors)", to_img(sum_y, True, plane),
                       f"{m_in['psnr']:.2f} dB, SSIM {m_in['ssim']:.3f}", True))
    panels.append(("ground truth", gt, "reference", False))
    panels += [(lab, to_img(res["x_rec"], normalize, plane),
                f"{r.psnr:.2f} dB, SSIM {r.ssim:.3f}", False)
               for lab, res, r in zip(short, results, rows)]

    ny, nx = gt.shape
    if zoom is None:
        zoom = _brightest_window(gt, zoom_size or min(ny, nx) // 3)
    zy0, zy1, zx0, zx1 = zoom
    if row is None and col is None:            # inside the middle half of the zoom box
        q = (zy1 - zy0) // 4
        row = zy0 + q + int(np.argmax(gt[zy0 + q:zy1 - q, zx0:zx1].sum(axis=1)))
    n_line = nx if row is not None else ny
    a, b = profile_range if profile_range is not None else (0, n_line)

    def line(img):
        return img[row, a:b] if row is not None else img[a:b, col]

    vmax = max(float(im.max()) for _, im, _, own in panels if not own)
    n = len(panels)
    fig = plt.figure(figsize=(2.3 * n, 2.3 * 2 + 2.6), layout="constrained")
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 1.1])
    for j, (lab, im, sub, own) in enumerate(panels):
        vm = float(im.max()) if own else vmax
        ax_full, ax_zoom = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        ax_full.imshow(im, cmap=cmap, vmin=0, vmax=vm)
        ax_full.add_patch(Rectangle((zx0 - 0.5, zy0 - 0.5), zx1 - zx0, zy1 - zy0,
                                    fill=False, ec="lime", lw=1.2))   # visible on cmap "hot"
        if row is not None:
            ax_full.plot([a - 0.5, b - 0.5], [row, row], color="c", lw=0.7, alpha=0.8)
        else:
            ax_full.plot([col, col], [a - 0.5, b - 0.5], color="c", lw=0.7, alpha=0.8)
        ax_full.set_title(f"{lab}\n{sub}", fontsize=8)
        ax_zoom.imshow(im[zy0:zy1, zx0:zx1], cmap=cmap, vmin=0, vmax=vm)
        for ax in (ax_full, ax_zoom):
            ax.set_xticks([])
            ax.set_yticks([])

    ax_p = fig.add_subplot(gs[2, :])
    pos = np.arange(a, b) * pxsize_um
    colors = iter(plt.cm.tab10.colors)
    profiles = {}
    for lab, im, _, own in panels:
        p = line(im)
        profiles[lab] = p
        if lab == "ground truth":
            kw = dict(color="k", lw=1.8, zorder=3)
        elif own:                               # input
            kw = dict(color="0.75", lw=1.0, zorder=1)
        else:
            kw = dict(color=next(colors), lw=1.2, zorder=2)
        ax_p.plot(pos, p, label=lab, **kw)
    where = f"row {row}" if row is not None else f"column {col}"
    ax_p.set_title(f"profile along the cyan line ({where})", fontsize=9)
    ax_p.set_xlabel("x (um)" if row is not None else "y (um)")
    ax_p.set_ylabel("intensity (normalized)" if normalize else "intensity")
    ax_p.set_xlim(pos[0], pos[-1])
    ax_p.spines[["top", "right"]].set_visible(False)
    top = max(float(np.max(p)) for p in profiles.values())
    ax_p.set_ylim(top=1.22 * top)               # room for the legend above the curves
    ax_p.legend(fontsize=7, ncol=len(panels), frameon=False, loc="upper right")

    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    fig.suptitle(title or (" | ".join(s for s in (common, data, f"flux {flux:g}") if s) + fid
                           + (_plane_label(plane) if is_3d else "")), fontsize=11)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()
    return profiles


def plot_all_images(df, flux, fidelity=None, methods=None, root=ROOT, plane=REG_PLANE, crop=None,
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


def show_method(df, data, method, flux=None, fidelity=None, root=ROOT,
                plane=REG_PLANE, pxsize_um=0.04, crop=None, normalize=False,
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
    cols += ["iter", "stepsize", "search", "run_id"]
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
        gr.ShowImg(torch.as_tensor(rec), pxsize_um)     # ShowImg needs a tensor
        plt.gcf().suptitle(f"{name} | flux {r.flux:g}{_plane_label(plane)}", fontsize=10)
        if save:
            root_, ext_ = os.path.splitext(save)
            plt.gcf().savefig(f"{root_}_flux{r.flux:g}{ext_ or '.png'}", dpi=200,
                              bbox_inches="tight")
        plt.show()
    return g


# ---------------------------------------------------------------------------
# Single images (gr.ShowImg), every plane of 3D data
# ---------------------------------------------------------------------------

def plot_single_images(df, data, flux, methods=None, fidelity=None, root=ROOT,
                       planes=(REG_PLANE, 0), pxsize_um=0.04, crop=None, normalize=False,
                       scale_by_flux=True, shared_scale=True, show_input=True, cmap="hot",
                       save_dir=None):
    """
    One figure per image (gr.ShowImg: scale bar + colorbar, as show_method / get_image):
    the input (sum of the 25 detectors), the ground truth and the best reconstruction of
    each method, for every plane in `planes` (3D: 1 = in focus, 0 = out of focus,
    None = max projection; 2D data: one image each).

    Titles: method + PSNR / SSIM / MicroSSIM of the plane shown, computed here (the tables
    only have the in-focus plane); the input's metrics are vs the in-focus ground truth.
    methods      : substring or list of substrings to keep; None = all
    crop         : (y0, y1, x0, x1) in pixels
    scale_by_flux: intensity in photons (the input: photon counts of the detector sum)
    shared_scale : True = per plane, the same colour range for the ground truth and every
                   reconstruction (the input keeps its own range); False = one per image
    save_dir     : folder; one file per figure, <data>_flux<f>_plane<p>_<label>.png
    """
    import re
    from opt_functions.Solver_functions.pnp_utils import compute_final_metrics

    rows = list(_select(df, data, flux, fidelity, methods).itertuples())
    results = [load_rec(r, root) for r in rows]
    x_gt = torch.as_tensor(results[0]["x_gt"]).float().cpu()
    is_3d = x_gt.ndim == 4 and x_gt.shape[1] > 1
    if not is_3d:
        planes = (REG_PLANE,)                      # to_img ignores the plane for 2D data
    scale = flux if (scale_by_flux and not normalize) else 1.0
    clabel = "photons" if scale != 1.0 else ("intensity (normalized)" if normalize else "intensity")
    fid = f" | fidelity {fidelity.upper()}" if fidelity else ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def show(img, title, vmax, tag):
        img = torch.as_tensor(np.ascontiguousarray(_crop(img, crop)))
        fig, ax = gr.ShowImg(img, pxsize_um, clabel=clabel, cmap=cmap, vmin=torch.tensor(0.),
                             vmax=torch.as_tensor(float(vmax)))
        ax.set_title(title, fontsize=10)
        if save_dir:
            name = re.sub(r"[^\w.-]+", "_", f"{data}_flux{flux:g}_{tag}").strip("_")
            fig.savefig(os.path.join(save_dir, name + ".png"), dpi=200, bbox_inches="tight")
        plt.show()

    def metrics(x, p):
        """Metrics of the plane shown (2D: the stored ones are recomputed the same way)."""
        x = torch.as_tensor(x).float().cpu()
        if not is_3d:
            return compute_final_metrics(x_gt, x)
        if p is None:
            return None
        return compute_final_metrics(x_gt[:, p:p + 1], x[:, p:p + 1])

    def mline(m):
        return (f"\nPSNR {m['psnr']:.2f}  SSIM {m['ssim']:.3f}  "
                f"MicroSSIM {_fmt(m['microssim'], '.3f')}") if m else ""

    if show_input:                                 # does not depend on the plane
        sum_y, m_in = sum_measurements(results[0])
        img = to_img(sum_y, normalize, None) * scale
        show(img, f"input (sum of detectors)\n{data} | flux {flux:g}{fid}"
                  + (" | vs in-focus GT" if is_3d else "") + mline(m_in),
             _crop(img, crop).max(), "input")

    for p in planes:
        where = ""
        if is_3d:
            where = (" | max projection" if p is None else
                     f" | plane {p} ({'in focus' if p == REG_PLANE else 'out of focus'})")
        imgs = [("Ground truth", to_img(x_gt, normalize, p) * scale, None)]
        imgs += [(r.method, to_img(res["x_rec"], normalize, p) * scale, metrics(res["x_rec"], p))
                 for r, res in zip(rows, results)]
        vmax = max(float(_crop(im, crop).max()) for _, im, _ in imgs)
        for label, img, m in imgs:
            show(img, f"{label}\n{data} | flux {flux:g}{fid}{where}{mline(m)}",
                 vmax if shared_scale else _crop(img, crop).max(),
                 f"plane{'max' if p is None else p}_{label}" if is_3d else label)

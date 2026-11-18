"""
Real-data figures (run_real.py results, no ground truth): every run vs the raw data, one
run with a zoom, the raw data with several colormaps, the 25 detectors, and the publicity
figures (diagonal split, three bands).
"""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from opt_functions.Solver_functions.prior_config import REG_PLANE
from .real import (RAW_UNITS, real_raw, real_detectors, _real_px_nm, _select_real,
                   _pick_real, _real_label, _real_name, _real_units, _theme, _real_cmap,
                   _real_norm, _real_pnorm, _real_colorbar, _CB_UL, _CB_LL, _CB_LR,
                   _real_text, _real_scalebar, _INSET_CORNERS, _inset_rect, _real_inset,
                   _PROFILE_LINE, _line_profile, _real_profile_axes, _parse_aspect,
                   _aspect_window, _diag_dist, _frame_map, _real_canvas)


def plot_real(runs, data, methods=None, lambdas=None, plane=REG_PLANE, crop="any",
              iters=None, picks=None, zoom=None, trim=12, cmap="magma", pct=99.8, gamma=0.8,
              ncols=4, labels=None, panel_size=4.0, colorbar=True, inset=None,
              inset_loc="lower right", inset_size=0.4, profile=None, profile_width=1,
              profile_norm="max", theme=None, sigmas=None, save=None):
    """
    Raw data (sum of the 25 detectors) + every selected reconstruction of `data`, one panel
    each, publicity style. methods / lambdas: None = all runs of `data` (run_real.py).
    sigmas: only these sigmas (WCRR, WCRR_ISM, WCHS, DRUNet); the methods without sigma
    (TV_smooth, L1, RL) are kept. In picks: ("WCRR_ISM", (lambda, sigma)).
    theme: "white" / "black" background around the images (default REAL_THEME).
    plane: REG_PLANE (in focus) or 0 (out of focus). crop: the run_real.py --crop to use
    (None = full image, "any" = whatever is there, if unique). iters: only these iteration
    counts (e.g. the Richardson-Lucy ones: methods=["TV_smooth", "RL"], iters=[20, 5000]).
    zoom=(y0, y1, x0, x1) (after trim): box on every panel and a row of zooms below.
    trim: px removed from each border (wrap-around of the circular FFT model).
    picks: exactly these runs, in this order, instead of methods / lambdas / iters, e.g.
    picks=[("TV_smooth", 0.00708), ("RL", 20)] (lambda for regularized methods, iteration
    count for RL; see _pick_real).
    labels: {method: display name} or {(method, lambda): name}. save: path (.png / .pdf).
    colorbar: one per panel (and zoom), in the real units: photon counts for the raw data
    and the flux-1 runs, intensity / alpha for --flux auto / <alpha> runs.
    inset=(y0, y1, x0, x1) (after trim): the same crop shown in a corner of every panel
    (inset_loc: "lower right", "lower left" or "upper right"; inset_size: its longer side,
    as a fraction of the panel), box + connecting lines on the image.
    profile=(y0, x0, y1, x1) (after trim): end points of a line (e.g. a horizontal one at
    row r: (r, 0, r, W - 1)); drawn in cyan on every panel, profiles of the raw data and of
    every reconstruction in a row below. profile_width: px averaged across the line.
    profile_norm: "max" = each profile / its max on the line (the units differ), None = own
    units. With profile: returns (sel, {label: profile}).
    """
    if inset is not None and inset_loc not in _INSET_CORNERS:
        raise ValueError(f"inset_loc: one of {list(_INSET_CORNERS)}")
    sel = (_pick_real(runs, data, picks, crop) if picks is not None
           else _select_real(runs, data, methods, lambdas, crop, iters, sigmas))
    if sel.empty:
        print(f"no run_real.py result for {data}")
        return
    first = sel.iloc[0]
    raw = real_raw(data, first.crop)
    imgs = [("Raw · Σ 25 detectors", raw, RAW_UNITS)]
    dup = sel.duplicated(["method", "lambda", "sigma"], keep=False)
    for i, r in sel.iterrows():
        x = torch.load(r.path, map_location="cpu", weights_only=False)["x"]
        lab = _real_label(r, detail=bool(dup[i]))
        if labels:
            lab = labels.get((r.method, r["lambda"]), labels.get(r.method, lab))
        imgs.append((lab, x[0, plane].numpy(), _real_units(r)))
    if trim:
        imgs = [(l, im[trim:-trim, trim:-trim], u) for l, im, u in imgs]
    norms = [_real_pnorm(im, pct, gamma) for _, im, _ in imgs]   # same display as before
    cm, px_nm = _real_cmap(cmap), first.px_nm

    n = len(imgs)
    ncols = min(ncols, n)
    nrows_full = int(np.ceil(n / ncols))
    rows = nrows_full * (2 if zoom else 1)
    prof_h = 0.85 * panel_size if profile is not None else 0.0   # height of the profile row
    fig_h = panel_size * rows + prof_h
    th = _theme(theme)
    fig = plt.figure(figsize=(panel_size * ncols, fig_h), facecolor=th["bg"])
    top = 1 - 0.5 / fig_h                                         # room for the title
    if profile is None:
        gs = fig.add_gridspec(rows, ncols, wspace=0.02, hspace=0.02, left=0.005, right=0.995,
                              bottom=0.005, top=top)
    else:
        outer = fig.add_gridspec(2, 1, height_ratios=[panel_size * rows, prof_h - 0.6],
                                 hspace=0.6 / (panel_size * rows), left=0.005, right=0.995,
                                 bottom=0.55 / fig_h, top=top)
        gs = outer[0].subgridspec(rows, ncols, wspace=0.02, hspace=0.02)
        ax_prof = fig.add_subplot(outer[1].subgridspec(1, 3, width_ratios=[0.07, 0.9, 0.03])[1])
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(ncols)] for i in range(rows)])
    for ax in axes.flat:
        ax.set_facecolor(th["bg"])
        ax.axis("off")

    H, W = imgs[0][1].shape
    inset_rect = _inset_rect((H, W), inset, inset_loc, inset_size) if inset else None
    cb_rect = (0.925, 0.22, 0.022, 0.5)
    if inset_rect is not None and _INSET_CORNERS[inset_loc][0]:  # inset on the right: no overlap
        ix, iy, iw, ih = inset_rect
        if _INSET_CORNERS[inset_loc][1]:                         # upper right: colorbar below
            cb_rect = (0.925, 0.22, 0.022, max(0.1, min(0.5, iy - 0.05 - 0.22)))
        else:                                                    # lower right: colorbar above
            b = iy + ih + 0.05
            cb_rect = (0.925, b, 0.022, max(0.1, min(0.5, 0.88 - b)))
    scalebar_left = inset_loc == "lower right" and inset is not None

    def draw_line(a, dx=0, dy=0):
        y0, x0, y1, x1 = profile
        a.plot([x0 - dx, x1 - dx], [y0 - dy, y1 - dy], color=_PROFILE_LINE, lw=1.2,
               scalex=False, scaley=False)

    for k, ((lab, im, units), nrm) in enumerate(zip(imgs, norms)):
        r, c = divmod(k, ncols)
        ax = axes[r * (2 if zoom else 1), c]
        ax.imshow(im, cmap=cm, norm=nrm, interpolation="lanczos")
        _real_text(ax, 0.035, 0.965, lab)
        if colorbar:
            _real_colorbar(ax, nrm, cm, units, rect=cb_rect)
        if k == 0:
            _real_scalebar(ax, im.shape[1], px_nm, left=scalebar_left)
        if profile is not None:
            draw_line(ax)
        if inset is not None:
            axins = _real_inset(ax, im, inset, nrm, cm, inset_rect)
            if profile is not None:
                draw_line(axins)
            if k == 0:
                _real_scalebar(axins, inset[3] - inset[2], px_nm, size=8, y=0.08)
        if zoom:
            y0, y1, x0, x1 = zoom
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white",
                                   lw=1.2, ls="--"))
            az = axes[r * 2 + 1, c]
            az.imshow(im[y0:y1, x0:x1], cmap=cm, norm=nrm, interpolation="lanczos")
            _real_text(az, 0.035, 0.965, "zoom", 11, weight="normal")
            if colorbar:
                _real_colorbar(az, nrm, cm, units)
            if profile is not None:
                draw_line(az, x0, y0)
            if k == 0:
                _real_scalebar(az, x1 - x0, px_nm, size=11)
    profiles = None
    if profile is not None:
        profiles = _real_profile_axes(ax_prof, [(lab, im) for lab, im, _ in imgs], profile,
                                      profile_width, profile_norm, px_nm, theme)
    where = f"plane {plane} ({'in focus' if plane == REG_PLANE else 'out of focus'})"
    fig.suptitle(f"{data} · {where}" + (f" · crop {int(first.crop)}" if not pd.isna(first.crop)
                                        else ""), color=th["ink"], fontsize=14,
                 y=1 - 0.12 / fig_h)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight", facecolor=th["bg"])
    plt.show()
    return sel if profile is None else (sel, profiles)


def plot_real_single(runs, data, pick, zoom, plane=REG_PLANE, crop="any", layout="side",
                     trim=12, cmap="magma", pct=99.8, gamma=0.8, zoom_norm="full",
                     inset_loc="lower right", inset_size=0.4, label=None, colorbar=True,
                     panel_size=5.0, theme=None, save=None):
    """
    ONE real-data result (or the raw data) with a zoom on a crop, publicity style as plot_real.
    theme: "white" / "black" background around the images (default REAL_THEME).

    pick     : (method, value) as in plot_real(picks=...): lambda for the regularized methods,
               iteration count for "RL", PnP: iterations, (sigma, it) or (sigma, step, it);
               value None = any (see _pick_real). "raw" = the raw data (sum of the 25 det.)
    zoom     : (y0, y1, x0, x1) in pixels, after trim (as plot_real)
    layout   : "side"  full image with the box + the zoom next to it (same height), joined
               "inset" the zoom in a corner of the full image (inset_loc, inset_size)
               "zoom"  only the crop
    zoom_norm: "full" = the zoom keeps the colour scale of the full image (comparable);
               "zoom" = renormalised on the crop (dim structures visible)
    label    : text instead of the default "<method> λ=..."
    Returns (full image, crop) as arrays (after trim), for your own plots.
    """
    from matplotlib.patches import ConnectionPatch
    if layout not in ("side", "inset", "zoom"):
        raise ValueError("layout: 'side', 'inset' or 'zoom'")
    if layout == "inset" and inset_loc not in _INSET_CORNERS:
        raise ValueError(f"inset_loc: one of {list(_INSET_CORNERS)}")
    if isinstance(pick, str) and pick == "raw":
        sel = _select_real(runs, data, crop=crop)          # for the crop and the pixel size
        if sel.empty:
            raise ValueError(f"no run_real.py result for {data}")
        r = sel.iloc[0]
        img, lab, units = real_raw(data, r.crop), "Raw · Σ 25 detectors", RAW_UNITS
    else:
        r = _pick_real(runs, data, [pick], crop).iloc[0]
        img = torch.load(r.path, map_location="cpu", weights_only=False)["x"][0, plane].numpy()
        lab, units = _real_label(r), _real_units(r)
    lab = label or lab
    if trim:
        img = img[trim:-trim, trim:-trim]
    H, W = img.shape
    y0, y1, x0, x1 = zoom
    if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
        raise ValueError(f"zoom {zoom} outside the image ({H} x {W} after trim={trim})")
    crop_img = img[y0:y1, x0:x1]
    nrm = _real_pnorm(img, pct, gamma)
    nrm_z = nrm if zoom_norm == "full" else _real_pnorm(crop_img, pct, gamma)
    cm, px_nm = _real_cmap(cmap), r.px_nm
    zh, zw = y1 - y0, x1 - x0

    # ---- figure: panels of the same height, room for the title on top ----
    th = _theme(theme)
    title_h = 0.45
    if layout == "side":
        ratios = [W / H, zw / zh]
        fig_w, fig_h = panel_size * sum(ratios), panel_size + title_h
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=th["bg"])
        gs = fig.add_gridspec(1, 2, width_ratios=ratios, wspace=0.03, left=0.005,
                              right=0.995, bottom=0.005, top=1 - title_h / fig_h)
        ax, az = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    else:
        h_over_w = zh / zw if layout == "zoom" else H / W
        fig_h = panel_size * h_over_w + title_h
        fig = plt.figure(figsize=(panel_size, fig_h), facecolor=th["bg"])
        gs = fig.add_gridspec(1, 1, left=0.005, right=0.995, bottom=0.005,
                              top=1 - title_h / fig_h)
        ax = az = fig.add_subplot(gs[0])
    for a in {ax, az}:
        a.set_facecolor(th["bg"])
        a.axis("off")

    if layout != "zoom":                                  # full image
        ax.imshow(img, cmap=cm, norm=nrm, interpolation="lanczos")
        _real_text(ax, 0.035, 0.965, lab)
        cb_rect = (0.925, 0.22, 0.022, 0.5)
        if layout == "inset" and _INSET_CORNERS[inset_loc][0]:   # inset on the right
            ix, iy, iw, ih = _inset_rect((H, W), zoom, inset_loc, inset_size)
            cb_rect = ((0.925, 0.22, 0.022, max(0.1, min(0.5, iy - 0.27)))
                       if _INSET_CORNERS[inset_loc][1]
                       else (0.925, iy + ih + 0.05, 0.022, max(0.1, min(0.5, 0.83 - iy - ih))))
        if colorbar:
            _real_colorbar(ax, nrm, cm, units, rect=cb_rect)
        _real_scalebar(ax, W, px_nm, left=(layout == "inset" and inset_loc == "lower right"))
        if layout == "inset":
            axins = _real_inset(ax, img, zoom, nrm_z, cm,
                                _inset_rect((H, W), zoom, inset_loc, inset_size))
            _real_scalebar(axins, zw, px_nm, size=8, y=0.08)
        else:
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), zw, zh, fill=False, ec="white",
                                   lw=1.2, ls="--"))

    if layout in ("side", "zoom"):                        # the crop
        az.imshow(crop_img, cmap=cm, norm=nrm_z, interpolation="lanczos")
        _real_text(az, 0.035, 0.965, lab if layout == "zoom" else "zoom",
                   13 if layout == "zoom" else 11, weight="bold" if layout == "zoom" else "normal")
        if colorbar:
            _real_colorbar(az, nrm_z, cm, units)
        _real_scalebar(az, zw, px_nm, size=11)
        if layout == "side":                              # box corners -> zoom corners
            for ya, yb in ((y0, -0.5), (y1, zh - 0.5)):
                fig.add_artist(ConnectionPatch(
                    xyA=(x1 - 0.5, ya - 0.5), coordsA=ax.transData, xyB=(-0.5, yb),
                    coordsB=az.transData, color=th["connector"], lw=0.8, alpha=0.7, ls="--"))

    where = f"plane {plane} ({'in focus' if plane == REG_PLANE else 'out of focus'})"
    fig.suptitle(f"{data} · {where} · zoom ({y0}:{y1}, {x0}:{x1})"
                 + (f" · crop {int(r.crop)}" if not pd.isna(r.crop) else ""),
                 color=th["ink"], fontsize=12, y=1 - 0.1 / fig_h)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight", facecolor=th["bg"])
    plt.show()
    return img, crop_img


def plot_real_raw(data, cmaps=("magma", "inferno", "hot", "glow_cyan", "glow_green", "gray"),
                  crop=None, trim=12, zoom=None, pct=99.8, gamma=0.8, ncols=3, panel_size=4.5,
                  colorbar=True, separate=False, theme=None, save=None):
    """
    Raw real data (sum of the 25 detectors, photon counts) with several colormaps, one panel
    each (same normalisation: pct-th percentile and gamma, as plot_real).

    cmaps    : names of REAL_CMAPS ("magma", "inferno", "hot", "glow_cyan", "glow_green") or
               any matplotlib colormap ("gray", "viridis", "cividis", "bone", ...; "_r" reversed)
    crop     : centre crop of this size (as run_real.py --crop); None = full image
    trim     : px removed from each border; zoom=(y0, y1, x0, x1) after trim: only that region
    pct/gamma: display normalisation (gamma < 1 lifts the dim structures)
    separate : True = one figure per colormap (e.g. for slides); save: path, with separate=True
               the colormap name is added (raw_magma.png, raw_gray.png, ...)
    Returns the raw image shown (array).
    """
    th = _theme(theme)
    raw = real_raw(data, crop)
    if trim:
        raw = raw[trim:-trim, trim:-trim]
    if zoom is not None:
        y0, y1, x0, x1 = zoom
        raw = raw[y0:y1, x0:x1]
    px_nm = _real_px_nm(data)
    nrm = _real_pnorm(raw, pct, gamma)
    H, W = raw.shape
    cmaps = [cmaps] if isinstance(cmaps, str) else list(cmaps)
    where = (f" · crop {crop}" if crop else "") + (f" · zoom {tuple(zoom)}" if zoom else "")

    def draw(ax, name):
        cm = _real_cmap(name)
        ax.imshow(raw, cmap=cm, norm=nrm, interpolation="lanczos")
        ax.set_facecolor(th["bg"])
        ax.axis("off")
        if colorbar:
            _real_colorbar(ax, nrm, cm, RAW_UNITS)
        _real_scalebar(ax, W, px_nm, size=11)

    if separate:
        for name in cmaps:
            fig = plt.figure(figsize=(panel_size, panel_size * H / W + 0.45), facecolor=th["bg"])
            ax = fig.add_axes([0.005, 0.005, 0.99, 1 - 0.45 / (panel_size * H / W + 0.45)])
            draw(ax, name)
            if save:
                root_, ext_ = os.path.splitext(save)
                fig.savefig(f"{root_}_{name}{ext_ or '.png'}", dpi=200, bbox_inches="tight",
                            facecolor=th["bg"])
            plt.show()
        return raw

    ncols = min(ncols, len(cmaps))
    nrows = int(np.ceil(len(cmaps) / ncols))
    fig_h = panel_size * nrows * H / W + 0.5
    fig = plt.figure(figsize=(panel_size * ncols, fig_h), facecolor=th["bg"])
    gs = fig.add_gridspec(nrows, ncols, wspace=0.02, hspace=0.02, left=0.005, right=0.995,
                          bottom=0.005, top=1 - 0.5 / fig_h)
    for k in range(nrows * ncols):
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        if k < len(cmaps):
            draw(ax, cmaps[k])
        else:
            ax.set_facecolor(th["bg"])
            ax.axis("off")
    fig.suptitle(f"{data} · raw (Σ 25 detectors){where} · p{pct:g}, γ={gamma:g}",
                 color=th["ink"], fontsize=12, y=1 - 0.12 / fig_h)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight", facecolor=th["bg"])
    plt.show()
    return raw


def plot_real_detectors(data, cmap="magma", crop=None, trim=12, zoom=None, shared_scale=True,
                        pct=99.8, gamma=0.8, labels=False, colorbar=True, panel_size=2.2,
                        theme=None, save=None):
    """
    The 25 detector images of the real data in a 5 x 5 grid, as the SPAD array (detector
    12 = centre), photon counts.

    shared_scale: True = one colour scale for all (pct-th percentile over the 25 images: the
                  relative brightness of the detectors, the "fingerprint", is visible);
                  False = each image normalised to its own percentile (shapes / shifts)
    crop / trim / zoom=(y0, y1, x0, x1) (after trim) / pct / gamma / cmap: as plot_real_raw
    labels      : detector index in the corner of each image
    Returns the (25, h, w) images shown.
    """
    th = _theme(theme)
    y = real_detectors(data, crop)
    if trim:
        y = y[:, trim:-trim, trim:-trim]
    if zoom is not None:
        y0, y1, x0, x1 = zoom
        y = y[:, y0:y1, x0:x1]
    H, W = y.shape[-2:]
    cm, px_nm = _real_cmap(cmap), _real_px_nm(data)
    shared = _real_pnorm(y, pct, gamma)                    # percentile over all 25 images
    fig_h = 5 * panel_size * H / W + 0.5
    fig = plt.figure(figsize=(5 * panel_size, fig_h), facecolor=th["bg"])
    gs = fig.add_gridspec(5, 5, wspace=0.02, hspace=0.02, left=0.005, right=0.995,
                          bottom=0.005, top=1 - 0.5 / fig_h)
    for j in range(25):
        ax = fig.add_subplot(gs[j // 5, j % 5])
        nrm = shared if shared_scale else _real_pnorm(y[j], pct, gamma)
        ax.imshow(y[j], cmap=cm, norm=nrm, interpolation="lanczos")
        ax.set_facecolor(th["bg"])
        ax.axis("off")
        if labels:
            _real_text(ax, 0.06, 0.94, f"{j}", 9)
        if colorbar and (not shared_scale or j == 24):    # shared: one, bottom right
            _real_colorbar(ax, nrm, cm, "counts" if j == 24 else None, size=6,
                           rect=(0.86, 0.2, 0.05, 0.6))
        if j == 20:                                        # scale bar bottom left
            _real_scalebar(ax, W, px_nm, size=8, y=0.1, left=True)
    fig.suptitle(f"{data} · 25 detectors ({'shared scale' if shared_scale else 'own scale each'})"
                 + (f" · crop {crop}" if crop else "") + (f" · zoom {tuple(zoom)}" if zoom else ""),
                 color=th["ink"], fontsize=12, y=1 - 0.12 / fig_h)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight", facecolor=th["bg"])
    plt.show()
    return y


def plot_real_split(runs, data, method=None, lam=None, plane=REG_PLANE, crop="any", iters=None,
                    trim=12, region=None, cmap="magma", pct=99.8, gamma=0.8, norm="visible",
                    flip=False, label=None,
                    title=None, size_px=None, colorbar=True, save=None, aspect=None, pos=0.5):
    """
    Publicity figure (as publicity_figures.fig_split): diagonal split, raw data (upper left)
    | reconstruction (lower right), glowing seam, labels and scale bar, no axes.
    aspect: e.g. "16:9": a 16:9 frame (1920 x 1080 PNG; size_px = width), the full width of
    the field at vertical position pos (0 = top, 0.5 = centre, 1 = bottom); with region,
    the window keeps the region's centre and height and is widened. The seam then runs
    corner to corner. None: the square field (size_px default 1600).
    method / lam / iters: which run_real.py result (first match), e.g. method="RL", iters=20. region=(y0, x0, size): a square
    detail instead of the full field of view (after trim). label: text of the
    reconstruction side (default: "<method> · λ ...").
    pct / gamma: one value, or (raw, rec) to normalise the two sides differently (the
    in-focus reconstruction is sparser than the raw data, which also holds the out-of-focus
    haze: e.g. gamma=(0.8, 0.6) lifts its dim structures).
    norm: "visible" = each side normalised on the half it shows (balanced split, default),
    "full" = on the whole field (the same normalisation as plot_real).
    flip: reconstruction upper left, raw lower right (when the sample is dense there).
    colorbar: one small horizontal colorbar per side, next to its label, in real units.
    """
    sel = _select_real(runs, data, method, lam, crop, iters)
    if sel.empty:
        print(f"no run_real.py result for {data} {method} {lam} {iters}")
        return
    r = sel.iloc[0]
    rec = torch.load(r.path, map_location="cpu", weights_only=False)["x"][0, plane].numpy()
    raw = real_raw(data, r.crop)
    if trim:
        raw, rec = raw[trim:-trim, trim:-trim], rec[trim:-trim, trim:-trim]
    aspect = _parse_aspect(aspect)
    if aspect:
        y0, y1, x0, x1 = _aspect_window(raw.shape, aspect, region, pos)
        raw, rec = raw[y0:y1, x0:x1], rec[y0:y1, x0:x1]
    elif region:
        y0, x0, s = region
        raw, rec = raw[y0:y0 + s, x0:x0 + s], rec[y0:y0 + s, x0:x0 + s]
    cm = _real_cmap(cmap)
    H, W = raw.shape
    d, _ = _diag_dist(H, W)                                         # distance from the '/' diagonal
    (p_raw, p_rec), (g_raw, g_rec) = (np.broadcast_to(np.asarray(v, float), 2)
                                      for v in (pct, gamma))
    # each side normalised on the half it shows (norm="visible") or on the whole field
    rec_side = (d <= 0) if flip else (d > 0)
    m_raw, m_rec = (~rec_side, rec_side) if norm == "visible" else (None, None)
    raw_rgb = cm(_real_norm(raw, p_raw, g_raw, m_raw))[..., :3]
    rec_rgb = cm(_real_norm(rec, p_rec, g_rec, m_rec))[..., :3]
    n_raw, n_rec = _real_pnorm(raw, p_raw, g_raw, m_raw), _real_pnorm(rec, p_rec, g_rec, m_rec)
    rgb = np.where(rec_side[..., None], rec_rgb, raw_rgb)
    w = W / 1000                                   # seam widths scale with the field of view
    seam = np.clip(np.exp(-(d / (1.5 * w)) ** 2) + 0.35 * np.exp(-(d / (12 * w)) ** 2),
                   0, 1)[..., None]
    rgb = rgb * (1 - seam) + seam                                   # white glowing seam

    fig, ax, s, (X, Y, kx, ky, R) = _real_canvas(rgb, size_px, aspect)
    rec_label = label or _real_name(r)
    upper, lower = (rec_label, "Raw ISM data") if flip else ("Raw ISM data", rec_label)
    _real_text(ax, X(0.035), Y(0.965), upper, 26 * s)
    _real_text(ax, X(0.965), Y(0.035), lower, 26 * s, "right", "bottom")
    if colorbar:
        cbs = [(n_raw, RAW_UNITS), (n_rec, _real_units(r))]
        (n_up, u_up), (n_lo, u_lo) = cbs[::-1] if flip else cbs
        _real_colorbar(ax, n_up, cm, u_up, rect=R(_CB_UL), orientation="horizontal", size=11 * s)
        _real_colorbar(ax, n_lo, cm, u_lo, rect=R(_CB_LR), orientation="horizontal", size=11 * s)
    if title:
        _real_text(ax, X(0.965), Y(0.965), title, 15 * s, "right", "top", alpha=0.85,
                   weight="normal")
    _real_scalebar(ax, W, r.px_nm, size=17 * s, y=Y(0.11), x_pad=0.04 * kx,
                   dy=0.015 * ky)                                  # above the label
    if save:
        fig.savefig(save, dpi=100, facecolor="black")
    plt.show()
    return r


def plot_real_bands(runs, data, picks=(("RL", 20), ("TV_smooth", None)), plane=REG_PLANE,
                    crop="any", trim=12, region=None, band=0.3, cmap="magma", pct=99.8,
                    gamma=0.8, norm="visible", flip=False, labels=None, title=None,
                    size_px=None, colorbar=True, save=None, aspect=None, pos=0.5):
    """
    aspect / pos: e.g. aspect="16:9", as in plot_real_split (1920 x 1080 PNG, seams parallel
    to the frame diagonal).
    Publicity figure with two diagonal seams (as publicity_figures.fig_bands): three bands,
    raw data (upper left) | picks[0] (middle) | picks[1] (lower right), e.g.
    raw | Richardson-Lucy | regularized. picks: two (method, value) as in _pick_real
    (lambda for regularized methods, iteration count for RL; None = any / the best match).
    band: half-width of the middle band, as a fraction of the centre-to-corner distance.
    pct / gamma: one value or one per band (raw, middle, lower right). norm: "visible" =
    each band normalised on the pixels it shows (default), "full" = on the whole field.
    flip: the order reversed (picks[1] upper left, raw lower right).
    labels: three texts (raw, middle, lower right) instead of the defaults.
    region=(y0, x0, size): a square detail (after trim). save: path (.png).
    colorbar: one small horizontal colorbar per band, next to its label, in real units.
    """
    if len(picks) != 2:
        raise ValueError("picks: two (method, value) pairs, e.g. [('RL', 20), ('TV_smooth', 0.00708)]")
    sel = _pick_real(runs, data, picks, crop)
    rows = [r for _, r in sel.iterrows()]
    imgs = [real_raw(data, rows[0].crop)]
    imgs += [torch.load(r.path, map_location="cpu", weights_only=False)["x"][0, plane].numpy()
             for r in rows]
    names = list(labels) if labels else ["Raw ISM data"] + [_real_name(r) for r in rows]
    units = [RAW_UNITS] + [_real_units(r) for r in rows]
    if trim:
        imgs = [im[trim:-trim, trim:-trim] for im in imgs]
    aspect = _parse_aspect(aspect)
    if aspect:
        y0, y1, x0, x1 = _aspect_window(imgs[0].shape, aspect, region, pos)
        imgs = [im[y0:y1, x0:x1] for im in imgs]
    elif region:
        y0, x0, s = region
        imgs = [im[y0:y0 + s, x0:x0 + s] for im in imgs]
    if flip:                                       # band order: upper left -> lower right
        imgs, names, units = imgs[::-1], names[::-1], units[::-1]

    H, W = imgs[0].shape
    d, reach = _diag_dist(H, W)                    # distance from the '/' diagonal
    half = band * reach                            # centre-to-corner distance * band
    masks = [d < -half, (d >= -half) & (d <= half), d > half]
    pcts, gammas = (np.broadcast_to(np.asarray(v, float), 3) for v in (pct, gamma))
    if flip:
        pcts, gammas = pcts[::-1], gammas[::-1]
    cm = _real_cmap(cmap)
    rgb = np.zeros((H, W, 3))
    band_norms = []
    for im, m, p_, g_ in zip(imgs, masks, pcts, gammas):
        mk = m if norm == "visible" else None
        layer = cm(_real_norm(im, p_, g_, mk))[..., :3]
        band_norms.append(_real_pnorm(im, p_, g_, mk))
        rgb[m] = layer[m]
    w = W / 1000                                   # seam widths scale with the field of view
    for e in (-half, half):
        de = d - e
        seam = np.clip(np.exp(-(de / (1.5 * w)) ** 2) + 0.35 * np.exp(-(de / (12 * w)) ** 2),
                       0, 1)[..., None]
        rgb = rgb * (1 - seam) + seam              # white glowing seams

    fig, ax, s, (X, Y, kx, ky, R) = _real_canvas(rgb, size_px, aspect)
    # upper-left corner: first band, lower-left corner: middle band, lower-right: last band
    _real_text(ax, X(0.035), Y(0.965), names[0], 26 * s)
    _real_text(ax, X(0.035), Y(0.035), names[1], 26 * s, "left", "bottom")
    _real_text(ax, X(0.965), Y(0.035), names[2], 26 * s, "right", "bottom")
    if colorbar:
        for nrm, u, rect in zip(band_norms, units, (_CB_UL, _CB_LL, _CB_LR)):
            _real_colorbar(ax, nrm, cm, u, rect=R(rect), orientation="horizontal", size=11 * s)
    if title:
        _real_text(ax, X(0.965), Y(0.965), title, 15 * s, "right", "top", alpha=0.85,
                   weight="normal")
    _real_scalebar(ax, W, rows[0].px_nm, size=17 * s, y=Y(0.11), x_pad=0.04 * kx,
                   dy=0.015 * ky)                  # above the lower-right label
    if save:
        fig.savefig(save, dpi=100, facecolor="black")
    plt.show()
    return sel

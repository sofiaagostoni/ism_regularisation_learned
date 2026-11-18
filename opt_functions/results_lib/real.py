"""
Real data (run_real.py): loading the runs and the raw data, selecting runs, labels, and
the publicity-style drawing helpers (themes, colormaps, normalisation, colorbars, labels,
scale bars, insets, profiles, non-square frames) used by real_figures.py.
"""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .common import PRIOR_NAMES


# ---------------------------------------------------------------------------
# Real data (run_real.py): no ground truth, figures only
# ---------------------------------------------------------------------------
# Look of ism_regularisation/publicity_figures.py (background: REAL_THEME, white by default;
# plot_real_split / plot_real_bands always black), each image
# normalised to its own percentile with a gamma (raw = counts summed over the 25 detectors,
# reconstruction = x in its own units, so absolute scales are not comparable), white bold
# labels with a dark stroke, scale bar. The reconstructions are shown on plane REG_PLANE
# (in focus) unless plane=0 (out-of-focus background).

REAL_ROOT = "Results/real_data"             # run_real.py
REAL_DATA_DIR = "Data/Real_data"
REAL_CMAPS = {
    "magma": "magma", "inferno": "inferno", "hot": "afmhot",
    "glow_cyan": ["#000000", "#03141c", "#0b4f6c", "#14a3c7", "#7fe7ff", "#ffffff"],
    "glow_green": ["#000000", "#03160a", "#0c5a26", "#23b14d", "#a6ff9e", "#ffffff"],
}
_REAL_FIELDS = ("data", "method", "algo", "fidelity", "lambda", "sigma", "step_scale", "flux",
                "tv_eps", "crop", "max_iter", "iterations_run", "bt_s", "time_s", "px_nm")
# PnP methods of run_real.py (DRUNet, PnP-PGD): no lambda, the iteration count is a parameter
PNP_METHODS = {"DRUNet", "DRUNet_ft", "DRUNet_scratch"}


def _real_cmap(name):
    from matplotlib.colors import LinearSegmentedColormap
    c = REAL_CMAPS.get(name, name)
    return (LinearSegmentedColormap.from_list(name, c) if isinstance(c, list)
            else plt.get_cmap(c))


def load_real_runs(root=REAL_ROOT, datas=None):
    """One row per run_real.py reconstruction (root/<data>/<tag>.pth), with its settings,
    "converged" (stopped early before max_iter) and "path". datas: only these datasets
    (a name or a list; None = all)."""
    rows = []
    keep = None if datas is None else ({datas} if isinstance(datas, str) else set(datas))
    for data in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if keep is not None and data not in keep:
            continue
        folder = os.path.join(root, data)
        if not os.path.isdir(folder):         # e.g. whiteness_real.csv (whiteness_real.py)
            continue
        for f in sorted(os.listdir(folder)):
            if f.endswith(".pth"):
                r = torch.load(os.path.join(folder, f), map_location="cpu", weights_only=False)
                rows.append({**{k: r.get(k) for k in _REAL_FIELDS},
                             "path": os.path.join(folder, f)})
    df = pd.DataFrame(rows, columns=list(_REAL_FIELDS) + ["path"])
    if len(df):
        df["converged"] = (df.iterations_run < df.max_iter).astype(object)
        # fixed iteration count, by design (Richardson-Lucy, PnP)
        df.loc[(df.method == "RL") | df.method.isin(PNP_METHODS), "converged"] = None
        df = df.sort_values(["data", "method", "crop", "lambda", "iterations_run"],
                            na_position="first")
    return df.reset_index(drop=True)


def real_raw(data, crop=None):
    """Sum of the 25 detectors (counts) of Data/Real_data/<data>_data.pth, with the same
    centre crop as run_real.py."""
    d = torch.load(os.path.join(REAL_DATA_DIR, f"{data}_data.pth"), map_location="cpu",
                   weights_only=False)
    raw = d["measurment"].float().sum(dim=(0, 1))
    if crop is not None and not pd.isna(crop) and crop < raw.shape[-1]:
        crop, n = int(crop), raw.shape[-1]
        s = (n - crop) // 2
        raw = raw[s:s + crop, s:s + crop]
    return raw.numpy()


def _not_converged(row):
    """True only for an iterative method stopped at max_iter (RL: None, never flagged)."""
    return row.converged is not None and not pd.isna(row.converged) and not bool(row.converged)


def _keep_values(col, values):
    """Rows whose value is one of `values`, plus the rows without a value (NaN / None)."""
    vals = np.atleast_1d(values).astype(float)
    return col.apply(lambda v: v is None or pd.isna(v)
                     or bool(np.isclose(float(v), vals, rtol=1e-6).any()))


def _select_real(runs, data, methods=None, lambdas=None, crop="any", iters=None, sigmas=None):
    """lambdas: keep these lambdas, plus the runs without one (Richardson-Lucy);
    sigmas: keep these sigmas (WCRR, WCRR_ISM, WCHS, DRUNet), plus the runs without one
    (TV_smooth, L1, RL); iters: keep these iteration counts (iterations_run, e.g. RL)."""
    sel = runs[runs.data == data]
    if methods is not None:
        sel = sel[sel.method.isin([methods] if isinstance(methods, str) else methods)]
    if lambdas is not None:
        sel = sel[_keep_values(sel["lambda"], lambdas)]
    if sigmas is not None:
        sel = sel[_keep_values(sel["sigma"], sigmas)]
    if iters is not None:
        sel = sel[sel.iterations_run.isin(np.atleast_1d(iters).astype(int))]
    if crop == "any" and sel.crop.nunique(dropna=False) > 1:
        crops = [int(c) for c in sel.crop.dropna().unique()]
        print(f"{data}: full-image runs shown (crops {crops} also there: crop=<size>)")
        crop = None
    if crop != "any":
        sel = sel[sel.crop.isna()] if crop is None else sel[sel.crop == crop]
    return sel


def _real_norm(img, pct=99.8, gamma=0.8, mask=None):
    """img / its pct-th percentile (over mask if given), clipped to [0, 1], ** gamma."""
    img = np.clip(img, 0, None)
    v = np.percentile(img if mask is None else img[mask], pct)
    return np.clip(img / max(v, 1e-12), 0, 1) ** gamma


def _real_pnorm(img, pct=99.8, gamma=0.8, mask=None):
    """The matplotlib norm of _real_norm (same display: (img / percentile) ** gamma, clipped
    to [0, 1]), for imshow on the real values and for colorbars in the real units."""
    from matplotlib.colors import PowerNorm
    img = np.clip(img, 0, None)
    v = np.percentile(img if mask is None else img[mask], pct)
    return PowerNorm(gamma, vmin=0, vmax=max(float(v), 1e-12), clip=True)


RAW_UNITS = "photon counts (Σ 25 det.)"


def _real_units(row):
    """Units of a reconstruction: photon counts (flux 1) or x = intensity / alpha."""
    flux = row.flux
    if flux is None or pd.isna(flux) or float(flux) == 1:
        return "photon counts"
    return f"intensity / α  (α = {float(flux):.3g})"


def _real_colorbar(ax, norm, cmap, label=None, rect=(0.925, 0.22, 0.022, 0.5),
                   orientation="vertical", size=8):
    """Colorbar inside ax (rect in axes fraction), white ticks with a dark stroke, in the real
    units of the image (norm from _real_pnorm). Vertical: ticks and label on the left
    (towards the image centre); horizontal: ticks below, label above."""
    import matplotlib.patheffects as pe
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator
    cax = ax.inset_axes(rect)
    cb = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                            orientation=orientation)
    cb.locator = MaxNLocator(4)
    cb.update_ticks()
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(0.6)
    stroke = [pe.withStroke(linewidth=2.5, foreground=(0, 0, 0, 0.6))]
    cax.tick_params(colors="white", labelsize=size, length=2, width=0.6)
    if orientation == "vertical":
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    else:
        cax.xaxis.set_ticks_position("bottom")
        cax.xaxis.set_label_position("top")
    for t in cax.get_xticklabels() + cax.get_yticklabels():
        t.set_path_effects(stroke)
    if label:
        cb.set_label(label, color="white", fontsize=size, path_effects=stroke)
    return cb


# horizontal colorbar positions (axes fraction) in the split / bands figures, next to the
# label of each corner: upper left (below the label), lower left (above the label), lower
# right (above the scale bar)
_CB_UL, _CB_LL, _CB_LR = (0.035, 0.855, 0.22, 0.014), (0.035, 0.10, 0.22, 0.014), \
    (0.745, 0.21, 0.22, 0.014)


def _real_text(ax, x, y, s, size=13, ha="left", va="top", alpha=1.0, weight="bold"):
    import matplotlib.patheffects as pe
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, color="white", alpha=alpha,
            fontsize=size, weight=weight,
            path_effects=[pe.withStroke(linewidth=3, foreground=(0, 0, 0, 0.6 * alpha))])


def _real_scalebar(ax, width_px, px_nm, um=None, size=13, y=0.07, left=False, x_pad=0.04,
                   dy=0.015):
    """White scale bar bottom right (left=True: bottom left); um=None: a round length ~20%
    of the field of view. x_pad: margin from the side, dy: label above the bar (axes fraction)."""
    import matplotlib.patheffects as pe
    if not px_nm:
        return
    if um is None:
        fov = width_px * px_nm / 1000
        um = min([0.5, 1, 2, 5, 10, 20], key=lambda u: abs(u - 0.2 * fov))
    L = um * 1000 / px_nm / width_px
    x1 = x_pad + L if left else 1 - x_pad
    ax.plot([x1 - L, x1], [y, y], transform=ax.transAxes, color="white", lw=4,
            solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=6.5, foreground=(0, 0, 0, 0.5))])
    _real_text(ax, x1 - L / 2, y + dy, f"{um:g} µm", size, "center", "bottom")


# --- non-square publicity frames (plot_real_split / plot_real_bands, aspect="16:9") ---
def _parse_aspect(a):
    """'16:9' / '16/9' / 1.777 -> width / height (float); None -> None."""
    if a is None:
        return None
    if isinstance(a, str):
        for sep in (":", "/", "x"):
            if sep in a:
                w, h = a.split(sep)
                return float(w) / float(h)
    return float(a)


def _aspect_window(shape, aspect, region=None, pos=0.5):
    """(y0, y1, x0, x1) of a window of `aspect` (width / height) in an image of `shape`.
    region=None: the largest such window (full width for a square image) at vertical
    position pos (0 = top, 1 = bottom). region=(y0, x0, size): the window keeps the centre
    and the height of that square, widened to `aspect` and shifted to stay inside the image
    (lower if the image is not wide enough)."""
    H, W = shape
    if region is None:
        if W / H >= aspect:
            w = int(round(H * aspect))
            x0 = (W - w) // 2
            return 0, H, x0, x0 + w
        h = int(round(W / aspect))
        y0 = int(round(np.clip(pos, 0, 1) * (H - h)))
        return y0, y0 + h, 0, W
    y0, x0, s = region
    cy, cx = y0 + s / 2, x0 + s / 2
    h = min(s, H)
    w = int(round(h * aspect))
    if w > W:
        w, h = W, int(round(W / aspect))
    y0 = int(round(np.clip(cy - h / 2, 0, H - h)))
    x0 = int(round(np.clip(cx - w / 2, 0, W - w)))
    return y0, y0 + h, x0, x0 + w


def _diag_dist(H, W):
    """Signed distance (px) of every pixel from the '/' frame diagonal (lower-left to
    upper-right corner: 45 degrees for a square, ~29 degrees for 16:9), negative on the
    upper-left side, and the centre-to-corner distance along its normal (W / sqrt(2) for
    a square)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    reach = W * H / np.hypot(W, H)
    return ((xx - (W - 1) / 2) / W + (yy - (H - 1) / 2) / H) * reach, reach


def _frame_map(w_px, h_px):
    """Positions given in the axes fraction of a SQUARE frame -> the same distance (in units
    of sqrt(w_px * h_px)) from the nearest edge of a w_px x h_px frame (identity if square).
    Returns X, Y (coordinate maps), kx, ky (scale of x / y lengths) and R (rect map)."""
    S = np.sqrt(w_px * h_px)
    kx, ky = S / w_px, S / h_px

    def X(x):
        return x * kx if x <= 0.5 else 1 - (1 - x) * kx

    def Y(y):
        return y * ky if y <= 0.5 else 1 - (1 - y) * ky

    def R(rect):                       # (x0, y0, w, h), anchored at its nearest corner
        x0, y0, w, h = rect
        w2, h2 = w * kx, h * ky
        return (X(x0) if x0 + w / 2 <= 0.5 else X(x0 + w) - w2,
                Y(y0) if y0 + h / 2 <= 0.5 else Y(y0 + h) - h2, w2, h2)
    return X, Y, kx, ky, R


def _real_canvas(rgb, size_px=None, aspect=None):
    """Black full-bleed figure showing rgb. aspect: exact output width / height (1920 x 1080
    for 16:9 and size_px=None), None = the image's own (size_px=None: 1600 px wide).
    Returns fig, ax, s (text scale, 1 for 1600 x 1600) and _frame_map of the frame."""
    size_px = size_px or (1920 if aspect else 1600)
    H, W = rgb.shape[:2]
    h_px = size_px / aspect if aspect else size_px * H / W
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, h_px / dpi), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="lanczos", aspect="auto")
    ax.set_axis_off()
    return fig, ax, np.sqrt(size_px * h_px) / 1600, _frame_map(size_px, h_px)


_INSET_CORNERS = {"lower right": (1, 0), "lower left": (0, 0), "upper right": (1, 1)}
_PROFILE_LINE = "#7fe7ff"                      # profile line on the images (cyan)
# profile curves on the black background: raw data light grey, then the reconstructions
# in a fixed order (dark-mode categorical palette, validated adjacent-pair CVD-safe)
_PROFILE_RAW = "#c3c2b7"
_PROFILE_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                   "#9085e9", "#e66767"]

# Figure theme of plot_real / plot_real_single: background around the images, titles and
# the profile plot. The labels, colorbars and scale bars drawn ON the images stay white
# (they sit on the dark image). plot_real_split / plot_real_bands stay black (glow design).
REAL_THEMES = {
    "white": dict(bg="white", ink="#0b0b0b", muted="#52514e", grid="#e4e3dd",
                  spine="#b9b8b2", legend_edge="#d6d5cf", raw="#6b6b66", connector="#6b6b66",
                  colors=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                          "#6250d6", "#e34948"]),        # light-mode categorical slots 1-8
    "black": dict(bg="black", ink="#f0efec", muted="#c3c2b7", grid="#2c2c2a",
                  spine="#5a5955", legend_edge="#3a3a38", raw=_PROFILE_RAW, connector="white",
                  colors=_PROFILE_COLORS),
}
REAL_THEME = "white"                           # default; theme="black" per call for the old look


def _theme(theme):
    name = theme or REAL_THEME
    if name not in REAL_THEMES:
        raise ValueError(f"theme: one of {list(REAL_THEMES)}")
    return REAL_THEMES[name]


def set_real_theme(name):
    """Default theme of the real-data figures ("white" / "black"). Use this instead of
    assigning REAL_THEME from outside: compare_results.REAL_THEME = ... has no effect."""
    global REAL_THEME
    _theme(name)                                   # validates the name
    REAL_THEME = name


def _inset_rect(shape, box, loc, size, pad=0.025):
    """[x, y, w, h] (axes fraction) of a corner inset showing box=(y0, y1, x0, x1) of an
    image of `shape` with the right aspect; the longer side is `size`."""
    y0, y1, x0, x1 = box
    H, W = shape
    w, h = size, size * (W / H) * (y1 - y0) / (x1 - x0)
    if h > size:
        w, h = w * size / h, size
    cx, cy = _INSET_CORNERS[loc]
    return [1 - pad - w if cx else pad, 1 - pad - h if cy else pad, w, h]


def _real_inset(ax, im, box, norm, cmap, rect):
    """Zoom of box=(y0, y1, x0, x1) in a corner of ax (rect from _inset_rect), with the box
    and the connecting lines drawn on the image."""
    y0, y1, x0, x1 = box
    axins = ax.inset_axes(rect)
    axins.imshow(im, cmap=cmap, norm=norm, interpolation="lanczos")
    axins.set_xlim(x0 - 0.5, x1 - 0.5)
    axins.set_ylim(y1 - 0.5, y0 - 0.5)
    axins.set_xticks([])
    axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("white")
        s.set_linewidth(1.2)
    ax.indicate_inset_zoom(axins, edgecolor="white", linewidth=1.0, alpha=0.9)
    return axins


def _line_profile(img, line, width=1):
    """Profile of img along line=(y0, x0, y1, x1) (pixels, bilinear), averaged over `width`
    parallel lines 1 px apart. Returns (distance in px, values)."""
    from scipy.ndimage import map_coordinates
    y0, x0, y1, x1 = map(float, line)
    L = np.hypot(y1 - y0, x1 - x0)
    if L == 0:
        raise ValueError(f"profile line {line} has zero length")
    t = np.linspace(0, 1, int(round(L)) + 1)
    ys, xs = y0 + t * (y1 - y0), x0 + t * (x1 - x0)
    ny, nx = -(x1 - x0) / L, (y1 - y0) / L              # unit normal to the line
    vals = [map_coordinates(img, [ys + o * ny, xs + o * nx], order=1, mode="nearest")
            for o in np.arange(width) - (width - 1) / 2]
    return t * L, np.mean(vals, axis=0)


def _real_profile_axes(ax, imgs, line, width, norm, px_nm, theme=None):
    """Profiles of every (label, image) of imgs along line, in the figure theme (REAL_THEMES).
    norm: "max" = each profile / its max on the line, None = values in their own units."""
    th = _theme(theme)
    ink, muted, colors = th["ink"], th["muted"], th["colors"]
    px_nm = px_nm if px_nm and not pd.isna(px_nm) else None
    ax.set_facecolor(th["bg"])
    profiles = {}
    for k, (lab, im) in enumerate(imgs):
        d, p = _line_profile(im, line, width)
        if norm == "max":
            p = p / max(float(p.max()), 1e-12)
        lab = lab.replace("\n", " · ")
        profiles[lab] = p
        x = d * px_nm / 1000 if px_nm else d
        color = th["raw"] if k == 0 else colors[(k - 1) % len(colors)]
        ax.plot(x, p, color=color, lw=1.4 if k == 0 else 1.8, label=lab,
                alpha=0.9 if k == 0 else 1.0, zorder=2 if k == 0 else 3)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("distance along the line (µm)" if px_nm else "distance along the line (px)",
                  color=ink)
    ax.set_ylabel("intensity / max on the line" if norm == "max" else "intensity (own units)",
                  color=ink)
    ax.tick_params(colors=muted, labelcolor=muted)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["spine"])
    ax.grid(True, color=th["grid"], lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor=th["bg"],
                    edgecolor=th["legend_edge"], labelcolor=ink)
    leg.get_frame().set_alpha(0.85)
    y0, x0, y1, x1 = line
    ax.set_title(f"profile along the cyan line ({y0:g}, {x0:g}) → ({y1:g}, {x1:g})"
                 + (f", {width} px wide" if width > 1 else ""), color=ink, fontsize=10)
    return profiles


def _real_label(row, detail=False):
    """detail: also step cap and max iterations (runs with the same method and lambda)."""
    if row.method == "RL":
        return f"Richardson–Lucy · {int(row.iterations_run)} it"
    name = PRIOR_NAMES.get(row.method, row.method)
    if row.method in PNP_METHODS:
        step = row.get("step_scale")
        lab = f"{name}  σ={row.sigma:.3g}"
        if step is not None and not pd.isna(step):
            lab += f"  step={step:.3g}"
        return lab + f" · {int(row.iterations_run)} it"
    lab = f"{name}  λ={row['lambda']:.3g}"
    if row.sigma is not None and not pd.isna(row.sigma):
        lab += f"  σ={row.sigma:.3g}"
    if detail:
        cap = "" if row.bt_s is None or pd.isna(row.bt_s) else f"step≤{1 / row.bt_s:g}, "
        lab += f"\n{cap}≤{int(row.max_iter)} it"
    return lab


def _real_name(row):
    """Short display name for publicity figures: "Richardson–Lucy · 20 it" for RL,
    "Model-based · <prior>" otherwise."""
    if row.method == "RL":
        return f"Richardson–Lucy · {int(row.iterations_run)} it"
    if row.method in PNP_METHODS:
        return f"PnP · {PRIOR_NAMES.get(row.method, row.method)}"
    return f"Model-based · {PRIOR_NAMES.get(row.method, row.method)}"


def _pick_real(runs, data, picks, crop="any"):
    """
    One run_real.py result per pick, in the order given. picks: list of (method, value),
    value = lambda for the regularized methods, the iteration count for "RL", or None (any).
    Methods with lambda AND sigma (WCRR, WCRR_ISM, WCHS_free, ...): lambda or
    (lambda, sigma), e.g. ("WCRR_ISM", (50, 40)); lambda alone = any sigma.
    PnP (DRUNet, DRUNet_ft, DRUNet_scratch): the iteration count, (sigma, iterations) or
    (sigma, step, iterations); None = any.
    Several matches (e.g. the same lambda with another step cap / max_iter): the converged
    one with the most iterations allowed. All picks must have the same crop (crop=).
    """
    rows = []
    for method, value in picks:
        if method == "RL":
            sel = _select_real(runs, data, method, None, crop,
                               None if value is None else [value])
        elif method in PNP_METHODS:
            v = () if value is None else tuple(np.atleast_1d(value))
            sel = _select_real(runs, data, method, None, crop, [v[-1]] if v else None)
            if len(v) >= 2:
                sel = sel[np.isclose(sel.sigma.astype(float), float(v[0]), rtol=1e-6)]
            if len(v) == 3:
                sel = sel[np.isclose(sel.step_scale.astype(float), float(v[1]), rtol=1e-6)]
        else:
            v = () if value is None else tuple(np.atleast_1d(value))      # lambda or (lambda, sigma)
            sel = _select_real(runs, data, method, [v[0]] if v else None, crop,
                               sigmas=[v[1]] if len(v) >= 2 else None)
        if sel.empty:
            have = runs[runs.data == data]
            same = have[have.method == method]
            if same.empty:
                hint = f"methods available: {sorted(have.method.unique())}"
            else:
                hint = "available (lambda, sigma, iterations_run, crop): " + ", ".join(
                    sorted({str((r["lambda"], r.sigma, r.iterations_run, r.crop))
                            for _, r in same.iterrows()}))
            raise ValueError(f"{data}: no run_real.py result for {method} {value}"
                             + ("" if crop == "any" else f" (crop {crop})") + f"\n  {hint}")
        sel = sel.assign(_ok=[not _not_converged(r) for r in sel.itertuples()])
        rows.append(sel.sort_values(["_ok", "max_iter"]).iloc[-1].drop("_ok"))
    picked = pd.DataFrame(rows)
    if picked.crop.nunique(dropna=False) > 1:
        raise ValueError(f"{data}: the picks have different crops "
                         f"{picked.crop.tolist()}: choose one with crop=")
    return picked


def _real_px_nm(data):
    """Pixel size (nm) of Data/Real_data/<data>_data.pth (metadata dx in um), or None."""
    d = torch.load(os.path.join(REAL_DATA_DIR, f"{data}_data.pth"), map_location="cpu",
                   weights_only=False)
    dx = getattr(d.get("metadati"), "dx", None)
    return dx * 1e3 if dx else None


def real_detectors(data, crop=None):
    """The 25 detector images (25, N, N), photon counts, of Data/Real_data/<data>_data.pth,
    with the same centre crop as run_real.py."""
    d = torch.load(os.path.join(REAL_DATA_DIR, f"{data}_data.pth"), map_location="cpu",
                   weights_only=False)
    y = d["measurment"].float().reshape(25, *d["measurment"].shape[-2:])
    if crop is not None and not pd.isna(crop) and crop < y.shape[-1]:
        crop, n = int(crop), y.shape[-1]
        s = (n - crop) // 2
        y = y[:, s:s + crop, s:s + crop]
    return y.numpy()

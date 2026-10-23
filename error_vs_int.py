#%%
"""
Error as a function of ground-truth intensity.

Pixels are binned by their ground-truth value; inside each bin we compute error
statistics for every reconstruction. Curves are computed per sample (first dim =
images or noise realizations) and then averaged, so you also get a spread.

Metrics returned per bin:
    mae   : mean |x_hat - x|
    rmse  : sqrt(mean (x_hat - x)^2)
    rel   : sum |x_hat - x| / sum |x|        (relative error, robust to x ~ 0)
    nrmse : rmse / mean |x|
    bias  : mean (x_hat - x)                 (signed; L2 often biased in dark areas)
    snr         : 10 log10(sum x^2 / sum (x_hat - x)^2)        [dB]
    psnr_local  : 10 log10(max x in bin ^2 / mse)                [dB]
dB metrics: higher = better; the reference panel shows differences instead of ratios.

Also: Fourier ring correlation (FRC) of every reconstruction with the ground truth,
as a function of spatial frequency (frc_vs_gt / plot_frc).
"""
import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

METRICS = ("mae", "rmse", "rel", "nrmse", "bias", "snr", "psnr_local")
DB_METRICS = ("snr", "psnr_local")       # in dB, higher = better


# --------------------------------------------------------------------------- bins
def make_bins(x, n_bins=20, scheme="log", lo=None, hi=None, max_quantile_elems=2**24):
    """Bin edges from the ground truth. scheme in {'log', 'linear', 'quantile', 'levels'}.
    'levels': one bin per distinct GT value (piecewise-constant phantoms)."""
    x = x.detach().flatten().float()
    if scheme == "levels":
        u = torch.unique(x)
        if u.numel() > 256:
            raise ValueError(f"{u.numel()} distinct GT values: too many for 'levels'")
        return torch.cat([u[:1], (u[1:] + u[:-1]) / 2, u[-1:]])
    if x.numel() > max_quantile_elems:                        # torch.quantile size limit
        x = x[torch.randint(x.numel(), (max_quantile_elems,), device=x.device)]
    hi = x.max().item() if hi is None else hi

    if scheme == "linear":
        lo = x.min().item() if lo is None else lo
        edges = torch.linspace(lo, hi, n_bins + 1)
    elif scheme == "log":
        if lo is None:
            pos = x[x > 0]
            lo = torch.quantile(pos, 0.01).item()             # ignore the extreme tail
        edges = torch.logspace(math.log10(lo), math.log10(hi), n_bins + 1)
    elif scheme == "quantile":                                # ~equal pixel counts per bin
        edges = torch.unique(torch.quantile(x, torch.linspace(0, 1, n_bins + 1, device=x.device)))
    else:
        raise ValueError(scheme)
    return edges.to(x.device)


def _bin_index(x, edges):
    # interior edges -> indices 0..n_bins-1; values outside [lo, hi] go to first/last bin
    return torch.bucketize(x.contiguous(), edges[1:-1].contiguous())


def bin_map(x, edges):
    """Bin index of every pixel of x (same shape as x)."""
    return _bin_index(x.float().flatten(), edges.to(x.device)).reshape(x.shape)


# ------------------------------------------------------------------------ core stats
def _binned_sums(x, xh, idx, n_bins):
    e = (xh - x).double()
    bc = lambda w: torch.bincount(idx, weights=w, minlength=n_bins)
    return dict(
        n=torch.bincount(idx, minlength=n_bins).double(),
        abs=bc(e.abs()), sq=bc(e ** 2), err=bc(e),
        x=bc(x.double().abs()), x2=bc(x.double() ** 2),   # x2: signal energy per bin
    )


def _bin_max(x, idx, n_bins):
    """Max of x in every bin (0 for empty bins, whose metrics are NaN anyway)."""
    return torch.zeros(n_bins, dtype=torch.float64, device=x.device).scatter_reduce(
        0, idx, x.double(), "amax", include_self=False)


def _metrics_from_sums(s, peak_local):
    n = s["n"]
    mse = s["sq"] / n
    mean_x = s["x"] / n
    rmse = torch.sqrt(mse)
    return dict(
        mae=s["abs"] / n,
        rmse=rmse,
        rel=s["abs"] / s["x"],
        nrmse=rmse / mean_x,
        bias=s["err"] / n,
        snr=10 * torch.log10(s["x2"] / s["sq"]),
        psnr_local=10 * torch.log10(peak_local ** 2 / mse),
        center=mean_x,                                       # mean GT intensity in bin
    )


@torch.no_grad()
def error_vs_intensity(x_true, recons, n_bins=20, scheme="log", edges=None,
                       mask=None, min_count=50):
    """
    x_true : tensor (B, ...)  ground truth; B = images or noise realizations
    recons : dict name -> tensor with the same shape as x_true
    mask   : optional bool tensor (same shape) of pixels to include (e.g. crop borders)
    Returns dict with 'edges', 'centers', 'counts' and, for every method,
            {metric: (mean over B, std over B)}.
    """
    if edges is None:
        edges = make_bins(x_true if mask is None else x_true[mask], n_bins, scheme)
    n_bins = edges.numel() - 1
    B = x_true.shape[0]

    per_sample = {k: {m: [] for m in METRICS} for k in recons}
    centers, counts = [], []
    for b in range(B):
        x = x_true[b].flatten().float()
        keep = torch.ones_like(x, dtype=torch.bool) if mask is None else mask[b].flatten()
        x = x[keep]
        idx = _bin_index(x, edges)
        peak_local = _bin_max(x, idx, n_bins)
        for name, xh in recons.items():
            xh_b = xh[b].flatten().float()[keep]
            met = _metrics_from_sums(_binned_sums(x, xh_b, idx, n_bins), peak_local)
            for m in METRICS:
                per_sample[name][m].append(met[m])
        centers.append(met["center"])
        counts.append(torch.bincount(idx, minlength=n_bins))

    counts = torch.stack(counts).sum(0)
    valid = counts >= min_count * B                          # drop poorly populated bins
    out = dict(edges=edges.cpu(), counts=counts.cpu(),
               centers=torch.nanmean(torch.stack(centers), 0).cpu())
    out["centers"][~valid.cpu()] = float("nan")
    for name in recons:
        out[name] = {}
        for m in METRICS:
            v = torch.stack(per_sample[name][m])             # (B, n_bins)
            mean, std = torch.nanmean(v, 0), _nanstd(v, 0)
            mean[~valid], std[~valid] = float("nan"), float("nan")
            out[name][m] = (mean.cpu(), std.cpu())
    return out


def _nanstd(v, dim):
    mu = torch.nanmean(v, dim, keepdim=True)
    n = (~torch.isnan(v)).sum(dim).clamp(min=1)
    return torch.sqrt(torch.nansum((v - mu) ** 2, dim) / n)


# ------------------------------------------------------------------------------ FRC
def _tukey2d(H, W, alpha, device):
    """Separable Tukey window (alpha = tapered fraction; 0 = none, 1 = Hann)."""
    def tukey(n):
        w = torch.ones(n, dtype=torch.float64)
        if alpha <= 0:
            return w
        t = torch.linspace(0, 1, n, dtype=torch.float64)
        lo, hi = t < alpha / 2, t > 1 - alpha / 2
        w[lo] = 0.5 * (1 - torch.cos(2 * math.pi * t[lo] / alpha))
        w[hi] = 0.5 * (1 - torch.cos(2 * math.pi * (1 - t[hi]) / alpha))
        return w
    return (tukey(H)[:, None] * tukey(W)[None]).to(device)


@torch.no_grad()
def ring_spectra(a, b, window=0.25):
    """
    Ring sums of a and b over their last two dims, (..., H, W):
        cross = Re sum A B*,   Pa = sum |A|^2,   Pb = sum |B|^2      (each (..., R))
    window : Tukey taper applied before the FFT to suppress edge artefacts
    Returns (freq, cross, Pa, Pb), freq (R,) in cycles/pixel (0 .. 0.5).
    The error splits exactly per ring: sum |A - B|^2 = Pa + Pb - 2 FRC sqrt(Pa Pb).
    """
    H, W = a.shape[-2:]
    w = _tukey2d(H, W, window, a.device)
    Fa = torch.fft.fft2(a.double() * w).reshape(-1, H * W)
    Fb = torch.fft.fft2(b.double() * w).reshape(-1, H * W)
    n = min(H, W)                                             # ring width = 1 / n
    fy, fx = torch.fft.fftfreq(H, device=a.device), torch.fft.fftfreq(W, device=a.device)
    ring = torch.round(torch.sqrt(fy[:, None] ** 2 + fx[None] ** 2) * n).long().flatten()
    R = n // 2 + 1                                            # up to Nyquist; corners dropped
    keep = ring < R
    ring, Fa, Fb = ring[keep], Fa[:, keep], Fb[:, keep]

    rsum = lambda v: torch.zeros(v.shape[0], R, dtype=v.dtype, device=v.device).index_add_(1, ring, v)
    shape = lambda v: v.reshape(*a.shape[:-2], R)
    return (torch.arange(R, dtype=torch.float64) / n, shape(rsum((Fa * Fb.conj()).real)),
            shape(rsum(Fa.abs() ** 2)), shape(rsum(Fb.abs() ** 2)))


@torch.no_grad()
def frc(a, b, window=0.25):
    """
    Fourier ring correlation of a and b over their last two dims, (..., H, W).
    Returns (freq, curve): freq (R,) in cycles/pixel (0 .. 0.5), curve (..., R).
    """
    freq, cross, Pa, Pb = ring_spectra(a, b, window)
    return freq, cross / torch.sqrt(Pa * Pb)


def _crossing(freq, curve, threshold):
    """First frequency (linearly interpolated) where curve drops below threshold."""
    below = torch.nonzero(curve[1:] < threshold).flatten()
    if below.numel() == 0:
        return float("nan")                                   # never drops: resolution >= Nyquist
    i = below[0].item() + 1
    c0, c1 = curve[i - 1].item(), curve[i].item()
    return (freq[i - 1] + (c0 - threshold) / (c0 - c1) * (freq[i] - freq[i - 1])).item()


@torch.no_grad()
def frc_vs_gt(x_true, recons, window=0.25, threshold=0.5):
    """
    FRC of every reconstruction with the ground truth.
    x_true : tensor (B, ..., H, W); every 2-D slice (images, noise realizations, planes)
             gives one curve, and curves are then averaged
    threshold : 0.5 against a noise-free reference (equivalent to the 1/7 criterion
                between two independent half-data reconstructions)
    Returns dict with 'freq', 'threshold' and, for every method,
            {'frc': (mean, std), 'power_ratio': (mean, std), 'f_cut': crossing frequency
             [cycles/px]}, power_ratio = P_xhat / P_x per ring. Reading the two together:
        high FRC, ratio < 1        : right structure, attenuated (blur)
        low FRC,  ratio ~ 1 or > 1 : right amount of detail, wrong detail (hallucination)
        low FRC,  ratio << 1       : information lost, and not invented
    """
    out = dict(threshold=threshold)
    for name, xh in recons.items():
        freq, cross, Pr, Pg = ring_spectra(xh.float(), x_true.float().expand_as(xh), window)
        flat = lambda v: v.reshape(-1, v.shape[-1]).cpu()
        c, ratio = flat(cross / torch.sqrt(Pr * Pg)), flat(Pr / Pg)
        mean = c.mean(0)
        out["freq"] = freq
        out[name] = dict(frc=(mean, c.std(0, unbiased=False)),
                         power_ratio=(ratio.mean(0), ratio.std(0, unbiased=False)),
                         f_cut=_crossing(freq, mean, threshold))
    return out


@torch.no_grad()
def local_frc(a, b, size=64, stride=16, window=0.25, threshold=0.5):
    """
    Rolling FRC: FRC of a and b in size x size windows every `stride` pixels.
    a, b : 2-D images (leading singleton dims are squeezed)
    Returns dict with 'f_cut' (nW_y, nW_x) cut-off frequency of every window [cycles/px]
    (0.5 when the FRC never drops below threshold), 'centers_y', 'centers_x' (window
    centres in pixels), 'size', 'stride'. Windows with almost no ground-truth power at
    high frequency (flat regions) give noisy FRC: read those regions with care.
    """
    a, b = a.detach().squeeze().double(), b.detach().squeeze().double()
    assert a.ndim == 2, f"expected a 2-D image, got shape {tuple(a.shape)}"
    pa = a.unfold(0, size, stride).unfold(1, size, stride)    # (nY, nX, size, size)
    pb = b.unfold(0, size, stride).unfold(1, size, stride)
    freq, curves = frc(pa, pb, window)                        # (nY, nX, R)
    curves = curves.cpu()
    f_cut = torch.tensor([[_crossing(freq, c, threshold) for c in row] for row in curves])
    f_cut = torch.nan_to_num(f_cut, nan=0.5)
    first = torch.arange(f_cut.shape[1]) * stride + size / 2
    return dict(f_cut=f_cut, centers_y=torch.arange(f_cut.shape[0]) * stride + size / 2,
                centers_x=first, size=size, stride=stride, threshold=threshold)


# ---------------------------------------------------------------------------- plot
def plot_error_vs_intensity(res, metric="rel", reference=None, ax=None, logx=True,
                            logy=None, title=None, xlabel="ground-truth intensity"):
    """
    Plots metric vs intensity (mean ± std band). If `reference` names a method,
    a second panel shows metric(method) / metric(reference) for the others
    (metric(method) - metric(reference) for the dB metrics).
    """
    methods = [k for k in res if k not in ("edges", "centers", "counts")]
    db = metric in DB_METRICS
    logy = (metric != "bias" and not db) if logy is None else logy
    if ax is None:
        nrows = 2 if reference else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(6, 4.5 if nrows == 1 else 6.5),
                                 sharex=True, squeeze=False,
                                 gridspec_kw=dict(height_ratios=[3, 1.4][:nrows]))
        ax, ax_r = axes[0, 0], (axes[1, 0] if reference else None)
    else:
        fig, ax_r = ax.figure, None

    c = res["centers"].numpy()
    for name in methods:
        mu, sd = (t.numpy() for t in res[name][metric])
        line, = ax.plot(c, mu, "o-", ms=3, label=name)
        ax.fill_between(c, mu - sd, mu + sd, color=line.get_color(), alpha=0.2, lw=0)
    if metric == "bias":
        ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(f"{metric} [dB]" if db else metric)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if title: ax.set_title(title)

    if ax_r is not None:
        ref = res[reference][metric][0].numpy()
        for name in methods:
            if name == reference:
                continue
            v = res[name][metric][0].numpy()
            ax_r.plot(c, v - ref if db else v / ref, "o-", ms=3,
                      label=f"{name} {'-' if db else '/'} {reference}")
        ax_r.axhline(0 if db else 1, color="k", lw=0.8)
        ax_r.set_ylabel("difference [dB]" if db else "ratio")
        ax_r.grid(True, which="both", alpha=0.3)
        ax_r.legend(fontsize=8)
        ax_r.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig


def plot_bin_map(x, edges, mask=None, log_image=True, cmap="viridis", title=None):
    """
    Spatial map of the intensity bins: left = ground truth, right = bin index of
    every pixel. Colorbar labels give each bin's range and its share of pixels.
    x     : 2-D image (leading singleton dims are squeezed), e.g. gt[0]
    edges : bin edges, e.g. res['edges']
    mask  : optional bool tensor (same shape); excluded pixels are left blank
    """
    from matplotlib.colors import BoundaryNorm, LogNorm
    x = x.detach().squeeze()
    assert x.ndim == 2, f"expected a 2-D image, got shape {tuple(x.shape)}"
    edges = edges.to(x.device)
    n_bins = edges.numel() - 1
    idx = bin_map(x, edges).cpu().numpy().astype(float)
    img = x.float().cpu().numpy()
    if mask is not None:
        m = mask.detach().squeeze().cpu().numpy().astype(bool)
        idx[~m], img = np.nan, np.where(m, img, np.nan)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.3))
    e = edges.cpu().numpy()
    norm = LogNorm(vmin=e[0], vmax=e[-1]) if log_image and e[0] > 0 else None
    gray = plt.get_cmap("gray").copy(); gray.set_bad("black")
    fig.colorbar(a0.imshow(img, cmap=gray, norm=norm), ax=a0)
    a0.set_title("ground truth")

    im = a1.imshow(idx, cmap=plt.get_cmap(cmap, n_bins), interpolation="nearest",
                   norm=BoundaryNorm(np.arange(n_bins + 1) - 0.5, n_bins))
    cb = fig.colorbar(im, ax=a1, ticks=np.arange(n_bins))
    labels = [f"[{e[i]:.3g}, {e[i + 1]:.3g})" for i in range(n_bins)]
    if n_bins > 1:                                            # outer bins are open-ended
        labels[0], labels[-1] = f"< {e[1]:.3g}", f"\u2265 {e[-2]:.3g}"
    valid_idx = idx[~np.isnan(idx)].astype(int)
    frac = 100 * np.bincount(valid_idx, minlength=n_bins) / max(valid_idx.size, 1)
    cb.ax.set_yticklabels([f"{l}  {p:.1f}%" for l, p in zip(labels, frac)], fontsize=7)
    a1.set_title("intensity bins")
    for a in (a0, a1):
        a.set_xticks([]); a.set_yticks([])
    if title: fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_frc(res, axes=None, pixel_size=None, unit="px", ratio_thr=0.5, title=None):
    """
    Top: FRC vs spatial frequency (mean ± std band), threshold and, per method, the
    cut-off frequency; legend gives the resolution 1 / f_cut.
    Bottom: power ratio P_xhat / P_x per ring (log scale). Filled markers flag rings
    with FRC < threshold but power ratio > ratio_thr: detail that is there but does
    not match the ground truth (hallucination-like). Low FRC with ratio << 1 is lost,
    not invented, information.
    pixel_size : physical size of a pixel (e.g. in nm, then unit="nm"); None = pixels
    """
    methods = [k for k in res if k not in ("freq", "threshold")]
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 2]))
    ax, ax_p = axes
    fig = ax.figure
    ps = 1.0 if pixel_size is None else pixel_size
    f = res["freq"].numpy() / ps
    thr = res["threshold"]

    for name in methods:
        mu, sd = (t.numpy() for t in res[name]["frc"])
        fc = res[name]["f_cut"] / ps
        lab = f"{name}  (res = {1 / fc:.3g} {unit})" if np.isfinite(fc) and fc > 0 else name
        line, = ax.plot(f, mu, "-", lw=1.5, label=lab)
        col = line.get_color()
        ax.fill_between(f, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
        pr = res[name]["power_ratio"][0].numpy()
        ax_p.plot(f, pr, "-", lw=1.5, color=col)
        flag = (mu < thr) & (pr > ratio_thr)
        ax_p.plot(f[flag], pr[flag], "o", ms=3, color=col)
        if np.isfinite(fc):
            for a in (ax, ax_p):
                a.axvline(fc, color=col, ls="--", lw=0.8)
    ax.axhline(thr, color="k", ls=":", lw=0.8, label=f"threshold {thr:g}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("FRC with ground truth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax_p.axhline(1, color="k", lw=0.8)
    ax_p.axhline(ratio_thr, color="k", ls=":", lw=0.8)
    ax_p.set_yscale("log")
    ax_p.set_ylabel("power ratio P_rec / P_gt")
    ax_p.set_title(f"dots: FRC < {thr:g} and ratio > {ratio_thr:g} (hallucination-like)", fontsize=8)
    ax_p.grid(True, which="both", alpha=0.3)
    ax_p.set_xlim(0, f[-1])
    ax_p.set_xlabel(f"spatial frequency [1/{unit}]")
    if title: ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_local_frc(gt, recons, pairs=(), sample=0, plane=0, size=64, stride=16,
                   threshold=0.5, pixel_size=None, unit="px", title=None):
    """
    Local resolution maps from rolling FRC with the ground truth (see local_frc):
    ground truth, one map of 1 / f_cut per method (shared scale; smaller = finer),
    and res(a) - res(b) for every pair (blue: a finer). Regions with much worse local
    resolution than their neighbours are candidate artefacts.
    """
    from matplotlib.colors import TwoSlopeNorm
    ps = 1.0 if pixel_size is None else pixel_size
    img = lambda t: t[sample, plane].detach().float().cpu()
    x = img(gt)
    maps = {}
    for name, xh in recons.items():
        lf = local_frc(img(xh), x, size, stride, threshold=threshold)
        maps[name] = (ps / lf["f_cut"]).numpy()
    half = stride / 2                                          # each window shown as a stride tile
    cy, cx = lf["centers_y"].numpy(), lf["centers_x"].numpy()
    extent = (cx[0] - half, cx[-1] + half, cy[-1] + half, cy[0] - half)

    n = 1 + len(maps) + len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.2), squeeze=False)
    axes = axes[0]

    def show(ax, im, label, **kw):
        h = ax.imshow(im, interpolation="nearest", **kw)
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, x.shape[1]); ax.set_ylim(x.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])

    show(axes[0], x.numpy(), "ground truth", cmap="hot")
    vals = np.concatenate([m.ravel() for m in maps.values()])
    for ax, name in zip(axes[1:], maps):
        show(ax, maps[name], f"local resolution [{unit}]\n{name}", cmap="viridis_r",
             vmin=vals.min(), vmax=vals.max(), extent=extent)
    for ax, (a, b) in zip(axes[1 + len(maps):], pairs):
        d = maps[a] - maps[b]
        lim = max(np.abs(d).max(), 1e-12)
        show(ax, d, f"res a - res b [{unit}]\na = {a}\nb = {b}", cmap="RdBu_r",
             norm=TwoSlopeNorm(0, -lim, lim), extent=extent)
    fig.suptitle((title + "\n" if title else "")
                 + f"rolling FRC, {size}x{size} px windows every {stride} px, threshold {threshold:g}",
                 fontsize=10)
    fig.tight_layout()
    return fig


def _gauss_smooth(img, sigma):
    """2-D Gaussian smoothing of a numpy image (reflect padding)."""
    import torch.nn.functional as F
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1).double() ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    t = torch.from_numpy(img).double()[None, None]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return t[0, 0].numpy()


def plot_error_maps(gt, recons, a, b, edges, sample=0, plane=0, smooth=2.0,
                    pct=99, title=None):
    """
    Where does method `a` beat method `b`? Blue = `a` better, red = `b` better.
    Row 1: ground truth, reconstructions a and b, and the MAE ratio a/b of every
           intensity bin painted on that bin's pixels (log2 scale).
    Row 2: |error| of a and b (same scale), the pixel-wise difference |e_a| - |e_b|,
           and the same difference smoothed with a Gaussian of `smooth` pixels.
    gt, recons : as for error_vs_intensity, (B, C, H, W)
    edges      : bin edges, e.g. res['edges']
    pct        : percentile used to set the colour limits (robust to outliers)
    """
    from matplotlib.colors import TwoSlopeNorm
    img = lambda t: t[sample, plane].detach().float().cpu().numpy()
    x, ra, rb = img(gt), img(recons[a]), img(recons[b])
    ea, eb = np.abs(ra - x), np.abs(rb - x)
    d = ea - eb

    # per-bin MAE ratio, painted on the regions
    idx = bin_map(torch.from_numpy(x), edges.cpu()).numpy()
    n_bins = edges.numel() - 1
    n = np.bincount(idx.ravel(), minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log2(np.bincount(idx.ravel(), ea.ravel(), n_bins)
                        / np.bincount(idx.ravel(), eb.ravel(), n_bins))
    ratio[n == 0] = np.nan
    region = ratio[idx]

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    vmax_img = max(x.max(), np.percentile(ra, pct), np.percentile(rb, pct))
    vmax_err = max(np.percentile(ea, pct), np.percentile(eb, pct), 1e-12)

    def show(ax, im, label, **kw):
        h = ax.imshow(im, interpolation="nearest", **kw)
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    def diverging(im):                                        # symmetric limits around 0
        lim = np.nanpercentile(np.abs(im), pct) if np.isfinite(im).any() else 1.0
        return dict(cmap="RdBu_r", norm=TwoSlopeNorm(0, -max(lim, 1e-12), max(lim, 1e-12)))

    show(axes[0, 0], x, "ground truth", cmap="hot", vmin=0, vmax=vmax_img)
    show(axes[0, 1], ra, a, cmap="hot", vmin=0, vmax=vmax_img)
    show(axes[0, 2], rb, b, cmap="hot", vmin=0, vmax=vmax_img)
    show(axes[0, 3], region, "per-bin log2(MAE a / MAE b)", **diverging(ratio))
    show(axes[1, 0], ea, f"|error| {a}", cmap="magma", vmin=0, vmax=vmax_err)
    show(axes[1, 1], eb, f"|error| {b}", cmap="magma", vmin=0, vmax=vmax_err)
    show(axes[1, 2], d, "|e_a| - |e_b|", **diverging(d))
    ds = _gauss_smooth(d, smooth)
    show(axes[1, 3], ds, f"|e_a| - |e_b|, smoothed (\u03c3={smooth:g} px)", **diverging(ds))

    fig.suptitle((title + "\n" if title else "")
                 + f"a = {a}   vs   b = {b}      (blue: a better, red: b better)", fontsize=11)
    fig.tight_layout()
    return fig

def plot_metric_maps(gt, res, metric="psnr_local", pairs=(), sample=0, plane=0,
                     cmap="viridis", title=None):
    """
    Per-bin metric painted on the pixels of every bin: ground truth, one map per
    method (shared colour scale), and metric(a) - metric(b) for every pair (a, b)
    (blue: a better, for the dB metrics where higher = better).
    gt  : (B, C, H, W) ground truth used for res (bins are taken from gt[sample, plane])
    res : output of error_vs_intensity (bin values are the means over B)
    """
    from matplotlib.colors import TwoSlopeNorm
    methods = [k for k in res if k not in ("edges", "centers", "counts")]
    x = gt[sample, plane].detach().float().cpu()
    idx = bin_map(x, res["edges"].cpu()).numpy()
    maps = {name: res[name][metric][0].numpy()[idx] for name in methods}
    db = metric in DB_METRICS
    unit = " [dB]" if db else ""

    n = 1 + len(methods) + len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.2), squeeze=False)
    axes = axes[0]

    def show(ax, im, label, **kw):
        h = ax.imshow(im, interpolation="nearest", **kw)
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    show(axes[0], x.numpy(), "ground truth", cmap="hot")
    vals = np.concatenate([m[np.isfinite(m)] for m in maps.values()])
    vmin, vmax = (vals.min(), vals.max()) if vals.size else (0, 1)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("lightgray")    # NaN = sparse bin
    for ax, name in zip(axes[1:], methods):
        show(ax, maps[name], f"{metric}{unit}\n{name}", cmap=cm, vmin=vmin, vmax=vmax)
    for ax, (a, b) in zip(axes[1 + len(methods):], pairs):
        d = maps[a] - maps[b]
        lim = max(np.nanmax(np.abs(d)) if np.isfinite(d).any() else 1.0, 1e-12)
        rdbu = plt.get_cmap("RdBu" if db else "RdBu_r").copy(); rdbu.set_bad("lightgray")
        show(ax, d, f"{metric}{unit}: a - b\na = {a}\nb = {b}",
             cmap=rdbu, norm=TwoSlopeNorm(0, -lim, lim))
    if title: fig.suptitle(title)
    fig.tight_layout()
    return fig

#%%

# ---------------------------------------------------------------------------- data
def load_best(methods, data, flux, fidelity=None, root=None, split_by_fidelity=False,
              df=None, return_results=False):
    """
    Best reconstruction (highest PSNR in summary_all.csv) of each method, loaded as in
    compare_results.py.
    methods : list of method names or unambiguous substrings, e.g. ["WCRR_ISM [PROJ_GD] KL"]
    df      : optional table from compare_results.load (otherwise read from `root`)
    Returns (gt, recons) with gt (1, C, H, W) and recons = {method name: tensor like gt};
    with return_results, also {method name: full saved result dict} (best parameters etc.).
    """
    import compare_results as cr
    root = cr.ROOT if root is None else root
    if df is None:
        df = cr.load(root, split_by_fidelity=split_by_fidelity)
    g = cr._subset(df, data, fidelity)
    g = g[g.flux == flux]
    if g.empty:
        raise ValueError(f"no results for data={data}, flux={flux}, fidelity={fidelity}")

    gt, recons, results = None, {}, {}
    for m in methods:
        name = cr._match_method(g, m)
        row = g.loc[g[g.method == name].psnr.idxmax()]
        d = cr.load_rec(row, root)
        x = d["x_gt"].float()
        if gt is None:
            gt = x
        elif x.shape != gt.shape or not torch.allclose(x, gt):
            raise ValueError(f"{name}: ground truth differs from that of {next(iter(recons))}")
        recons[name], results[name] = d["x_rec"].float(), d
        print(f"{name:45s} psnr={row.psnr:.2f}  run_id={row.run_id}  {row.params}")
    return (gt, recons, results) if return_results else (gt, recons)


# ---------------------------------------------------------------------------- main
if __name__ == "__main__":
    # ---- set these ----------------------------------------------------------
    DATA, FLUX = "tub_level", 20
    METHODS = [                   # method names (or unambiguous substrings) as in compare_results
        "TV-smooth [PROJ_GD] L2",
        "TV-smooth [PROJ_GD] KL",
    ]
    COMPARE = "auto"              # (a, b) pairs for the error maps, a = blue when better.
                                  # "auto": each KL method with the same method in L2;
                                  # if there is none, every KL method vs every L2 method
    FIDELITY = None               # "KL" / "L2" to restrict, None = both
    REFERENCE = None              # method (full name) to divide by in the ratio panel
    METRIC = "rel"                # one of METRICS
    MAP_METRICS = ("psnr_local",)               # painted on the bin regions as images
    SCHEME = "levels"             # 'levels' (one bin per GT value), 'log', 'linear', 'quantile'
    N_BINS = 16                   # ignored by 'levels'
    PLANE = None                  # 3-D data: channel to analyse (None = all channels)
    FRC_THRESHOLD = 0.5           # vs noise-free GT (= 1/7 criterion between half-data recons)
    PIXEL_SIZE, PIXEL_UNIT = None, "px"   # e.g. 40, "nm" to plot FRC in physical units
    LOCAL_FRC = dict(size=64, stride=16)  # rolling FRC windows; None = skip
    RESIDUAL = True               # normalized data residual (regenerates the measurements)
    HALF_DATA = False             # half-data FRC: 2 extra reconstructions per method per split
    HALF_DATA_SPLITS = 1
    OUT_DIR = "Results/error_vs_intensity"
    # -------------------------------------------------------------------------

    gt, recons, results = load_best(METHODS, DATA, FLUX, FIDELITY, return_results=True)
    recons_full = dict(recons)                                # all planes, for the data checks
    if PLANE is not None:
        gt, recons = gt[:, PLANE:PLANE + 1], {k: v[:, PLANE:PLANE + 1] for k, v in recons.items()}

    res = error_vs_intensity(gt, recons, n_bins=N_BINS, scheme=SCHEME)
    res_frc = frc_vs_gt(gt, recons, threshold=FRC_THRESHOLD)
    for name in recons:
        fc = res_frc[name]["f_cut"]
        print(f"FRC {name:41s} f_cut={fc:.4f} cycles/px  res={1 / fc:.2f} px")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{DATA}_flux{FLUX:g}"
    figs = {
        "error_vs_intensity": plot_error_vs_intensity(res, metric=METRIC, reference=REFERENCE,
                                                      title=f"{DATA}, flux {FLUX:g}"),
        "bias_vs_intensity": plot_error_vs_intensity(res, metric="bias", logy=False,
                                                     title=f"Signed bias, {DATA}, flux {FLUX:g}"),
        "snr_vs_intensity": plot_error_vs_intensity(res, metric="snr", reference=REFERENCE,
                                                    title=f"SNR per bin, {DATA}, flux {FLUX:g}"),
        "bin_map": plot_bin_map(gt[0, 0], res["edges"], title="Bin regions"),
        "frc": plot_frc(res_frc, pixel_size=PIXEL_SIZE, unit=PIXEL_UNIT,
                        title=f"FRC, {DATA}, flux {FLUX:g}"),
    }
    if COMPARE == "auto":
        kl = [m for m in recons if m.endswith(" KL")]
        l2 = [m for m in recons if m.endswith(" L2")]
        COMPARE = [(m, m[:-2] + "L2") for m in kl if m[:-2] + "L2" in recons] \
            or [(a, b) for a in kl for b in l2]
    for a, b in COMPARE:
        fig = plot_error_maps(gt, recons, a, b, res["edges"], title=f"{DATA}, flux {FLUX:g}")
        figs["maps_" + "".join(c if c.isalnum() else "_" for c in f"{a}_vs_{b}")] = fig
    for m in MAP_METRICS:
        figs[f"{m}_map"] = plot_metric_maps(gt, res, metric=m, pairs=COMPARE,
                                            title=f"{m} per bin, {DATA}, flux {FLUX:g}")
    if LOCAL_FRC:
        figs["local_frc"] = plot_local_frc(gt, recons, pairs=COMPARE, threshold=FRC_THRESHOLD,
                                           pixel_size=PIXEL_SIZE, unit=PIXEL_UNIT,
                                           title=f"{DATA}, flux {FLUX:g}", **LOCAL_FRC)

    if RESIDUAL or HALF_DATA:
        import data_consistency as dc
    if RESIDUAL:
        _, meas, _ = dc.regenerate(next(iter(results.values())))
        res_resid = dc.residual_check(recons_full, meas, FLUX)
        for name, r in res_resid.items():
            print(f"residual {name:36s} mean={r['mean']:+.3f}  var={r['var']:.3f}  (white: 0, 1)")
        figs["residual"] = dc.plot_residuals(res_resid, title=f"Data residual, {DATA}, flux {FLUX:g}")
    if HALF_DATA:
        res_half = dc.half_data_frc(results, n_splits=HALF_DATA_SPLITS)
        for name in recons:
            print(f"cut-off {name:37s} half-data {res_half[name]['f_cut']:.4f}"
                  f"  vs GT {res_frc[name]['f_cut']:.4f} cycles/px")
        figs["half_data_frc"] = dc.plot_half_data_frc(res_half, res_frc, pixel_size=PIXEL_SIZE,
                                                      unit=PIXEL_UNIT, title=f"{DATA}, flux {FLUX:g}")

    for name, fig in figs.items():
        fig.savefig(os.path.join(OUT_DIR, f"{name}_{tag}.png"), dpi=150)
    print(f"saved {', '.join(figs)} in {os.path.abspath(OUT_DIR)}")
    plt.show()
# %%

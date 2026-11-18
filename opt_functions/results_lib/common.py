"""
Shared by the simulated and the real-data results: display names of the priors, metric
labels and formats, image helpers (to_img, _crop) and method-name matching.

3D data: images show one axial plane (plane=REG_PLANE by default: plane 1, in focus, the
regularized one and the one of the metrics; plane 0 is out of focus);
plane=None shows the max projection over the planes.
"""
import pandas as pd
import torch
import matplotlib.pyplot as plt

from opt_functions.Solver_functions.prior_config import REG_PLANE   # 3D: in-focus plane (1)

# Display names; unknown priors are shown with their own name
PRIOR_NAMES = {
    "Drunet": "DRUNet",
    "Drunet_finetune": "DRUNet-ft",
    "Drunet_fromscratch": "DRUNet-scratch",
    "weakly_convex": "WCRR",
    "TV_smooth": "TV-smooth",
    "TV_nonsmooth": "TV",
    "L1": "L1-norm",              # not "L1": that is also a training-loss name
    "WCHS_free": "WCHS-free",
    "WCHS_free_smlm": "WCHS-free (SMLM)",
    # evaluate_methods.METHODS names of the PnP runs (run_real.py)
    "DRUNet_ft": "DRUNet-ft",
    "DRUNet_scratch": "DRUNet-scratch",
    # names used by old (legacy) result files
    "drunet": "DRUNet-ft",
    "drunet_gauss": "DRUNet",
}
EXPLICIT = {"WCRR", "WCRR_ISM", "WCRR_softplus_ISM", "weakly_convex",
            "weakly_convex_ISM", "TV_smooth", "TV_nonsmooth", "L1",
            "WCRR_allpass", "WCRR_ISM_allpass", "WCRR_softplus_ISM_allpass",
            "WCHS_free", "WCHS_free_smlm"}
PRIORS_WITHOUT_SIGMA = {"TV_smooth", "TV_nonsmooth", "L1"}          # sigma is not used
PRIORS_USING_LOSS = {"Drunet_finetune", "Drunet_fromscratch", "drunet"}   # one checkpoint per training loss


METRIC_LABELS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "microssim": "MicroSSIM"}
METRIC_FMT = {"psnr": "{:.2f}", "ssim": "{:.3f}", "microssim": "{:.3f}"}


def _fmt(v, f):
    return "-" if pd.isna(v) else format(v, f)


def _as_list(v):
    return None if v is None else ([v] if isinstance(v, (str, int, float)) else list(v))


def to_img(x, normalize=False, plane=REG_PLANE):
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


def _crop(img, crop):
    if crop is None:
        return img
    y0, y1, x0, x1 = crop
    return img[y0:y1, x0:x1]


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


def _method_colors(methods):
    """One colour family per prior; tab10 for the others."""
    fams = {"DRUNet-ft": plt.cm.Blues, "DRUNet": plt.cm.Greys, "WCRR": plt.cm.Reds,
            "TV-smooth": plt.cm.Greens, "L1-norm": plt.cm.Purples, "WCHS": plt.cm.Oranges}
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

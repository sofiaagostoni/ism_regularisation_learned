"""
Data-consistency checks of saved reconstructions (complements the ground-truth FRC
of error_vs_int.py).

normalized_residual : (y - A x_hat - b) / sqrt((A x_hat + b) / flux) per detector.
                      Consistent with Poisson data -> white noise, mean 0, variance 1;
                      structure in it = something in x_hat that the data contradict,
                      or something in the data missing from x_hat. Needs no ground
                      truth, so it also applies to real data.
half_data_frc       : split the photons by binomial thinning, reconstruct both halves
                      with the method's best parameters, FRC between the halves
                      (threshold 1/7). Catches noise-driven artefacts only: anything the
                      prior adds in the same way to both halves counts as signal, so
                      compare with the FRC against the ground truth.

The measurements are not saved with the results: they are regenerated with
make_measurements (same seed). Run on the same device type as the experiments
(torch's CPU and CUDA generators differ), otherwise the noise realization differs.
"""
import dataclasses
import numpy as np
import torch
import matplotlib.pyplot as plt

from opt_functions.data_preparation import make_measurements
from opt_functions.Solver_functions.pnp_utils import Config, run_experiment
from opt_functions.Solver_functions.prior_config import get_denoiser
from error_vs_int import frc, _crossing, _gauss_smooth


# ---------------------------------------------------------------------------- setup
def config_from_result(d, device=None):
    """Config of the run that produced the saved result dict d (from load_best)."""
    sig = d["config"]
    names = {f.name for f in dataclasses.fields(Config)}
    kw = {k: v for k, v in sig.items() if k in names and not k.endswith("_grid")}
    if "bt" in sig:
        kw.update(zip(("bt_s", "bt_eta", "bt_delta", "bt_max"), sig["bt"]))
    nz = d["x_gt"].shape[1]
    kw.setdefault("Nz", nz)
    cfg = Config(**kw, is_3d=nz > 1, algo=d["algo"])
    if device is not None:
        cfg.device = torch.device(device)
    return cfg


def regenerate(d, device=None):
    """(x, meas, cfg): ground truth and the measurements used for the saved result d."""
    cfg = config_from_result(d, device)
    x = d["x_gt"].float().to(cfg.device)
    return x, make_measurements(x, d["flux"], cfg), cfg


# ------------------------------------------------------------------------- residual
@torch.no_grad()
def normalized_residual(x_hat, meas, flux):
    """(y_j - A_j x_hat - b_j) / sqrt((A_j x_hat + b_j) / flux), stacked: (N_det, B, 1, H, W)."""
    x_hat = x_hat.float().to(meas["y"][0].device)
    r = []
    for p, y in zip(meas["physics"].physics_list, meas["y"]):
        mean = p.A(x_hat) + p.noise_model.bkg
        r.append((y - mean) / torch.sqrt(mean.clamp_min(1e-12) / flux))
    return torch.stack(r)


def radial_power(r):
    """Ring-averaged power spectrum of r (..., H, W), normalized so that white noise of
    unit variance gives 1 at every frequency. Returns (freq [cycles/px], spectrum (R,))."""
    H, W = r.shape[-2:]
    P = (torch.fft.fft2(r.double()).abs() ** 2 / (H * W)).reshape(-1, H * W).mean(0).cpu()
    n = min(H, W)
    fy, fx = torch.fft.fftfreq(H), torch.fft.fftfreq(W)
    ring = torch.round(torch.sqrt(fy[:, None] ** 2 + fx[None] ** 2) * n).long().flatten()
    R = n // 2 + 1
    keep = ring < R
    s = torch.zeros(R, dtype=torch.float64).index_add_(0, ring[keep], P[keep])
    c = torch.bincount(ring[keep], minlength=R).double()
    return torch.arange(R, dtype=torch.float64) / n, s / c


def residual_check(recons, meas, flux):
    """For every method: normalized residual summary.
    Returns {name: dict(map, mean, var, freq, spectrum)}, map = sum over detectors / sqrt(N_det)
    of plane 0 (still unit variance if consistent)."""
    out = {}
    for name, xh in recons.items():
        r = normalized_residual(xh, meas, flux)
        freq, spec = radial_power(r)
        out[name] = dict(map=(r.sum(0) / r.shape[0] ** 0.5)[0, 0].cpu(),
                         mean=r.mean().item(), var=r.var().item(), freq=freq, spectrum=spec)
    return out


def plot_residuals(res, smooth=2.0, lim=3.0, title=None):
    """
    Row 1: normalized residual (summed over detectors). Row 2: the same smoothed with a
    Gaussian of `smooth` px and rescaled to unit variance for white noise, so structure
    stands out of the noise. Right: ring-averaged power spectrum (1 = white).
    """
    names = list(res)
    M = len(names)
    fig = plt.figure(figsize=(4 * M + 5, 8))
    gs = fig.add_gridspec(2, M + 1, width_ratios=[1] * M + [1.3])
    white_std = 1 / (2 * smooth * np.sqrt(np.pi))             # std of smoothed unit white noise
    for j, name in enumerate(names):
        rm = res[name]["map"].numpy()
        rs = _gauss_smooth(rm, smooth) / white_std
        for i, (im, lab) in enumerate(((rm, "normalized residual"),
                                       (rs, f"smoothed (σ={smooth:g} px), white-noise units"))):
            ax = fig.add_subplot(gs[i, j])
            h = ax.imshow(im, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
            sub = f"\nmean {res[name]['mean']:.3f}, var {res[name]['var']:.3f}" if i == 0 else ""
            ax.set_title(f"{name}\n{lab}{sub}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[:, M])
    for name in names:
        ax.plot(res[name]["freq"][1:], res[name]["spectrum"][1:], lw=1.3, label=name)  # f=0: the mean
    ax.axhline(1, color="k", lw=0.8, label="white noise")
    ax.set_yscale("log")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("spatial frequency [cycles/px]")
    ax.set_ylabel("residual power (1 = white)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    if title: fig.suptitle(title)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- half-data FRC
@torch.no_grad()
def thin(meas, flux, seed=0):
    """Binomial thinning of the photon counts: two independent halves, each scaled as a
    measurement at flux / 2. Returns (y1, y2), lists of per-detector tensors."""
    g = torch.Generator(device=meas["y"][0].device).manual_seed(seed)
    y1, y2 = [], []
    for y in meas["y"]:
        n = torch.round(y * flux)
        n1 = torch.binomial(n, torch.full_like(n, 0.5), generator=g)
        y1.append(n1 / (flux / 2)); y2.append((n - n1) / (flux / 2))
    return y1, y2


@torch.no_grad()
def measurements_with(x, y, flux, cfg):
    """Measurement dict at `flux` (physics, data fidelities) holding the given y."""
    meas = make_measurements(x, flux, cfg)                    # its own y is replaced
    phys = meas["physics"].physics_list
    meas["y"] = type(meas["y"])(y)
    ones = torch.ones_like(x)
    L_kl = 0.0
    for p, y_j in zip(phys, y):                               # as in data_preparation._finish
        norm_A = p.A(ones).max() * p.A_adjoint(torch.ones_like(y_j)).max()
        L_kl = L_kl + y_j.abs().max() / p.noise_model.bkg ** 2 * norm_A
    meas["L_kl"] = L_kl
    meas["sum_y"] = torch.stack(list(y)).sum(0)
    meas["avg_y"] = meas["sum_y"] / len(y)
    return meas


def reconstruct(d, x, meas, flux, cfg, denoiser=None):
    """Reconstruction of meas with the best parameters of the saved result d (no search)."""
    r = run_experiment(cfg, d["name"], flux, d["loss"], d["fidelity"], algo=d["algo"],
                       sigmas=None if d["sigma"] is None else [d["sigma"]],
                       iters=[d["iter"]],
                       lambdas=None if d["lambda"] is None else [d["lambda"]],
                       steps=None if d["step_scale"] is None else [d["step_scale"]],
                       x=x, meas=meas, denoiser=denoiser, save=False, verbose=False)
    return r["x_rec"]


def half_data_frc(results, threshold=1 / 7, n_splits=1, device=None, seed=0):
    """
    Half-data FRC of every method (results = {name: saved result dict}, from
    load_best(..., return_results=True)). Each split reconstructs both halves at flux/2
    with the full-data best parameters (which may not be optimal at half flux).
    Returns dict with 'freq', 'threshold' and, per method,
        {'frc': (mean, std) over splits and planes, 'f_cut', 'halves': (x1, x2) of split 0}.
    """
    out = dict(threshold=threshold)
    for name, d in results.items():
        x, meas, cfg = regenerate(d, device)
        flux = d["flux"]
        den = get_denoiser(d["loss"], cfg)
        curves, halves = [], None
        for s in range(n_splits):
            y1, y2 = thin(meas, flux, seed + s)
            x1, x2 = (reconstruct(d, x, measurements_with(x, y, flux / 2, cfg), flux / 2, cfg, den)
                      for y in (y1, y2))
            freq, c = frc(x1.float(), x2.float())
            curves.append(c.reshape(-1, c.shape[-1]).cpu())
            halves = halves or (x1.cpu(), x2.cpu())
        c = torch.cat(curves)
        mean = c.mean(0)
        out["freq"] = freq
        out[name] = dict(frc=(mean, c.std(0, unbiased=False)),
                         f_cut=_crossing(freq, mean, threshold), halves=halves)
        print(f"half-data FRC {name:35s} f_cut={out[name]['f_cut']:.4f} cycles/px")
    return out


def plot_half_data_frc(res, res_gt=None, pixel_size=None, unit="px", title=None):
    """FRC between the two half-data reconstructions; with res_gt (frc_vs_gt), the
    cut-off against the ground truth is drawn dotted for comparison: a half-data
    resolution much better than the ground-truth one = reproducible but wrong detail."""
    methods = [k for k in res if k not in ("freq", "threshold")]
    ps = 1.0 if pixel_size is None else pixel_size
    f = res["freq"].numpy() / ps
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in methods:
        mu, sd = (t.numpy() for t in res[name]["frc"])
        fc = res[name]["f_cut"] / ps
        lab = f"{name}  (res = {1 / fc:.3g} {unit})" if np.isfinite(fc) and fc > 0 else name
        line, = ax.plot(f, mu, "-", lw=1.5, label=lab)
        ax.fill_between(f, mu - sd, mu + sd, color=line.get_color(), alpha=0.2, lw=0)
        if np.isfinite(fc):
            ax.axvline(fc, color=line.get_color(), ls="--", lw=0.8)
        if res_gt is not None and name in res_gt and np.isfinite(res_gt[name]["f_cut"]):
            ax.axvline(res_gt[name]["f_cut"] / ps, color=line.get_color(), ls=":", lw=1.2)
    ax.axhline(res["threshold"], color="k", ls=":", lw=0.8, label=f"threshold {res['threshold']:.3g}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, f[-1])
    ax.set_xlabel(f"spatial frequency [1/{unit}]")
    ax.set_ylabel("FRC between half-data reconstructions")
    ax.set_title((title + "\n" if title else "")
                 + "dashed: half-data cut-off, dotted: cut-off vs ground truth", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

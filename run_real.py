#%%
"""
One reconstruction of REAL 3D ISM data with parameters you choose (no ground truth: no
search, no metrics). Same forward model, fidelities, background convention, step-size
rule, backtracking and early stopping as the simulations (evaluate_methods.METHODS),
via data_preparation.real_measurements_3D.

Data: Data/Real_data/<name>_data.pth ('measurment' (25, 1, N, N) photon counts, 'PSF'
(25, 2, k, k) estimated from the data, plane 0 out of focus / plane 1 in focus). The
regularizer acts on plane 1 only (REG_PLANE), as in the 3D simulations.

Intensity scale (--flux = alpha, counts ~ Poisson(alpha (A x + eta))):
    --flux 1 (default)  x in photon counts. For TV_smooth, TV, L1 (any alpha only rescales
                        lambda); not for the learned priors (they need x in ~[0, 1]).
                        lambda is then directly comparable to ism_regularisation's real-data
                        runs (same KL in counts, same smoothed TV on plane 1, same eps).
    --flux auto         quick estimate for the learned priors (estimate_alpha): 20
                        Richardson-Lucy iterations in counts on the full image, alpha =
                        1.1 x 99.9th percentile of the in-focus plane -> x in ~[0, 1]. Same
                        alpha with and without --crop. Printed, and in the file name (flux...).
    --flux <alpha>      a fixed alpha (e.g. the one printed by a previous --flux auto run).

    python run_real.py --data 03_tomm20 --method WCRR --flux auto --sigma 5 15 40 --lam 1 5 25 --crop 200
                        # every (sigma, lambda) pair; WCRR sigma in the 0..255 convention of x in [0, 1]

    python run_real.py --data 01_tomm20 --method TV_smooth --lam 0.0111
    python run_real.py --data 01_tomm20 --method TV_smooth --lam 0.003 0.0111 0.03   # a few lambdas
    python run_real.py --data 03_tomm20 --method TV_smooth --lam 0.01 --crop 200    # centre crop, quick test
    python run_real.py --data 01_tomm20 --method TV_smooth --lam 0.0111 --eps 1     # other TV smoothing

Richardson-Lucy (no regularizer, early stopping is the regularization): one run, the
image is saved after each of the --iter counts (default RL_ITERS), one result file each:
    python run_real.py --data 01_tomm20 --method RL                     # 5, 10, 20, 50, 100 it
    python run_real.py --data 01_tomm20 --method RL --iter 10 30 100 300
Same forward model, background and KL (in counts for --flux 1) as the other methods:
    x <- x / A^T 1 * A^T( y / (A x + eta) ),   x >= 0,  x_0 = A^T y / A^T 1

PnP DRUNet (--method DRUNet / DRUNet_ft / DRUNet_scratch, PnP-PGD as in the simulations:
x <- D_sigma(x - tau grad f(x)) on the in-focus plane, tau = step / L_kl, not convergent,
the iteration count is a parameter): needs --flux auto; no --lam; every (sigma, step) pair
is one run up to max(--iter), one result file per iteration count (default PNP_ITERS):
    python run_real.py --data 03_tomm20 --method DRUNet --flux auto --sigma 0.005 0.01 0.02 --crop 200
    python run_real.py --data 03_tomm20 --method DRUNet_ft --flux auto --sigma 0.005 --step 1e4 1e5 --iter 20 50 100
(simulations, tub flux 20: sigma ~ 0.004 .. 0.012, step ~ 2e4 .. 3e5, 35 .. 500 it; L_kl
depends on the data, so the step does not transfer exactly: try a few)

Saved in Results/real_data/<data>/<tag>.pth (reconstruction in units of x, both planes,
on the CPU, the settings, the step sizes and, unless --no-history, the per-iteration
histories: cost F, ||z_{k+1} - z_k||, ||z_k||, criticality -> compare_results.
plot_real_convergence) and <tag>.png (sum of the 25
detectors, in-focus plane, out-of-focus plane). Finished runs are skipped (--overwrite).
"""
import argparse
import os
import time
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import evaluate_methods as em
from opt_functions.data_preparation import real_measurements_3D, batch_measurements
from opt_functions.Solver_functions import pnp_utils as pu
from opt_functions.Solver_functions.prior_config import REG_PLANE, get_denoiser, make_prior

SAVE_DIR = "Results/real_data"
DATA_DIR = "Data/Real_data"
LEARNED = {"WCRR", "WCRR_ISM", "WCHS_free", "WCHS_free_smlm"}
RL_ITERS = [5, 10, 20, 50, 100]      # Richardson-Lucy: default iteration counts saved
PNP_ITERS = [25, 50, 100, 200, 500]  # PnP (DRUNet): default iteration counts saved
PNP_STEPS = [1e4]                    # PnP: default step multiplier (tau = step / L_kl)


def richardson_lucy(meas, iters, history=True):
    """
    Richardson-Lucy for the 3D ISM model y_j ~ Poisson(flux (A_j x + eta_j)) / flux (meas of
    real_measurements_3D + batch_measurements: y (1, 25, N, N), BatchedISMPhysics), i.e.
    multiplicative EM on the same KL (with background) as the regularized methods, no prior:
        x_{k+1} = x_k / A^T 1 * A^T( y / (A x_k + eta) ),  x_0 = A^T y / A^T 1.
    A x is clamped to >= 0 before adding eta (FFT round-off near 0), x stays >= 0.
    Returns ({n: x after n iterations (CPU)} for n in iters, history or None, {n: seconds}).
    history (as ProxGDIteration with log_cost, for compare_results.plot_real_convergence):
      cost[k]         F(x_{k+1}) = flux * sum(A x + eta - y log(A x + eta)), float64
      step_norm[k]    ||x_{k+1} - x_k||,  iterate_norm[k]  ||x_{k+1}||
      criticality[k]  ||x_k - P_+(x_k - grad F(x_k))||, grad F = flux (A^T 1 - A^T(y / (A x + eta)))
    """
    physics, y, flux = meas["physics"], meas["y"], float(meas["flux"])
    eta = torch.as_tensor(meas["eta_vec"]).reshape(1, -1, 1, 1).to(y)
    iters = sorted({int(n) for n in iters})
    ones_T = physics.A_adjoint(torch.ones_like(y)).clamp_min(1e-12)      # A^T 1, per plane
    x = (physics.A_adjoint(y) / ones_T).clamp_min(0)
    out, times, h = {}, {}, {k: [] for k in ("cost", "step_norm", "iterate_norm", "criticality")}

    def kl(z):                                    # F in counts, up to a constant
        z64, y64 = z.double(), y.double()
        return flux * (z64 - y64 * torch.log(z64)).sum()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    z = physics.A(x).clamp_min(0) + eta
    for k in range(1, iters[-1] + 1):
        back = physics.A_adjoint(y / z)
        x_new = (x * back / ones_T).clamp_min(0)
        if history:                               # kept on the GPU, one transfer at the end
            grad = flux * (ones_T - back)
            h["criticality"].append((x - (x - grad).clamp_min(0)).double().norm())
            h["step_norm"].append((x_new - x).double().norm())
            h["iterate_norm"].append(x_new.double().norm())
        x = x_new
        z = physics.A(x).clamp_min(0) + eta
        if history:
            h["cost"].append(kl(z))
        if k in iters:
            out[k] = x.detach().cpu().clone()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times[k] = time.time() - t0
    hist = ({k: torch.stack(v).cpu().numpy() for k, v in h.items()} if history else None)
    return out, hist, times


def load_real(name, crop=None):
    d = torch.load(os.path.join(DATA_DIR, f"{name}_data.pth"), map_location="cpu",
                   weights_only=False)
    counts, PSF = d["measurment"].float(), d["PSF"].float()
    if crop is not None and crop < counts.shape[-1]:        # centre crop (quick tests)
        n = counts.shape[-1]
        s = (n - crop) // 2
        counts = counts[..., s:s + crop, s:s + crop]
    meta = d.get("metadati")
    px = getattr(meta, "dx", None)
    return counts, PSF, (px * 1e3 if px else None)


def tag_of(method, flux, lam, sigma, eps, crop, n_iter, bt_s=None):
    t = f"{method}_flux{flux:g}_lam{lam:g}"
    if sigma is not None:
        t += f"_sigma{sigma:g}"
    if eps is not None:
        t += f"_eps{eps:g}"
    if crop is not None:
        t += f"_crop{crop}"
    if bt_s is not None:
        t += f"_bts{bt_s:g}"
    return t + f"_it{n_iter}"


def figure(counts, x, path, title, px_nm):
    """Sum of the detectors, in-focus plane, out-of-focus plane (each with its own scale)."""
    panels = [(counts.sum(0)[0], "sum of 25 detectors (counts)"),
              (x[0, REG_PLANE], f"in focus (plane {REG_PLANE})"),
              (x[0, 1 - REG_PLANE], f"out of focus (plane {1 - REG_PLANE})")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (img, name) in zip(axes, panels):
        im = ax.imshow(img.numpy(), cmap="magma")
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    sub = f" ({px_nm:.1f} nm/px)" if px_nm else ""
    fig.suptitle(title + sub)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


ALPHA_RL_ITERS = 20       # --flux auto: Richardson-Lucy iterations of the quick estimate
ALPHA_PCT = 99.9          # percentile of the in-focus plane taken as the "white" level
ALPHA_MARGIN = 1.1        # alpha = margin * percentile, so that x stays ~<= 1


def estimate_alpha_from(counts, PSF, device, n_iter=ALPHA_RL_ITERS, pct=ALPHA_PCT,
                        margin=ALPHA_MARGIN, n_iters_extra=(), verbose=True):
    """
    alpha of estimate_alpha for any (25, 1, N, N) photon counts and (25, 2, k, k) PSF (real
    or simulated). n_iters_extra: more RL iteration counts, all from the same run; then
    returns {n_iter: alpha} for n_iter and every extra count (check_alpha.py), else alpha.
    """
    meas = batch_measurements(real_measurements_3D(counts, PSF, 1.0, device))
    counts_all = sorted({n_iter, *n_iters_extra})
    recs, _, _ = richardson_lucy(meas, counts_all, history=False)
    alphas = {}
    for n in counts_all:
        x_in = recs[n][0, REG_PLANE].flatten()
        q = float(torch.quantile(x_in[::max(1, x_in.numel() // 4_000_000)], pct / 100))
        alphas[n] = float(f"{margin * q:.3g}")
        if verbose and n == n_iter:
            print(f"--flux auto: alpha = {margin} x p{pct:g}(in-focus plane, RL {n} it, "
                  f"counts) = {alphas[n]:g}  (max {float(x_in.max()):.4g} -> x max ~ "
                  f"{float(x_in.max()) / alphas[n]:.2f})")
    return alphas if n_iters_extra else alphas[n_iter]


def estimate_alpha(name, device, n_iter=ALPHA_RL_ITERS, pct=ALPHA_PCT, margin=ALPHA_MARGIN):
    """
    Quick intensity scale alpha for the learned priors (x = intensity / alpha in ~[0, 1]):
    n_iter Richardson-Lucy iterations in photon counts (flux 1) on the FULL image (also
    when the run uses --crop, so crop tests and full runs share alpha and lambda / sigma),
    alpha = margin * pct-th percentile of the in-focus plane (REG_PLANE), rounded to 3
    significant digits. A few seconds; only the order of magnitude matters (sigma and
    lambda are tuned at this alpha). Checked by check_alpha.py.
    """
    counts, PSF, _ = load_real(name, None)
    return estimate_alpha_from(counts, PSF, device, n_iter, pct, margin)


def _flux_arg(v):
    return v if v == "auto" else float(v)


def rl_tag(flux, crop, n_iter):
    t = f"RL_flux{flux:g}"
    if crop is not None:
        t += f"_crop{crop}"
    return t + f"_it{n_iter}"


def main_rl(args):
    """Richardson-Lucy: one run up to max(--iter), one result file per iteration count."""
    iters = sorted(set(args.iter or RL_ITERS))
    counts, PSF, px_nm = load_real(args.data, args.crop)
    N = counts.shape[-1]
    device = pu.default_device()
    out_dir = os.path.join(SAVE_DIR, args.data)
    os.makedirs(out_dir, exist_ok=True)
    paths = {n: os.path.join(out_dir, rl_tag(args.flux, args.crop, n) + ".pth") for n in iters}
    todo = [n for n in iters if args.overwrite or not os.path.exists(paths[n])]
    if not todo:
        print(f"{args.data} RL: all of {iters} done already")
        return
    meas = batch_measurements(real_measurements_3D(counts, PSF, args.flux, device))
    print(f"{args.data}: {N} x {N} px, {px_nm} nm/px, counts summed over detectors: "
          f"max {float(counts.sum(0).max()):.0f}, flux {args.flux:g}, Richardson-Lucy "
          f"(KL), iterations {iters}, device {device}")
    recs, hist, times = richardson_lucy(meas, todo, history=not args.no_history)
    for n in todo:
        x = recs[n]
        history = None if hist is None else {k: v[:n] for k, v in hist.items()}
        torch.save({"x": x, "data": args.data, "method": "RL", "algo": "RL", "fidelity": "KL",
                    "lambda": None, "sigma": None, "flux": args.flux, "tv_eps": None,
                    "crop": args.crop, "max_iter": n, "iterations_run": n, "L_history": None,
                    "stepsize0": None, "bt_s": None, "stop_tol": None, "history": history,
                    "px_nm": px_nm, "time_s": times[n]}, paths[n])
        figure(counts, x, paths[n].replace(".pth", ".png"),
               f"{args.data}  Richardson-Lucy  {n} it  flux {args.flux:g}", px_nm)
        print(f"  {n} it: {times[n]:.1f} s, finite {bool(torch.isfinite(x).all())}, "
              f"in-focus max {float(x[0, REG_PLANE].max()):.4g}, "
              f"out-of-focus max {float(x[0, 1 - REG_PLANE].max()):.4g}  -> {paths[n]}")


def pnp_tag(method, flux, sigma, step, crop, n_iter):
    t = f"{method}_flux{flux:g}_sigma{sigma:g}_step{step:g}"
    if crop is not None:
        t += f"_crop{crop}"
    return t + f"_it{n_iter}"


def main_pnp(args, m):
    """
    PnP-PGD with a DRUNet denoiser (evaluate_methods.METHODS "DRUNet", "DRUNet_ft",
    "DRUNet_scratch"), as in the simulations: x <- D_sigma(x - tau grad f(x)) on the
    in-focus plane (the out-of-focus plane: gradient step only), KL with background,
    tau = step / L_kl, no backtracking, no early stopping (not convergent: the iteration
    count is a parameter). One run per (sigma, step) up to max(--iter), one result file per
    iteration count. The denoiser output is clipped to [0, 1]: needs --flux auto (or alpha).
    """
    if args.sigma is None:
        raise SystemExit("PnP: --sigma is required (x in [0, 1] units, e.g. 0.005 0.01 0.02)")
    if args.flux == 1:
        raise SystemExit("PnP: DRUNet needs x in ~[0, 1] (its output is clipped to [0, 1]): "
                         "use --flux auto (or --flux <alpha>)")
    if args.lam is not None:
        print("PnP: --lam is ignored (the denoiser strength is sigma)")
    iters = sorted(set(args.iter or PNP_ITERS))
    steps = args.step or PNP_STEPS
    fidelity = m.get("fidelity", em.FIDELITY)
    counts, PSF, px_nm = load_real(args.data, args.crop)
    N = counts.shape[-1]
    cfg = replace(m["cfg"], is_3d=True, Nz=2, Nx=N, batched_physics=True,
                  x_max=None, init_choice="backprojection",
                  pxsizex=px_nm or m["cfg"].pxsizex)
    torch.backends.cudnn.benchmark = True
    out_dir = os.path.join(SAVE_DIR, args.data)
    os.makedirs(out_dir, exist_ok=True)
    meas = batch_measurements(real_measurements_3D(counts, PSF, args.flux, cfg.device))
    prior = make_prior(get_denoiser(em.LOSS, cfg), m["algo"], True)   # plane REG_PLANE only
    df, base_tau, _ = pu.fidelity_and_steps(fidelity, meas, 1.0, prior, cfg)
    print(f"{args.data}: {N} x {N} px, {px_nm} nm/px, counts summed over detectors: "
          f"max {float(counts.sum(0).max()):.0f}, flux {args.flux:g}, {args.method} "
          f"({m['algo']}, {fidelity}), 1 / L_kl = {base_tau:.3g}, iterations {iters}, "
          f"device {cfg.device}")

    for sigma, step in [(s, t) for s in args.sigma for t in steps]:
        paths = {n: os.path.join(out_dir, pnp_tag(args.method, args.flux, sigma, step,
                                                  args.crop, n) + ".pth") for n in iters}
        if not args.overwrite and all(os.path.exists(p_) for p_ in paths.values()):
            print(f"  sigma {sigma:g}, step {step:g}: all of {iters} done already")
            continue
        tau = base_tau * step
        t0 = time.time()
        recs, _ = pu.solve(None, meas["y"], meas["physics"], prior, df, tau, sigma, 1.0,
                           iters, m["algo"], cfg, None, log_cost=False,
                           metrics=not args.no_history)
        dt = time.time() - t0
        for n in iters:
            x, mets = recs[n]
            history = None
            if mets:                     # deepinv residual ||x_{k+1} - x_k|| / ||x_k||
                history = {k: np.asarray(mets[k][0][:n], dtype=np.float64)
                           for k in ("residual",) if k in mets and len(mets[k][0])}
            torch.save({"x": x, "data": args.data, "method": args.method, "algo": m["algo"],
                        "fidelity": fidelity, "lambda": None, "sigma": sigma,
                        "step_scale": step, "flux": args.flux, "tv_eps": None,
                        "crop": args.crop, "max_iter": n, "iterations_run": n,
                        "L_history": None, "stepsize0": tau, "bt_s": None, "stop_tol": None,
                        "history": history, "px_nm": px_nm,
                        "time_s": dt * n / max(iters)}, paths[n])
            figure(counts, x, paths[n].replace(".pth", ".png"),
                   f"{args.data}  {args.method}  sigma {sigma:g}  step {step:g}  {n} it  "
                   f"flux {args.flux:g}", px_nm)
            print(f"  sigma {sigma:g}, step {step:g} (tau {tau:.3g}), {n} it: finite "
                  f"{bool(torch.isfinite(x).all())}, in-focus max "
                  f"{float(x[0, REG_PLANE].max()):.4g}, out-of-focus max "
                  f"{float(x[0, 1 - REG_PLANE].max()):.4g}  -> {paths[n]}")
        print(f"    {dt / 60:.1f} min for {max(iters)} iterations")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="Data/Real_data/<data>_data.pth, e.g. 01_tomm20")
    p.add_argument("--method", required=True, choices=list(em.METHODS) + ["RL"],
                   help="a method of evaluate_methods.METHODS, or RL (Richardson-Lucy)")
    p.add_argument("--lam", nargs="+", type=float, default=None,
                   help="regularization weights (not for RL)")
    p.add_argument("--sigma", nargs="+", type=float, default=None,
                   help="learned priors only; several values: every (sigma, lambda) pair "
                        "(DRUNet: every (sigma, step) pair, sigma in x in [0, 1] units)")
    p.add_argument("--step", nargs="+", type=float, default=None,
                   help=f"PnP (DRUNet) only: step multiplier, tau = step / L_kl (default "
                        f"{PNP_STEPS}; simulations: ~1e4 .. 3e5)")
    p.add_argument("--flux", type=_flux_arg, default=1.0,
                   help="intensity scale alpha (1 = x in photon counts), or 'auto': quick "
                        "estimate so that x is in ~[0, 1] (estimate_alpha; for WCRR / WCHS)")
    p.add_argument("--eps", type=float, default=None, help="TV_smooth smoothing (default: method's)")
    p.add_argument("--iter", nargs="+", type=int, default=None,
                   help=f"max iterations (default {em.N_ITER}, early stop as the sims); RL: the "
                        f"iteration counts to save (default {RL_ITERS})")
    p.add_argument("--crop", type=int, default=None, help="centre crop of this size (quick tests)")
    p.add_argument("--bt-s", type=float, default=None,
                   help="largest step = 1 / bt_s (default: the method's, 1 for KL). In counts "
                        "(flux 1) the KL curvature is ~1 / counts, so steps > 1 are often accepted")
    p.add_argument("--no-history", action="store_true",
                   help="do not record the convergence histories (cost F, step norms, "
                        "criticality; ~45%% more time per iteration with them)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.flux == "auto":
        args.flux = estimate_alpha(args.data, pu.default_device())
    if args.method == "RL":
        return main_rl(args)
    if em.METHODS[args.method]["algo"].upper() == "PGD":           # PnP: DRUNet
        return main_pnp(args, em.METHODS[args.method])
    if args.lam is None:
        p.error("--lam is required (except for --method RL)")
    if args.iter is not None and len(args.iter) > 1:
        p.error("one --iter value (several only for RL)")
    args.iter = em.N_ITER if args.iter is None else args.iter[0]

    m = em.METHODS[args.method]
    if args.method.split("_L2")[0].split("_MD")[0] in LEARNED and args.flux == 1:
        print("WARNING: learned prior with flux 1 (x in counts): it was trained on x in [0, 1]; "
              "use --flux auto (or --flux ~ peak counts)")
    fidelity = m.get("fidelity", em.FIDELITY)
    counts, PSF, px_nm = load_real(args.data, args.crop)
    N = counts.shape[-1]
    # x_max=None: the output is never clipped at 1 (with --flux auto the brightest ~0.1% may
    # be slightly above 1; the iterates are not clipped either way)
    cfg = replace(m["cfg"], is_3d=True, Nz=2, Nx=N, batched_physics=True,
                  x_max=None, init_choice="backprojection",
                  pxsizex=px_nm or m["cfg"].pxsizex)
    if args.eps is not None:
        cfg = replace(cfg, tv_eps=args.eps)
    if args.bt_s is not None:
        cfg = replace(cfg, bt_s=args.bt_s)
    torch.backends.cudnn.benchmark = True        # large images: avoids slow cuDNN fallbacks

    out_dir = os.path.join(SAVE_DIR, args.data)
    os.makedirs(out_dir, exist_ok=True)
    meas = batch_measurements(real_measurements_3D(counts, PSF, args.flux, cfg.device))
    prior = make_prior(get_denoiser(em.LOSS, cfg), m["algo"], True)   # plane REG_PLANE only
    eps = cfg.tv_eps if cfg.prior_name == "TV_smooth" else None
    print(f"{args.data}: {N} x {N} px, {px_nm} nm/px, counts summed over detectors: "
          f"max {float(counts.sum(0).max()):.0f}, flux {args.flux:g}, {args.method} "
          f"({m['algo']}, {fidelity}), device {cfg.device}")

    sigmas = args.sigma if args.sigma is not None else [None]     # None: prior without sigma
    for sig_arg, lam in [(s, l) for s in sigmas for l in args.lam]:
        sigma = sig_arg if sig_arg is not None else 1.0
        tag = tag_of(args.method, args.flux, lam, sig_arg, eps, args.crop, args.iter,
                     args.bt_s)
        path = os.path.join(out_dir, tag + ".pth")
        if os.path.exists(path) and not args.overwrite:
            print(f"  {tag}: done already ({path})")
            continue
        df, tau, L_safe = pu.fidelity_and_steps(fidelity, meas, lam, prior, cfg)
        t0 = time.time()
        hist_on = not args.no_history
        recs, L_hist = pu.solve(None, meas["y"], meas["physics"], prior, df, tau, sigma, lam,
                                [args.iter], m["algo"], cfg, L_safe,
                                log_cost=hist_on, metrics=hist_on)
        x, mets = recs[args.iter]
        # per-iteration histories (no ground truth: no PSNR): F(z_k) in float64, ||z_{k+1} -
        # z_k||, ||z_k||, criticality ||z - P_+(z - grad F(z))|| (prox-gradient map for
        # non-smooth priors), deepinv residual; plotted by compare_results.plot_real_convergence
        history = None
        if mets:
            history = {k: np.asarray(mets[k][0], dtype=np.float64)
                       for k in ("cost", "step_norm", "iterate_norm", "criticality", "residual")
                       if k in mets and len(mets[k]) and len(mets[k][0])}
        dt = time.time() - t0
        n_run = len(L_hist) if L_hist else None
        finite = bool(torch.isfinite(x).all())
        L_end = f", final step {1 / L_hist[-1]:.3g} (cap {1 / cfg.bt_s:g})" if L_hist else ""
        sig_txt = f"sigma {sig_arg:g}, " if sig_arg is not None else ""
        print(f"  {sig_txt}lambda {lam:g}: {dt / 60:.1f} min, {n_run} iterations{L_end}, finite {finite}, "
              f"in-focus max {float(x[0, REG_PLANE].max()):.4g}, "
              f"out-of-focus max {float(x[0, 1 - REG_PLANE].max()):.4g}")
        torch.save({"x": x, "data": args.data, "method": args.method, "algo": m["algo"],
                    "fidelity": fidelity, "lambda": lam, "sigma": sig_arg, "flux": args.flux,
                    "tv_eps": eps, "crop": args.crop, "max_iter": args.iter,
                    "iterations_run": n_run, "L_history": L_hist, "stepsize0": tau,
                    "bt_s": cfg.bt_s, "stop_tol": cfg.stop_tol, "history": history,
                    "px_nm": px_nm, "time_s": dt}, path)
        figure(counts, x, path.replace(".pth", ".png"),
               f"{args.data}  {args.method}  {sig_txt}lambda {lam:g}  flux {args.flux:g}", px_nm)
        print(f"    saved {path} (+ .png)")


if __name__ == "__main__":
    main()

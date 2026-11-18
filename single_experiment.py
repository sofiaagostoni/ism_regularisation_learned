#%%
"""
One reconstruction with parameters you choose (no search), saved next to the
evaluate_methods.py results so that compare_results.py also considers it.

The method name picks everything except the parameters: prior, algorithm, fidelity,
step scales, backtracking, early stopping (STOP_TOL), N_ITER, exactly as in
evaluate_methods.METHODS. You give the image, the flux and the parameters:

    WCRR, WCRR_ISM (+ _L2, _MD)     --sigma  --lam
    WCHS_free, WCHS_free_smlm (+ _L2)  --sigma  --lam   (learned-hs, prior_config.WCHSPrior)
    TV_smooth, TV, L1 (+ _L2, _MD)  --lam
    TV_lhs, HS (+ _L2)              --lam   (learned-hs TV / Hessian-Schatten, prior_config.LearnedHSAnalysisPrior)
    DRUNet, DRUNet_ft, DRUNet_scratch  --sigma  --step   (--iter: default = DRUNET_ITERS,
                                                         the best count is kept)

    CUDA_VISIBLE_DEVICES=1 python single_experiment.py --image tub --flux 20 --method WCRR --sigma 25 --lam 8
    python single_experiment.py --image tub --flux 5 --method L1 --lam 5
    python single_experiment.py --image tub --flux 20 --method WCRR --sigma 20 25 30 --lam 8   # small grid: best kept
    python single_experiment.py ... --plan          # print the settings and the paths only
    python single_experiment.py ... --no-plot       # no figure
    python single_experiment.py ... --stacked       # the 25 detectors one by one (as the evals,
                                                    # default: batched, same result, faster)

3D data (IMAGES_3D: tub_3D, tub_nuc_3D; 2 axial planes): chosen by the image name. The
regularizer acts on plane 1 only (REG_PLANE, in focus; plane 0 is out of focus) and every
metric (PSNR / SSIM / MicroSSIM, search score, per-iteration PSNR) is computed on plane 1;
the figure shows plane 1. --params-from
IMAGE takes the parameters of the 2D adaptive search of the same method and flux on IMAGE
(sigma / lambda; DRUNet: also step and iteration count) instead of --sigma / --lam / ...:

    python single_experiment.py --image tub_3D --flux 20 --method WCRR --params-from tub
    for m in WCRR WCRR_L2 TV_smooth TV L1; do python single_experiment.py --image tub_3D --flux 20 --method $m --params-from tub --no-progress; done

Saved in SAVE_DIR (= evaluate_methods.SAVE_DIR) exactly like a grid-search run:
    SAVE_DIR/single_<hash>/best_rec_<tag>.pth   (one folder per method settings + parameters)
    SAVE_DIR/single_<hash>/<tag>.png             figure, path printed at the end (also when
                                                 the run was already done: just re-plotted)
    SAVE_DIR/summary_all.csv                     one more row
Figure: ground truth, sum of the 25 detectors, this reconstruction, the adaptive-search one
(if done), |error|; PSNR, relative step (with the stop tolerance) and step size per
iteration, this run vs the adaptive search.
The "single_" prefix of the folder marks these runs: compare_results.load / load_merged
keep them by default (include_single=True), and best_per_method then takes the best
PSNR among the adaptive search and your single runs of the same method.
Same parameters twice: skipped (the file exists); --overwrite reruns.
"""
import argparse
import os
from dataclasses import replace

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")                 # no screen (server, nohup): only the png
import matplotlib.pyplot as plt
import numpy as np
import torch

import evaluate_methods as em
from opt_functions.data_preparation import load_data, make_measurements
from opt_functions.Solver_functions import pnp_utils
from opt_functions.Solver_functions.prior_config import REG_PLANE
from opt_functions.Solver_functions.pnp_utils import (PRIORS_WITHOUT_SIGMA, experiment_paths,
                                                      run_experiment, search_plan, searched_axes)

RUN_NAME = "single"       # prefix of the run folders; compare_results.SINGLE_PREFIX


IMAGES_3D = ["tub_3D", "tub_nuc_3D"]       # 2 axial planes, in Data/Simul_data


def method_config(method, seed, save_dir, batched=True, is_3d=False):
    """(cfg, algo, fidelity) of a method of evaluate_methods.METHODS, without the search.
    is_3d: 2 axial planes (Nz = 2 is in the run_id: own run folders)."""
    m = em.METHODS[method]
    cfg = replace(m["cfg"], search="grid", run_name=RUN_NAME, seed=seed, save_dir=save_dir,
                  batched_physics=batched, is_3d=is_3d, Nz=2 if is_3d else 1)
    return cfg, m["algo"], m.get("fidelity", em.FIDELITY)


def params_from_2d(method, image_2d, flux, seed, cfg, algo):
    """{--sigma, --lam, --step, --iter values} of the 2D adaptive search of `method` at
    this flux on image_2d, for the axes the method searches (explicit priors keep their
    default iterations: N_ITER with early stopping), + that result."""
    res = adaptive_best(method, image_2d, flux, seed)
    if res is None:
        raise SystemExit(f"--params-from {image_2d}: no 2D adaptive result of {method} at flux "
                         f"{flux} yet (run evaluate_methods.py first)")
    searched, _ = searched_axes(cfg, algo)
    given = {"sigma": None, "lambda": None, "step": None, "iter": None}
    if "sigma" in searched:
        given["sigma"] = [float(res["sigma"])]
    if "lambda" in searched:
        given["lambda"] = [float(res["lambda"])]
    if "step" in searched and res.get("step_scale") is not None:
        given["step"] = [float(res["step_scale"])]
    if not getattr(cfg, "stop_tol", None) and res.get("iter") is not None:   # PnP: its count
        given["iter"] = [int(res["iter"])]
    return given, res


def check_params(method, cfg, algo, given):
    """Every parameter the method needs is given; warn about the ones it does not use."""
    searched, fixed = searched_axes(cfg, algo)
    flags = {"sigma": "--sigma", "lambda": "--lam", "step": "--step", "iter": "--iter"}
    defaults = {"iter": cfg.iter_grid, "step": cfg.step_grid}
    missing = [flags[a] for a in searched
               if given[a] is None and defaults.get(a) is None]
    if missing:
        raise SystemExit(f"{method} needs {' '.join(missing)}")
    for a in fixed:
        if given[a] is not None:
            print(f"warning: {flags[a]} ignored ({fixed[a]})")


def adaptive_best(method, image, flux, seed):
    """Full result (.pth) of the adaptive search of the same experiment, None if not done."""
    m = em.METHODS[method]
    cfg = replace(m["cfg"], seed=seed)
    *_, path = experiment_paths(cfg, image, flux, em.LOSS, m.get("fidelity", em.FIDELITY), m["algo"])
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def to_img(x):
    """(B, C, H, W) -> (H, W) numpy, divided by its max (as for the metrics).
    3D (C = 2): plane REG_PLANE, the in-focus one."""
    img = torch.as_tensor(x).detach().float().cpu()
    if img.ndim == 4 and img.shape[1] > 1:
        img = img[:, REG_PLANE]
    while img.ndim > 2:
        img = img[0]
    return (img / img.max().clamp_min(1e-12)).numpy()


def histories(res):
    """Per-iteration PSNR, relative step ||x+ - x|| / ||x|| and step size of a result."""
    m = res.get("metrics") or {}
    h = {}
    if "psnr" in m:
        h["psnr"] = np.asarray(m["psnr"][0], float)
    if "step_norm" in m and "iterate_norm" in m:
        s, n = np.asarray(m["step_norm"][0], float), np.asarray(m["iterate_norm"][0], float)
        h["rel"] = s / np.r_[n[0], n[:-1]]
    if res.get("L_history"):
        h["tau"] = 1.0 / np.asarray(res["L_history"], float)
    return h


def plot_result(res, ref, sum_y, title, path, show=False):
    """Images (row 1) and convergence curves (row 2) of a single run vs the adaptive search."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), layout="constrained")
    gt = to_img(res["x_gt"])
    imgs = [("ground truth", gt), ("sum of the 25 detectors", to_img(sum_y)),
            (f"single run\nPSNR {res['psnr']:.2f}  SSIM {res['ssim']:.4f}", res["x_rec"]),
            (f"adaptive search\nPSNR {ref['psnr']:.2f}  SSIM {ref['ssim']:.4f}" if ref else
             "adaptive search: not done yet", ref["x_rec"] if ref else None)]
    for a, (t, im) in zip(axes[0], imgs):
        a.set_title(t, fontsize=10)
        a.axis("off")
        if im is not None:
            a.imshow(to_img(im) if not isinstance(im, np.ndarray) else im, cmap="hot", vmin=0, vmax=1)

    runs = {"single run": (res, "C3")}
    if ref:
        runs["adaptive search"] = (ref, "C0")
    ax_psnr, ax_rel, ax_tau, ax_err = axes[1]
    stop_tol = (res.get("config") or {}).get("stop_tol")
    for label, (r, color) in runs.items():
        h = histories(r)
        p = (f"lambda={r['lambda']:.3g}" if r.get("prior") in PRIORS_WITHOUT_SIGMA
             else f"sigma={r['sigma']:.3g} lambda={r['lambda']:.3g}")
        kw = dict(color=color, lw=1.2, label=f"{label} ({p}, {r.get('iter_run')} it)")
        if "psnr" in h:
            ax_psnr.plot(h["psnr"], **kw)
        if "rel" in h:
            ax_rel.plot(np.arange(1, len(h["rel"]) + 1), h["rel"], **kw)
        if "tau" in h:
            ax_tau.plot(np.arange(1, len(h["tau"]) + 1), h["tau"], **kw)
    if stop_tol:
        ax_rel.axhline(stop_tol, color="k", ls="--", lw=1, label=f"stop tol {stop_tol:g}")
    for a, t in [(ax_psnr, "PSNR per iteration (clamped to [0, 1], not / max)"),
                 (ax_rel, r"$\|x_{k+1} - x_k\| / \|x_k\|$"), (ax_tau, r"step size $\tau_k$")]:
        a.set_title(t, fontsize=10)
        a.set_xlabel("iteration k")
        a.grid(alpha=0.3, which="both")
    ax_rel.set_yscale("log")
    ax_tau.set_yscale("log")
    ax_psnr.legend(fontsize=8)
    ax_rel.legend(fontsize=8)

    err = np.abs(to_img(res["x_rec"]) - gt)
    h = ax_err.imshow(err, cmap="viridis", vmin=0, vmax=max(float(np.percentile(err, 99.5)), 1e-6))
    ax_err.set_title("|single run - ground truth| (images / max)", fontsize=10)
    ax_err.axis("off")
    fig.colorbar(h, ax=ax_err, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def fmt(r):
    def f(v, spec):
        return "-" if v is None else format(v, spec)
    return (f"PSNR {f(r['psnr'], '.2f')}  SSIM {f(r['ssim'], '.4f')}  sigma={f(r['sigma'], '.4g')}  "
            f"lambda={f(r['lambda'], '.4g')}  step x{f(r['step_scale'], '.3g')}  "
            f"iter={r['iter']} (ran {r['iter_run']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", required=True, choices=em.IMAGES + IMAGES_3D)
    parser.add_argument("--flux", required=True, type=float)
    parser.add_argument("--method", required=True, choices=list(em.METHODS))
    parser.add_argument("--sigma", nargs="+", type=float, help="regularizer / denoiser strength")
    parser.add_argument("--lam", nargs="+", type=float, help="regularization weight lambda")
    parser.add_argument("--step", nargs="+", type=float, help="step multiplier (DRUNet only)")
    parser.add_argument("--iter", nargs="+", type=int,
                        help="iterations (default: the method's, N_ITER = max with early stopping)")
    parser.add_argument("--params-from", metavar="IMAGE", choices=em.IMAGES,
                        help="use the parameters of the 2D adaptive search of this image "
                             "(same method and flux), e.g. for 3D data")
    parser.add_argument("--seed", type=int, default=0, help="noise realization")
    parser.add_argument("--save-dir", default=em.SAVE_DIR)
    parser.add_argument("--overwrite", action="store_true", help="rerun if already done")
    parser.add_argument("--plan", action="store_true", help="print settings and paths only")
    parser.add_argument("--no-plot", action="store_true", help="no figure")
    parser.add_argument("--show", action="store_true",
                        help="also open the figure in a window (needs a display)")
    parser.add_argument("--stacked", action="store_true",
                        help="25 separate detector operators instead of the batched one "
                             "(same result up to rounding, slower)")
    parser.add_argument("--no-progress", action="store_true",
                        help="no per-iteration progress bar (iteration, relative step)")
    args = parser.parse_args()
    pnp_utils.PROGRESS_BAR = not args.no_progress
    flux = int(args.flux) if args.flux == int(args.flux) else args.flux

    is_3d = args.image in IMAGES_3D
    cfg, algo, fidelity = method_config(args.method, args.seed, args.save_dir,
                                        batched=not args.stacked, is_3d=is_3d)
    given = {"sigma": args.sigma, "lambda": args.lam, "step": args.step, "iter": args.iter}
    src2d = None
    if args.params_from:
        if any(given[k] is not None for k in ("sigma", "lambda", "step")):
            print("warning: --params-from given: --sigma / --lam / --step ignored")
        given, src2d = params_from_2d(args.method, args.params_from, flux, args.seed, cfg, algo)
        if args.iter is not None:              # --iter still applies (e.g. a quick test)
            given["iter"] = args.iter
        print(f"parameters of the 2D search on {args.params_from}: " + "  ".join(
            f"{k}={v[0]:.4g}" for k, v in given.items() if v is not None))
    check_params(args.method, cfg, algo, given)
    sigmas, lambdas, steps, iters = given["sigma"], given["lambda"], given["step"], given["iter"]
    run_cfg, grids, notes, tag, out_path = experiment_paths(
        cfg, args.image, flux, em.LOSS, fidelity, algo,
        sigmas=sigmas, iters=iters, lambdas=lambdas, steps=steps)

    print(f"[{args.method}] {args.image}{f' (3D: regularizer and metrics on plane {REG_PLANE}, in focus)' if is_3d else ''}, "
          f"flux {flux}, seed {args.seed}, fidelity {fidelity}, "
          f"{search_plan(cfg, algo, grids, notes)}")
    if cfg.stop_tol is not None:
        print(f"  early stopping: relative step < {cfg.stop_tol:g} ({cfg.stop_patience} times in a row)")
    print(f"-> {out_path}" + ("  (exists: skipped, --overwrite to rerun)"
                              if os.path.exists(out_path) and not args.overwrite else ""))
    ref = None if is_3d else adaptive_best(args.method, args.image, flux, args.seed)
    if args.plan:
        if ref:
            print(f"adaptive search: {fmt(ref)}")
        if src2d:
            print(f"2D search ({args.params_from}): {fmt(src2d)}")
        return

    # simulated here (same seed, same as run_experiment would do) to plot the measurement
    x = load_data(args.image, cfg)
    meas = make_measurements(x, flux, cfg)
    res = run_experiment(cfg, args.image, flux, em.LOSS, fidelity, algo=algo,
                         sigmas=sigmas, iters=iters, lambdas=lambdas, steps=steps,
                         x=x, meas=meas, overwrite=args.overwrite, verbose=True)
    if res is None:                       # already done: show the saved result
        res = torch.load(out_path, map_location="cpu", weights_only=False)
    print(f"\nsingle run     : {fmt(res)}")
    if ref:
        d = res["psnr"] - ref["psnr"]
        verdict = "same as" if abs(d) < 0.005 else ("better than" if d > 0 else "worse than")
        print(f"adaptive search: {fmt(ref)}\n  -> single run {verdict} the search ({d:+.2f} dB)")
    elif not is_3d:
        print("adaptive search: not done yet for this (image, flux, method)")
    if src2d:
        print(f"2D search ({args.params_from}, parameters used): {fmt(src2d)}"
              + ("  [2D image: not comparable pixel by pixel]" if is_3d else ""))

    if not args.no_plot and res["x_rec"] is not None:
        png = os.path.join(os.path.dirname(out_path), f"{tag}.png")
        plot_result(res, ref, meas["sum_y"], f"{args.method} | {args.image} | flux {flux:g} | "
                    f"seed {args.seed} | fidelity {fidelity}"
                    + (f" | plane {REG_PLANE} (in focus)" if is_3d else "")
                    + (f" | parameters of the 2D search on {args.params_from}" if src2d else ""),
                    png, show=args.show)
        print(f"\nfigure: {png}")


if __name__ == "__main__":
    main()

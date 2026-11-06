#%%
"""
Evaluate the convergent explicit-prior methods (WCRR, WCRR_ISM, smoothed TV) on a set of images.

For every (image, flux, seed) the measurement is simulated once and shared by all
methods; each method then gets its own adaptive parameter search on that image
(per-image tuning with the ground truth = oracle parameters). Iterations are not tuned:
every method runs N_ITER iterations, so the reconstruction is the (approximate) minimizer.

    python evaluate_methods.py                 # everything, sequentially
    python evaluate_methods.py --shard 0/2     # half of the jobs (e.g. one per GPU)
    python evaluate_methods.py --plan          # print the jobs and searches, run nothing
    python evaluate_methods.py --methods TV_smooth  # only these methods (e.g. a new one while
                                               # the others are still running)

Safe to restart: finished (image, flux, seed, method) are loaded from disk, not rerun,
and an interrupted search resumes from its last evaluation.
Results:
    SAVE_DIR/<run_id>/best_rec_<tag>.pth       full result, incl. every point evaluated
    SAVE_DIR/evaluation.csv                    one row per (image, flux, seed, method)
Analysis: analyze_evaluation.py
"""
import argparse
import csv
import gc
import itertools
import os
import time
import traceback
from dataclasses import replace

import torch
from tqdm.auto import tqdm

from opt_functions.data_preparation import load_data, make_measurements
from opt_functions.Solver_functions.prior_config import get_denoiser
from opt_functions.Solver_functions.pnp_utils import (Config, experiment_paths, run_experiment,
                                                      search_plan)

# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------
SAVE_DIR = "Results/evaluation_stop"   # with early stopping (STOP_TOL); before: "Results/evaluation"
FIDELITY = "KL"                 # default; a method can override it with fidelity=...
N_ITER = 5000                   # maximum number of iterations
# Early stopping: ||x_{k+1} - x_k|| / ||x_k|| < STOP_TOL for STOP_PATIENCE consecutive
# iterations (None: always N_ITER). Changes the run_id: results go to new run folders.
STOP_TOL = 1e-6
STOP_PATIENCE = 10

# 2D, 256 x 256 ground truths in Data/Simul_data
# (Castle_gray is a natural image: add it only as an out-of-distribution test)
IMAGES = ["circle_phantom", "tub", "tub_level", "tub_balls", "balls", "mixture_256"]
FLUXES = [20]
SEEDS = [0]                     # noise realizations; add more to average them out

base = Config(is_3d=False, Nz=1, save_dir=SAVE_DIR, init_choice="adjoint_y",
              backtracking="custom", iter_grid=torch.tensor([N_ITER]), search="adaptive",
              search_max_expand=3.0, stop_tol=STOP_TOL, stop_patience=STOP_PATIENCE)

# Adaptive search (see adaptive_search.py), ranges in log10: a coarse scan of the
# starting range, extended by up to search_max_expand decades on each side when the
# best value is on its border, then refined to search_tol decades.
#   1 parameter  (TV, L1): golden section,           ~12-16 runs
#   2 parameters (WCRR)  : 3 x 3 pattern search,     ~45-60 runs
METHODS = {
    "WCRR": dict(cfg=replace(base, prior_name="WCRR", kl_step_scale=1e-2, l2_step_scale=1e-1,
                             sigma_range=(-1, 2.5), lam_range=(-2, 3),
                             search_points=5, search_tol=0.1),
                 algo="PROJ_GD"),
    "WCRR_ISM": dict(cfg=replace(base, prior_name="WCRR_ISM", kl_step_scale=1e-2, l2_step_scale=1e-1,
                             sigma_range=(-1, 2.5), lam_range=(-2, 3),
                             search_points=5, search_tol=0.1),
                 algo="PROJ_GD"),
    "TV_smooth": dict(cfg=replace(base, prior_name="TV_smooth", tv_eps=1e-2,
                                  kl_step_scale=1e-2, l2_step_scale=1e-1,
                                  lam_range=(-4, 3), search_points=8, search_tol=0.05),
                      algo="PROJ_GD"),
    # >>>>>>>>>> MD ADDED >>>>>>>>>>
    # Mirror descent (Burg entropy, backtracking) on the KL WITHOUT background ('KL0').
    # The measurements are the same as above (simulated with background): only the
    # model has no background. Starting ranges copied from the PROJ_GD / PROX_GD methods.
    "WCRR_MD": dict(cfg=replace(base, prior_name="WCRR", kl_step_scale=1e-2,
                                sigma_range=(-1, 2.5), lam_range=(-2, 3),
                                search_points=5, search_tol=0.1),
                    algo="MD_BURG", fidelity="KL0"),
    "WCRR_ISM_MD": dict(cfg=replace(base, prior_name="WCRR_ISM", kl_step_scale=1e-2,
                                    sigma_range=(-1, 2.5), lam_range=(-2, 3),
                                    search_points=5, search_tol=0.1),
                        algo="MD_BURG", fidelity="KL0"),
    "TV_smooth_MD": dict(cfg=replace(base, prior_name="TV_smooth", tv_eps=1e-2,
                                     kl_step_scale=1e-2,
                                     lam_range=(-4, 3), search_points=8, search_tol=0.05),
                         algo="MD_BURG", fidelity="KL0"),
    # <<<<<<<<<< MD ADDED <<<<<<<<<<
    # >>>>>>>>>> L2 ADDED >>>>>>>>>>
    # Same priors with the L2 fidelity 0.5 ||Ax - y||^2 (y in intensity units, gain 1/flux).
    # Its curvature at x_gt is ~1e3-1e5 smaller than the KL one at flux 20, so the optimal
    # lambda is ~3-4 decades lower: lam_range shifted by -3.5 decades w.r.t. the KL methods
    # (the adaptive search can still expand by search_max_expand). sigma is in image units,
    # independent of the fidelity: same range.
    "WCRR_L2": dict(cfg=replace(base, prior_name="WCRR", l2_step_scale=1e-1,
                                sigma_range=(-1, 2.5), lam_range=(-5.5, -0.5),
                                search_points=5, search_tol=0.1),
                    algo="PROJ_GD", fidelity="L2"),
    "WCRR_ISM_L2": dict(cfg=replace(base, prior_name="WCRR_ISM", l2_step_scale=1e-1,
                                    sigma_range=(-1, 2.5), lam_range=(-5.5, -0.5),
                                    search_points=5, search_tol=0.1),
                        algo="PROJ_GD", fidelity="L2"),
    "TV_smooth_L2": dict(cfg=replace(base, prior_name="TV_smooth", tv_eps=1e-2, l2_step_scale=1e-1,
                                     lam_range=(-7.5, -0.5), search_points=8, search_tol=0.05),
                         algo="PROJ_GD", fidelity="L2"),
    # <<<<<<<<<< L2 ADDED <<<<<<<<<<
}
LOSS = "L2"                     # unused by these priors, only part of the call signature

CSV_FIELDS = ["image", "flux", "seed", "method", "fidelity", "algo", "lambda", "sigma",
              "iter", "iter_run", "lambda_on_edge", "sigma_on_edge", "n_runs", "final_residual",
              "psnr", "ssim", "microssim", "run_id", "path"]


# ---------------------------------------------------------------------------

def jobs():
    """(image, flux, seed) groups: the measurement is shared by all methods of a group."""
    return list(itertools.product(IMAGES, FLUXES, SEEDS))


def on_edge(value, grid):
    """True if value is the first or last point of a searched grid (len > 1)."""
    g = torch.as_tensor(grid).flatten()
    return bool(len(g) > 1 and (torch.isclose(g[0], torch.tensor(value)) or
                                torch.isclose(g[-1], torch.tensor(value))))


def csv_row(res, image, flux, seed, method, path):
    residual = float("nan")
    m = res.get("metrics") or {}
    if m.get("residual") and len(m["residual"][0]):
        residual = float(m["residual"][0][-1])
    lam, sigma = res["lambda"], res["sigma"]
    hit = res.get("hit_bound") or {}           # adaptive search: at the expansion limit
    return {
        "image": image, "flux": flux, "seed": seed, "method": method,
        "fidelity": res["fidelity"], "algo": res["algo"], "lambda": lam, "sigma": sigma,
        "iter": res["iter"], "iter_run": res.get("iter_run", res["iter"]),
        "lambda_on_edge": hit.get("lambda", lam is not None and on_edge(lam, res["lambda_grid"])),
        "sigma_on_edge": hit.get("sigma", sigma is not None and on_edge(sigma, res["sigma_grid"])),
        "n_runs": (len(res["evaluations"]) if res.get("evaluations") is not None
                   else int(torch.isfinite(res["psnr_map"]).sum())),
        "final_residual": residual,
        "psnr": res["psnr"], "ssim": res["ssim"], "microssim": res["microssim"],
        "run_id": res["run_id"], "path": path,
    }


def write_csv(rows, suffix=""):
    """Rewrite evaluation.csv from all rows of this shard (merged by analyze_evaluation.py)."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"evaluation{suffix}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def log_failure(msg):
    tqdm.write(msg)
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, "failed.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def print_plan(my_jobs, methods):
    for method, m in methods.items():
        print(f"[{method}] fidelity {m.get('fidelity', FIDELITY)}, "   # MD ADDED: fidelity
              + search_plan(m["cfg"], m["algo"]))
    stop = (f", stop at relative step < {STOP_TOL:g} ({STOP_PATIENCE} times in a row)"
            if STOP_TOL is not None else "")
    print(f"\n{len(my_jobs)} (image, flux, seed) groups x {len(methods)} methods, "
          f"at most {N_ITER} iterations per run{stop}")


def main(shard, n_shards, plan_only, only=None):
    """only: list of method names to run (own csv file); None = all of METHODS."""
    methods = METHODS if only is None else {k: METHODS[k] for k in only}
    suffix = "" if n_shards == 1 else f"_shard{shard}of{n_shards}"
    if only is not None:
        suffix += "_" + "_".join(only)
    my_jobs = jobs()[shard::n_shards]
    print_plan(my_jobs, methods)
    if plan_only:
        return

    rows = []
    denoisers = {}                       # one regularizer per method, loaded once
    for image, flux, seed in tqdm(my_jobs, desc="images x flux x seed"):
        x = meas = None
        for method, m in methods.items():
            cfg = replace(m["cfg"], seed=seed)
            fidelity = m.get("fidelity", FIDELITY)          # MD ADDED: per-method fidelity
            *_, out_path = experiment_paths(cfg, image, flux, LOSS, fidelity, m["algo"])  # MD ADDED: fidelity
            try:
                if os.path.exists(out_path):
                    res = torch.load(out_path, map_location="cpu", weights_only=False)
                else:
                    if x is None:        # simulate once, shared by every method
                        x = load_data(image, cfg)
                        meas = make_measurements(x, flux, cfg)
                    if method not in denoisers:
                        denoisers[method] = get_denoiser(LOSS, cfg)
                    res = run_experiment(cfg, image, flux, LOSS, fidelity, algo=m["algo"],  # MD ADDED: fidelity
                                         x=x, meas=meas, denoiser=denoisers[method],
                                         verbose=False)
                rows.append(csv_row(res, image, flux, seed, method, out_path))
                write_csv(rows, suffix)
            except Exception:
                log_failure(f"FAILED {image} flux={flux} seed={seed} {method}\n"
                            f"{traceback.format_exc()}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    parser.add_argument("--plan", action="store_true", help="print the jobs and grids only")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS),
                        help="run only these methods (results in their own csv)")
    args = parser.parse_args()
    shard, n_shards = map(int, args.shard.split("/"))
    main(shard, n_shards, args.plan, args.methods)

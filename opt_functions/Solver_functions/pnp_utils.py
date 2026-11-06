"""
Grid search for PnP / RED / WCRR reconstructions.

How the prior is used:
    gradient algorithms ('MD', 'GD')     -> RED prior  (grad = x - D(x))
    prox algorithms ('PGD', 'HQS', ...)  -> PnP prior  (prox = D(x))
    explicit priors                      -> R(x) used directly:
        'weakly_convex' (WCRR)  smooth      -> 'PROJ_GD'
        'TV_smooth'             smooth      -> 'PROJ_GD'   (smoothing eps = cfg.tv_eps)
        'L1'                    non-smooth  -> 'PROX_GD'

Grids searched (see searched_axes); a grid that is not searched is fixed:
    WCRR, backtracking on            : sigma, lambda, iter         (step fixed)
    WCRR, backtracking 'none'        : sigma, lambda, step, iter
    TV_smooth / L1, backtracking on  : lambda, iter                (sigma, step fixed)
    TV_smooth / L1, backtracking none: lambda, step, iter
    DRUNet / TV, PnP                 : sigma, step, iter           (lambda fixed)
    DRUNet / TV, RED                 : sigma, step, iter, lambda
print_search_plan(cfg) shows what a run will do before starting it.

Results (a different grid never overwrites older results):
    save_dir/summary_all.csv           one row per experiment, all runs
    save_dir/<run_id>/config.json      grids and settings of this run
                     /log.txt
                     /summary.csv
                     /best_rec_<tag>.pth
                     /partial_<tag>.pth  checkpoint while a search is running
"""
import csv
import hashlib
import itertools
import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import Optional

import torch
import deepinv as dinv
from tqdm.auto import tqdm

from opt_functions.data_preparation import load_data, make_measurements
from .prior_config import GRADIENT_ALGOS, get_denoiser, make_prior
from .iterators_config import SnapshotIterator, build_model, get_iteration
from .adaptive_search import coarse_to_fine

try:
    from microssim import micro_structural_similarity   # pip install microssim
except ImportError:
    micro_structural_similarity = None

# Priors with an explicit cost R(x): lambda is searched, backtracking can be used
EXPLICIT_PRIORS = {"WCRR", "WCRR_ISM", "WCRR_softplus_ISM", "TV_smooth", "L1",
                   "WCRR_allpass", "WCRR_ISM_allpass", "WCRR_softplus_ISM_allpass"}
# Priors that do not use sigma (g_param): the sigma grid is not searched
PRIORS_WITHOUT_SIGMA = {"TV_smooth", "L1"}
# Priors whose weights depend on the training loss (one checkpoint per loss)
PRIORS_USING_LOSS = {"Drunet_finetune", 'Drunet_fromscratch'}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Config:
    device: torch.device = field(default_factory=default_device)

    # simulation
    Nx: int = 256
    Nz: int = 1
    pxsizex: float = 40
    is_3d: bool = False

    # prior: 'TV', 'Drunet', 'Drunet_finetune', 'WCRR', 'TV_smooth', 'L1'
    prior_name: str = "Drunet_finetune"
    tv_eps: float = 1e-2                       # smoothing of 'TV_smooth' (Lip grad R = 8/eps)
    ckpt_dir: str = "training_drunet_newloss"

    # paths
    data_dir: str = "Data/Simul_data"
    save_dir: str = "Results/results_gridsearch"
    run_name: Optional[str] = None            # readable prefix of the run folder

    # grids (only the searched ones are used, and those must be set)
    sigma_grid: Optional[torch.Tensor] = None  # denoiser / regularizer strength
    iter_grid: Optional[torch.Tensor] = None   # iteration counts
    lam_grid: Optional[torch.Tensor] = None    # regularization weight
    step_grid: Optional[torch.Tensor] = None   # multipliers of the base step

    # search: 'grid'     every point of the grids above
    #         'adaptive' coarse-to-fine in log10 space (adaptive_search.py): the ranges
    #                    below are only the starting scan, extended when the best value
    #                    is on their border; iter_grid must have a single value
    search: str = "grid"
    lam_range: Optional[tuple] = None          # (log10 lo, log10 hi)
    sigma_range: Optional[tuple] = None        # (log10 lo, log10 hi)
    search_points: int = 5                     # coarse scan: points per axis
    search_tol: float = 0.05                   # final resolution (decades)
    search_max_expand: float = 3.0             # decades allowed beyond each range

    # Early stopping (explicit priors, PROJ_GD / PROX_GD / MD_BURG): stop once
    # ||x_{k+1} - x_k|| / ||x_k|| < stop_tol for stop_patience consecutive iterations;
    # iter_grid is then the maximum number of iterations. None: always iter_grid.
    stop_tol: Optional[float] = None
    stop_patience: int = 10

    # solver
    algo: str = "PGD"                          # 'PGD', 'PNP_MD', 'PROJ_GD', 'PROX_GD', 'MD', 'GD', ...
    init_choice: str = "adjoint_y"             # 'adjoint_y', 'random', 'constant'
    # base step: tau = 1 / (L_f * scale [+ lam * L_R for WCRR])
    kl_step_scale: float = 5e3
    l2_step_scale: float = 1.0

    # step-size strategy for explicit priors (WCRR, TV_smooth, L1):
    #   'custom'  adaptive backtracking in ProxGDIteration
    #   'deepinv' deepinv's built-in backtracking
    #   'none'    fixed step (step_grid is searched)
    backtracking: str = "custom"
    bt_s: float = 1.0       # largest step allowed = 1 / bt_s
    bt_eta: float = 2.0     # L <- eta * L when the decrease test fails
    bt_delta: float = 0.9   # each iteration first tries step tau_k / delta
    bt_max: int = 50        # max backtracking loops per iteration

    seed: int = 0

    def __post_init__(self):
        if self.backtracking not in {"custom", "deepinv", "none"}:
            raise ValueError(f"Unknown backtracking mode: {self.backtracking}")
        if self.search not in {"grid", "adaptive"}:
            raise ValueError(f"Unknown search mode: {self.search}")

    def grid_signature(self):
        """Everything that changes the results of a grid search."""
        def vals(g, cast):
            return None if g is None else [cast(v) for v in torch.as_tensor(g).flatten()]

        def rnd(v):
            return round(float(v), 6)

        sig = {
            "prior_name": self.prior_name, "tv_eps": self.tv_eps,
            "sigma_grid": vals(self.sigma_grid, rnd), "iter_grid": vals(self.iter_grid, int),
            "lam_grid": vals(self.lam_grid, rnd), "step_grid": vals(self.step_grid, rnd),
            "kl_step_scale": self.kl_step_scale, "l2_step_scale": self.l2_step_scale,
            "init_choice": self.init_choice, "backtracking": self.backtracking,
            "bt": [self.bt_s, self.bt_eta, self.bt_delta, self.bt_max],
            "seed": self.seed, "Nx": self.Nx, "Nz": self.Nz, "pxsizex": self.pxsizex,
        }
        if self.search != "grid":      # grid runs keep the run_id they always had
            sig.update(search=self.search, lam_range=vals(self.lam_range, rnd),
                       sigma_range=vals(self.sigma_range, rnd), search_points=self.search_points,
                       search_tol=self.search_tol, search_max_expand=self.search_max_expand)
        if self.stop_tol is not None:  # runs without early stopping keep their run_id
            sig.update(stop_tol=self.stop_tol, stop_patience=self.stop_patience)
        return sig

    @property
    def run_id(self):
        sig = json.dumps(self.grid_signature(), sort_keys=True)
        h = hashlib.sha1(sig.encode()).hexdigest()[:8]
        return f"{self.run_name}_{h}" if self.run_name else h

    @property
    def run_dir(self):
        return os.path.join(self.save_dir, self.run_id)

    def init_run_dir(self):
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, "config.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(self.grid_signature(), f, indent=2)


def log(msg, cfg):
    tqdm.write(msg)
    cfg.init_run_dir()
    with open(os.path.join(cfg.run_dir, "log.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


# ---------------------------------------------------------------------------
# Which grids are searched
# ---------------------------------------------------------------------------

GRID_FIELDS = {"sigma": "sigma_grid", "lambda": "lam_grid", "step": "step_grid", "iter": "iter_grid"}
FIXED_VALUES = {"sigma": 1.0, "lambda": 1.0, "step": 1.0}


def searched_axes(cfg, algo):
    """({axis: reason} searched, {axis: reason} fixed) for this prior and algorithm."""
    searched = {"iter": "iteration counts (one run of the largest value)"}
    fixed = {}
    if cfg.prior_name in PRIORS_WITHOUT_SIGMA:
        fixed["sigma"] = f"{cfg.prior_name} does not use sigma"
    else:
        searched["sigma"] = "denoiser / regularizer strength"
    if cfg.prior_name in EXPLICIT_PRIORS:
        searched["lambda"] = "regularization weight"
        if cfg.backtracking == "none":
            searched["step"] = "fixed step size (no backtracking)"
        else:
            fixed["step"] = "backtracking adapts the step"
    else:
        searched["step"] = "multiplier of the base step"
        if algo.upper() in GRADIENT_ALGOS:
            searched["lambda"] = "RED regularization weight"
        else:
            fixed["lambda"] = "PnP: no lambda"
    return searched, fixed


def resolve_grids(cfg, algo, sigmas=None, iters=None, lambdas=None, steps=None):
    """The grids used, as 1D tensors (arguments override cfg), plus warnings."""
    searched, fixed = searched_axes(cfg, algo)
    given = {"sigma": sigmas, "iter": iters, "lambda": lambdas, "step": steps}
    grids, notes = {}, []
    adaptive = cfg.search == "adaptive"
    for axis, field_name in GRID_FIELDS.items():
        g = given[axis] if given[axis] is not None else getattr(cfg, field_name)
        g = None if g is None else torch.as_tensor(g).flatten()
        if adaptive and axis in searched and axis in ("sigma", "lambda"):
            # adaptive search: the "grid" is the starting range (its two ends)
            r = cfg.lam_range if axis == "lambda" else cfg.sigma_range
            if r is None:
                raise ValueError(f"adaptive search of '{axis}': set cfg."
                                 f"{'lam' if axis == 'lambda' else 'sigma'}_range")
            grids[axis] = 10 ** torch.as_tensor(r, dtype=torch.float64).flatten()
        elif axis in searched:
            if g is None or g.numel() == 0:
                raise ValueError(f"{cfg.prior_name} with {algo.upper()} searches '{axis}' "
                                 f"({searched[axis]}): set cfg.{field_name}")
            grids[axis] = g
        else:
            if g is not None and g.numel() > 1:
                notes.append(f"cfg.{field_name} is ignored ({fixed[axis]}), "
                             f"fixed to {FIXED_VALUES[axis]}")
            grids[axis] = torch.tensor([FIXED_VALUES[axis]])
    grids["iter"] = grids["iter"].round().long()
    for axis in ("sigma", "lambda", "step"):
        grids[axis] = grids[axis].float()
    if adaptive and len(grids["iter"]) > 1:
        raise ValueError("adaptive search: iter_grid must have a single value")
    return grids, notes


def search_plan(cfg, algo, grids=None, notes=()):
    """Readable description of what a run will search."""
    if grids is None:
        grids, notes = resolve_grids(cfg, algo)
    searched, fixed = searched_axes(cfg, algo)
    lines = [f"search plan: prior {cfg.prior_name}, algo {algo.upper()}"]
    if cfg.prior_name == "TV_smooth":
        lines.append(f"  tv_eps = {cfg.tv_eps:g}")
    adaptive = cfg.search == "adaptive"
    if adaptive:
        lines[0] += (f", adaptive search (coarse scan {cfg.search_points} points/axis, "
                     f"expansion <= {cfg.search_max_expand:g} decades, "
                     f"resolution {cfg.search_tol:g} decades)")
    for axis in ("sigma", "lambda", "step", "iter"):
        g = grids[axis].tolist()
        if adaptive and axis in searched and axis in ("sigma", "lambda"):
            lines.append(f"  {axis:<6} searched from range [{g[0]:.3g}, {g[-1]:.3g}]  "
                         f"<- {searched[axis]}")
        elif axis in searched:
            shown = g if len(g) <= 7 else g[:3] + ["..."] + g[-2:]
            lines.append(f"  {axis:<6} searched ({len(g)}): {shown}  <- {searched[axis]}")
        else:
            lines.append(f"  {axis:<6} fixed = {g[0]:g}  <- {fixed[axis]}")
    if adaptive:
        lines.append(f"  -> runs of {int(grids['iter'].max())} iterations, as many as needed")
    else:
        n_runs = len(grids["sigma"]) * len(grids["step"]) * len(grids["lambda"])
        lines.append(f"  -> {n_runs} runs of {int(grids['iter'].max())} iterations")
    lines += [f"  WARNING: {n}" for n in notes]
    return "\n".join(lines)


def print_search_plan(cfg, algo=None):
    print(search_plan(cfg, algo or cfg.algo))


# ---------------------------------------------------------------------------
# Step sizes and metrics
# ---------------------------------------------------------------------------

def fidelity_and_steps(fidelity_name, meas, lam, prior, cfg):
    """Data fidelity, base step 1/L, and the safe Lipschitz bound of grad(f + lam R)
    (without the step scale; used as fallback step by the custom backtracking)."""
    name = fidelity_name.upper()
    if name == "KL":
        df, L_f, scale = meas["df_kl"], float(meas["L_kl"]), cfg.kl_step_scale
    # >>>>>>>>>> MD ADDED >>>>>>>>>>
    elif name == "KL0":         # KL without background (mirror descent, 'MD_BURG')
        df, L_f, scale = meas["df_kl0"], float(meas["L_kl0"]), cfg.kl_step_scale
    # <<<<<<<<<< MD ADDED <<<<<<<<<<
    elif name == "L2":
        df, L_f, scale = meas["df_l2"], float(meas["L_l2"]), cfg.l2_step_scale
    else:
        raise ValueError(f"Unknown fidelity: {fidelity_name}")
    L_R = lam * float(prior.get_mu().detach()) if getattr(prior, "explicit_prior", False) else 0.0
    return df, 1.0 / (L_f * scale + L_R), L_f + L_R


psnr_fn = dinv.metric.PSNR()
ssim_fn = dinv.metric.SSIM()


def psnr(x_rec, x_gt):
    """PSNR after scaling the reconstruction to max 1 (as used in the grid search)."""
    return psnr_fn(x_rec / x_rec.max().clamp_min(1e-12), x_gt).item()


def compute_final_metrics(x_gt, x_rec):
    """PSNR / SSIM / MicroSSIM of the best reconstruction."""
    x_gt, x_rec = x_gt.detach().float().cpu(), x_rec.detach().float().cpu()
    x_rec = x_rec / x_rec.max().clamp_min(1e-12)
    out = {"psnr": psnr_fn(x_rec, x_gt).item(), "ssim": ssim_fn(x_rec, x_gt).item(),
           "microssim": float("nan")}
    if micro_structural_similarity is not None:
        try:   # not symmetric: ground truth first
            out["microssim"] = float(micro_structural_similarity(
                x_gt.squeeze().numpy(), x_rec.squeeze().numpy()))
        except Exception as e:
            tqdm.write(f"MicroSSIM failed: {e}")
    return out


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

GRID_AXES = ("sigma", "step", "iter", "lambda")     # axes of psnr_map


def iterations_run(metrics, n):
    """Iterations actually done for a count n (< n if the run stopped early)."""
    if not metrics or not metrics.get("residual"):
        return n
    return min(n, len(metrics["residual"][0]))


def solve(x, y, physics, prior, data_fidelity, tau, sigma, lam, iters, algo, cfg, L_max,
          log_cost=True):
    """Reconstruct with one parameter set. Returns ({n: (x after n iterations, deepinv
    metrics)} for every n in iters, the step-size history of the backtracking or None).
    log_cost: metrics["cost"] = full F = f + lam R in float64 (custom explicit-prior
    iterations, ~45% slower); False: the float32 smooth part of F."""
    explicit = getattr(prior, "explicit_prior", False)
    deepinv_bt = explicit and cfg.backtracking == "deepinv"
    bt_kwargs = {}
    if explicit and cfg.backtracking == "custom":
        bt_kwargs = dict(backtracking=True, s=cfg.bt_s, eta=cfg.bt_eta,
                         delta=cfg.bt_delta, max_bt=cfg.bt_max, L_max=L_max)
    stop = explicit and getattr(cfg, "stop_tol", None) is not None
    if stop:
        bt_kwargs.update(stop_tol=cfg.stop_tol, stop_patience=cfg.stop_patience)

    def run(iteration, n_iter):
        stop_fn = iteration.check_stop if stop and hasattr(iteration, "check_stop") else None
        model = build_model(prior, data_fidelity, iteration, tau, sigma, lam, n_iter,
                            cfg.init_choice, deepinv_bt, stop_fn=stop_fn)
        with torch.no_grad():
            x_out, metrics = model(y, physics, x_gt=x, compute_metrics=True)
        return x_out.detach().cpu(), metrics

    iteration = get_iteration(algo, snap_iters=iters, log_cost=log_cost, **bt_kwargs)
    if isinstance(iteration, SnapshotIterator):
        # one run of max(iters) iterations; the other counts come from the snapshots
        max_it = max(iters)
        x_out, metrics = run(iteration, max_it)
        hist = getattr(iteration, "history", None)     # explicit priors with log_cost
        if metrics is not None and hist and hist["step_norm"]:
            metrics.update({k: [v] for k, v in hist.items()})
        snaps = {**iteration.snapshots, max_it: x_out}
        # stopped early: the counts after the stop get the last iterate
        recs = {n: (snaps.get(n, x_out), metrics) for n in iters}
    else:
        # deepinv built-in iteration: one run per iteration count
        recs = {n: run(iteration, n) for n in iters}
    return recs, getattr(iteration, "L_history", None)


def grid_search(x, y, physics, prior, data_fidelity, base_steps, L_safe, grids, algo, cfg,
                desc="grid search", verbose=False, partial_path=None):
    """
    Search sigma x step x iter x lambda. For lambda index k the step used is
    base_steps[k] * step. Returns (best, psnr_map). If partial_path is given,
    progress is saved after every grid point and resumed on restart.
    """
    sigmas, steps, lambdas = grids["sigma"], grids["step"], grids["lambda"]
    iters = [int(n) for n in grids["iter"]]
    psnr_map = torch.full((len(sigmas), len(steps), len(iters), len(lambdas)), float("nan"))
    best = {"psnr": -float("inf"), "x": None}
    done = set()
    x_cpu = x.detach().float().cpu()

    if partial_path and os.path.exists(partial_path):
        state = torch.load(partial_path, weights_only=False)
        if state["psnr_map"].shape == psnr_map.shape:
            psnr_map, best, done = state["psnr_map"], state["best"], set(map(tuple, state["done"]))
            tqdm.write(f"resuming {desc}: {len(done)} grid points already done")

    combos = list(itertools.product(enumerate(sigmas), enumerate(steps), enumerate(lambdas)))
    for (i, sigma), (t, step), (k, lam) in tqdm(combos, desc=desc, leave=False):
        if (i, t, k) in done:
            continue
        sigma, step, lam = float(sigma), float(step), float(lam)
        tau = base_steps[k] * step
        # recs[n] = (reconstruction after n iterations, deepinv metrics)
        recs, L_hist = solve(x, y, physics, prior, data_fidelity, tau, sigma, lam, iters,
                             algo, cfg, L_safe[k])

        line = []
        for j, n in enumerate(iters):
            x_n, metrics = recs[n]
            if x_n is None or not torch.isfinite(x_n).all():
                line.append(f"{n}:div")
                continue
            p = psnr(x_n, x_cpu)
            psnr_map[i, t, j, k] = p
            line.append(f"{n}:{p:.2f}")
            if p > best["psnr"]:
                best = {"psnr": p, "sigma": sigma, "step_scale": step, "iter": n, "lambda": lam,
                        "iter_run": iterations_run(metrics, n),
                        "stepsize": tau, "x": x_n.clone(),
                        "metrics": None if metrics is None else
                        {key: [v[:n] for v in val] for key, val in metrics.items()},
                        "L_history": L_hist[:n] if L_hist else None}

        if verbose:
            tqdm.write(f"sigma={sigma:.4g}  step x{step:.3g} (tau={tau:.3e})  lambda={lam:.4g}  "
                       f"PSNR by iter: " + "  ".join(line))

        done.add((i, t, k))
        if partial_path:
            torch.save({"psnr_map": psnr_map, "best": best, "done": sorted(done)}, partial_path)

    return best, psnr_map


def adaptive_param_search(x, meas, prior, fidelity_name, n_iter, algo, cfg,
                          desc="adaptive search", verbose=False, partial_path=None):
    """
    Coarse-to-fine search of lambda and/or sigma (adaptive_search.coarse_to_fine) with
    n_iter iterations per point. Returns (best, evaluations, search) where evaluations
    is a list of {"sigma", "lambda", "psnr"} in the order they were computed.
    If partial_path is given, every evaluation is saved and replayed on restart.
    """
    searched, _ = searched_axes(cfg, algo)
    if "step" in searched:
        raise ValueError("adaptive search needs backtracking (the step is not searched)")
    ranges = {"lambda": cfg.lam_range, "sigma": cfg.sigma_range}
    ranges = {a: tuple(float(v) for v in ranges[a]) for a in ("sigma", "lambda") if a in searched}
    for a, r in ranges.items():
        if r is None or len(r) != 2:
            raise ValueError(f"adaptive search of '{a}': set cfg.{a[:3]}_range = (log10 lo, log10 hi)")

    x_cpu = x.detach().float().cpu()
    cache, evaluations = {}, []
    if partial_path and os.path.exists(partial_path):
        state = torch.load(partial_path, weights_only=False)
        cache, evaluations = state["cache"], state["evaluations"]
        tqdm.write(f"resuming {desc}: {len(cache)} points already done")
    bar = tqdm(desc=desc, leave=False, initial=len(cache))
    recs = {}           # point -> reconstruction of this session (resumed points: rerun)

    def reconstruct(point, log_cost=False):
        sigma = 10 ** point["sigma"] if "sigma" in point else FIXED_VALUES["sigma"]
        lam = 10 ** point["lambda"] if "lambda" in point else FIXED_VALUES["lambda"]
        df, tau, L_safe = fidelity_and_steps(fidelity_name, meas, lam, prior, cfg)
        out, L_hist = solve(x, meas["y"], meas["physics"], prior, df, tau, sigma, lam,
                            [n_iter], algo, cfg, L_safe, log_cost=log_cost)
        x_n, metrics = out[n_iter]
        p = (psnr(x_n, x_cpu) if x_n is not None and torch.isfinite(x_n).all()
             else -float("inf"))
        recs[tuple(sorted(point.items()))] = {
            "psnr": p, "sigma": sigma, "step_scale": 1.0, "iter": n_iter, "lambda": lam,
            "iter_run": iterations_run(metrics, n_iter),
            "stepsize": tau, "x": x_n if p > -float("inf") else None, "metrics": metrics,
            "L_history": L_hist[:n_iter] if L_hist else None}
        return p, sigma, lam

    def score(point):
        p, sigma, lam = reconstruct(point)
        evaluations.append({"sigma": sigma, "lambda": lam, "psnr": p})
        if verbose:
            tqdm.write(f"sigma={sigma:.4g}  lambda={lam:.4g}  PSNR={p:.2f}")
        bar.update(1)
        return p

    def save(point, value):
        if partial_path:
            torch.save({"cache": cache, "evaluations": evaluations}, partial_path)

    search = coarse_to_fine(score, ranges, n_points=cfg.search_points, tol=cfg.search_tol,
                            max_expand=cfg.search_max_expand, cache=cache, on_eval=save)
    bar.close()
    # The search runs without the float64 cost history; rerun the chosen point with it
    # (metrics["cost"] = F = f + lam R per iteration, for convergence rates).
    key = tuple(sorted(search["best"].items()))
    reconstruct(search["best"], log_cost=True)
    return recs[key], evaluations, search


# ---------------------------------------------------------------------------
# One experiment
# ---------------------------------------------------------------------------

CSV_FIELDS = ["run_id", "data", "flux", "loss", "fidelity", "prior", "algo", "lambda",
              "sigma", "step_scale", "stepsize", "iter", "psnr", "ssim", "microssim"]


def append_csv(path, result):
    row = {key: result.get("name" if key == "data" else key) for key in CSV_FIELDS}
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def experiment_paths(cfg, name, flux, loss, fidelity_name, algo=None,
                     sigmas=None, iters=None, lambdas=None, steps=None):
    """(run_cfg, grids, notes, tag, out_path) of one experiment.
    out_path is the best_rec_<tag>.pth file written by run_experiment."""
    algo = (algo or cfg.algo).upper()
    loss_tag = loss if cfg.prior_name in PRIORS_USING_LOSS else "none"
    grids, notes = resolve_grids(cfg, algo, sigmas, iters, lambdas, steps)
    tag = f"Fidelity{fidelity_name}_{name}_flux{flux}_{cfg.prior_name}_loss{loss_tag}_{algo}"
    # the grids actually used define the run folder
    run_cfg = replace(cfg, sigma_grid=grids["sigma"], iter_grid=grids["iter"],
                      lam_grid=grids["lambda"], step_grid=grids["step"])
    return run_cfg, grids, notes, tag, os.path.join(run_cfg.run_dir, f"best_rec_{tag}.pth")


def run_experiment(cfg, name, flux, loss, fidelity_name, algo=None,
                   sigmas=None, iters=None, lambdas=None, steps=None,
                   x=None, meas=None, denoiser=None,
                   save=True, overwrite=False, verbose=True):
    """
    Grid search for one (data, flux, loss, fidelity, algo) and save the best result.

    algo                        : None = cfg.algo
    sigmas/iters/lambdas/steps  : override the grids in cfg
    x, meas, denoiser           : pre-computed objects, to avoid reloading in loops
    save                        : write best_rec_<tag>.pth, the csv rows and checkpoints
    overwrite                   : rerun even if best_rec_<tag>.pth already exists
    Returns the result dict, or None if skipped.
    """
    algo = (algo or cfg.algo).upper()
    explicit = cfg.prior_name in EXPLICIT_PRIORS
    loss_tag = loss if cfg.prior_name in PRIORS_USING_LOSS else "none"

    run_cfg, grids, notes, tag, out_path = experiment_paths(
        cfg, name, flux, loss, fidelity_name, algo, sigmas, iters, lambdas, steps)
    partial_path = os.path.join(run_cfg.run_dir, f"partial_{tag}.pth")
    report = (lambda msg: log(msg, run_cfg)) if save else tqdm.write

    if save:
        run_cfg.init_run_dir()
        if os.path.exists(out_path) and not overwrite:
            report(f"skip {tag} (run {run_cfg.run_id})")
            return None
        if overwrite and os.path.exists(partial_path):
            os.remove(partial_path)

    if x is None:
        x = load_data(name, cfg)
    if meas is None:
        meas = make_measurements(x, flux, cfg)
    if denoiser is None:
        denoiser = get_denoiser(loss, cfg)
    prior = make_prior(denoiser, algo, cfg.is_3d)   # 3D: only plane 0 is regularized

    report(f"start {tag} in {run_cfg.run_dir}\n{search_plan(cfg, algo, grids, notes)}")
    t0 = time.time()
    partial = partial_path if save else None
    evaluations, search = None, None
    if cfg.search == "adaptive":
        best, evaluations, search = adaptive_param_search(
            x, meas, prior, fidelity_name, int(grids["iter"][0]), algo, cfg, desc=tag,
            verbose=verbose, partial_path=partial)
        psnr_map = None
        # the "grids" become the range actually explored (best on its border = hit_bound)
        for axis, (lo, hi) in search["explored"].items():
            grids[axis] = 10 ** torch.tensor([lo, hi])
    else:
        steps_info = [fidelity_and_steps(fidelity_name, meas, float(l), prior, cfg)
                      for l in grids["lambda"]]
        data_fidelity = steps_info[0][0]
        base_steps = [s[1] for s in steps_info]
        L_safe = [s[2] for s in steps_info]
        best, psnr_map = grid_search(x, meas["y"], meas["physics"], prior, data_fidelity,
                                     base_steps, L_safe, grids, algo, cfg, desc=tag,
                                     verbose=verbose, partial_path=partial)

    nan = float("nan")
    final = (compute_final_metrics(x, best["x"]) if best["x"] is not None
             else {"psnr": nan, "ssim": nan, "microssim": nan})

    result = {
        "x_rec": best["x"], "x_gt": x.cpu(),
        "sigma": best.get("sigma"), "iter": best.get("iter"), "lambda": best.get("lambda"),
        "iter_run": best.get("iter_run"),        # < iter if stopped early (cfg.stop_tol)
        "step_scale": best.get("step_scale"), "stepsize": best.get("stepsize"),
        "psnr": best["psnr"], "ssim": final["ssim"], "microssim": final["microssim"],
        "metrics": best.get("metrics"), "L_history": best.get("L_history"),
        # metrics["cost"]: full F = f + lam R per iteration, float64 (explicit priors with
        # PROJ_GD / PROX_GD / MD_BURG). Files without this key: float32 smooth part only.
        "cost_def": "F_fp64",
        "psnr_map": psnr_map, "grid_axes": GRID_AXES,
        "evaluations": evaluations,       # adaptive search: every point tried, in order
        "hit_bound": None if search is None else search["hit_bound"],
        "sigma_grid": grids["sigma"], "step_grid": grids["step"],
        "iter_grid": grids["iter"], "lambda_grid": grids["lambda"],
        "name": name, "flux": flux, "loss": loss_tag, "fidelity": fidelity_name,
        "prior": cfg.prior_name, "algo": algo,
        "prior_type": "explicit" if explicit else ("RED" if algo in GRADIENT_ALGOS else "PnP"),
        "run_id": run_cfg.run_id, "config": run_cfg.grid_signature(),
    }

    if save:
        torch.save(result, out_path)
        append_csv(os.path.join(run_cfg.run_dir, "summary.csv"), result)
        append_csv(os.path.join(cfg.save_dir, "summary_all.csv"), result)
        if os.path.exists(partial_path):
            os.remove(partial_path)

    if best["x"] is None:
        report(f"done {tag}: all configurations diverged")
    else:
        n_runs = "" if search is None else f", {len(evaluations)} runs"
        report(f"done {tag}: sigma={best['sigma']:.4g}, step x{best['step_scale']:.3g}, "
               f"iter={best['iter']}, lambda={best['lambda']:.4g}  PSNR={best['psnr']:.2f} dB  "
               f"SSIM={final['ssim']:.4f}  MicroSSIM={final['microssim']:.4f}  "
               f"({(time.time() - t0) / 60:.1f} min{n_runs})")
        if search is not None and any(search["hit_bound"].values()):
            report(f"WARNING {tag}: best value at the expansion limit {search['hit_bound']}: "
                   f"widen the range or search_max_expand")
    return result
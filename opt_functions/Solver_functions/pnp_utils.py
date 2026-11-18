"""
Grid search for PnP / RED / WCRR reconstructions.

How the prior is used:
    gradient algorithms ('MD', 'GD')     -> RED prior  (grad = x - D(x))
    prox algorithms ('PGD', 'HQS', ...)  -> PnP prior  (prox = D(x))
    explicit priors                      -> R(x) used directly:
        'weakly_convex' (WCRR)  smooth      -> 'PROJ_GD'
        'TV_smooth'             smooth      -> 'PROJ_GD'   (smoothing eps = cfg.tv_eps)
        'L1'                    non-smooth  -> 'PROX_GD'
        'TV_lhs', 'HS'          non-smooth  -> 'PROX_GD'   (learned-hs analysis priors, FGP prox,
                                                            cfg.tv_prox_iter/tol)
        'TV_nonsmooth'          non-smooth  -> 'PROX_GD'   (FGP prox, cfg.tv_prox_iter/tol)

Grids searched (see searched_axes); a grid that is not searched is fixed:
    WCRR, backtracking on            : sigma, lambda, iter         (step fixed)
    WCRR, backtracking 'none'        : sigma, lambda, step, iter
    TV_smooth / TV_nonsmooth / L1, backtracking on  : lambda, iter  (sigma, step fixed)
    TV_smooth / TV_nonsmooth / L1, backtracking none: lambda, step, iter
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
from .prior_config import GRADIENT_ALGOS, REG_PLANE, get_denoiser, make_prior
from .iterators_config import SnapshotIterator, build_model, get_iteration
from .adaptive_search import coarse_to_fine

try:
    from microssim import micro_structural_similarity   # pip install microssim
except ImportError:
    micro_structural_similarity = None

# Priors with an explicit cost R(x): lambda is searched, backtracking can be used
EXPLICIT_PRIORS = {"WCRR", "WCRR_ISM", "WCRR_softplus_ISM", "TV_smooth", "TV_nonsmooth", "L1",
                   "WCRR_allpass", "WCRR_ISM_allpass", "WCRR_softplus_ISM_allpass",
                   "WCHS_free", "WCHS_free_smlm",       # learned-hs (prior_config.WCHSPrior)
                   "TV_lhs", "HS"}                      # learned-hs analysis priors (non-learned)
# Priors that do not use sigma (g_param): the sigma grid is not searched
PRIORS_WITHOUT_SIGMA = {"TV_smooth", "TV_nonsmooth", "L1", "TV_lhs", "HS"}
# Priors whose weights depend on the training loss (one checkpoint per loss)
BT_TEST = "fp64_recheck"   # backtracking test version, in run_id (Config.grid_signature)
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

    # prior: 'TV', 'Drunet', 'Drunet_finetune', 'WCRR', 'TV_smooth', 'TV_nonsmooth', 'L1'
    # ('TV' = deepinv TV denoiser for PnP; 'TV_nonsmooth' = explicit TV for PROX_GD)
    # 'TV_lhs', 'HS': learned-hs TV / Hessian-Schatten (S1) for PROX_GD (tv_prox_iter / tol)
    prior_name: str = "Drunet_finetune"
    tv_eps: float = 1e-2                       # smoothing of 'TV_smooth' (Lip grad R = 8/eps)
    tv_prox_iter: int = 200                    # 'TV_nonsmooth', 'TV_lhs', 'HS': max FGP iterations per prox
    tv_prox_tol: float = 1e-6                  # 'TV_nonsmooth', 'TV_lhs', 'HS': FGP duality-gap tolerance
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
    #                    is on their border. iter_grid with several values: one run of
    #                    max(iter_grid) per point, scored by its best iteration
    #                    (oracle early stopping, e.g. PnP DRUNet)
    search: str = "grid"
    lam_range: Optional[tuple] = None          # (log10 lo, log10 hi)
    sigma_range: Optional[tuple] = None        # (log10 lo, log10 hi)
    step_range: Optional[tuple] = None         # (log10 lo, log10 hi) of the step multiplier,
                                               # PnP only (explicit priors: backtracking)
    search_points: int = 5                     # coarse scan: points per axis
    search_tol: float = 0.05                   # final resolution (decades)
    search_max_expand: float = 3.0             # decades allowed beyond each range
    sigma_max: Optional[float] = None          # log10, hard ceiling of the sigma search
                                               # (WCRR: NaN above ~10^3); not in run_id

    # Early stopping (explicit priors, PROJ_GD / PROX_GD / MD_BURG): stop once
    # ||x_{k+1} - x_k|| / ||x_k|| < stop_tol for stop_patience consecutive iterations;
    # iter_grid is then the maximum number of iterations. None: always iter_grid.
    stop_tol: Optional[float] = None
    stop_patience: int = 10
    # No early stop before stop_min_iter iterations: the backtracking step needs ~50-100
    # iterations to grow from 1 / L_safe, and meanwhile the relative step can be below
    # stop_tol with x still at x_0 (Castle_gray, KL: stopped at iteration 10). Not in
    # run_id: identical results for every run that stops later (all finished ones but
    # the Castle_gray KL runs).
    stop_min_iter: int = 200

    # solver
    algo: str = "PGD"                          # 'PGD', 'PNP_MD', 'PROJ_GD', 'PROX_GD', 'MD', 'GD', ...
    init_choice: str = "adjoint_y"             # 'adjoint_y', 'random', 'constant', 'backprojection'
    # output / snapshots clamped to [0, x_max]; None = no upper bound (real data in photon
    # counts, flux = 1). Not in run_id: 1.0 is what every simulated run has always used.
    x_max: Optional[float] = 1.0
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

    # 25 detectors as one batched operator (data_preparation.batch_measurements): same
    # operator, fidelities and Lipschitz constants up to float rounding, several times
    # faster on the GPU. Not in run_id: results equivalent to the stacked ones.
    batched_physics: bool = False

    # KL / L2 fidelity with the background of the measurement model (L2: ||Ax + b - y||^2,
    # KL: background b * flux in counts). Set by experiment_paths for the KL and L2 runs:
    # new run_id for them; the old runs (L2 without background, KL with a background flux
    # times too small) keep theirs, and KL0 runs (no background) are unchanged.
    fid_bkg: bool = False

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
            if self.step_range is not None:    # runs without it keep their run_id
                sig.update(step_range=vals(self.step_range, rnd))
        if self.stop_tol is not None:  # runs without early stopping keep their run_id
            sig.update(stop_tol=self.stop_tol, stop_patience=self.stop_patience)
        if self.prior_name in ("TV_nonsmooth", "TV_lhs", "HS"):   # other priors keep their run_id
            sig.update(tv_prox_iter=self.tv_prox_iter, tv_prox_tol=self.tv_prox_tol)
        if self.Nz > 1:                # 3D: the regularized / measured plane (2D: no change)
            sig.update(reg_plane=REG_PLANE)
        if self.fid_bkg:               # runs with the old background keep their run_id
            sig.update(fid_bkg=True)
        if self.backtracking == "custom" and self.prior_name in EXPLICIT_PRIORS:
            # backtracking test with the float64 re-check of near misses
            # (iterators_config.ProxGDIteration); runs with the float32-only test keep
            # their run_id
            sig.update(bt_test=BT_TEST)
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
# adaptive search: starting range (log10) of each continuous axis
RANGE_FIELDS = {"sigma": "sigma_range", "lambda": "lam_range", "step": "step_range"}
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
        if adaptive and axis in searched and axis in RANGE_FIELDS:
            # adaptive search: the "grid" is the starting range (its two ends)
            r = getattr(cfg, RANGE_FIELDS[axis])
            if r is None:
                raise ValueError(f"adaptive search of '{axis}': set cfg.{RANGE_FIELDS[axis]}")
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
    return grids, notes


def search_plan(cfg, algo, grids=None, notes=()):
    """Readable description of what a run will search."""
    if grids is None:
        grids, notes = resolve_grids(cfg, algo)
    searched, fixed = searched_axes(cfg, algo)
    lines = [f"search plan: prior {cfg.prior_name}, algo {algo.upper()}"]
    if cfg.prior_name == "TV_smooth":
        lines.append(f"  tv_eps = {cfg.tv_eps:g}")
    if cfg.prior_name in ("TV_nonsmooth", "TV_lhs", "HS"):
        lines.append(f"  {cfg.prior_name} prox (FGP, x >= 0): <= {cfg.tv_prox_iter} iterations, "
                     f"tol {cfg.tv_prox_tol:g}")
    adaptive = cfg.search == "adaptive"
    if adaptive:
        lines[0] += (f", adaptive search (coarse scan {cfg.search_points} points/axis, "
                     f"expansion <= {cfg.search_max_expand:g} decades, "
                     f"resolution {cfg.search_tol:g} decades)")
    for axis in ("sigma", "lambda", "step", "iter"):
        g = grids[axis].tolist()
        if adaptive and axis in searched and axis in RANGE_FIELDS:
            lines.append(f"  {axis:<6} searched from range [{g[0]:.3g}, {g[-1]:.3g}]  "
                         f"<- {searched[axis]}")
        elif adaptive and axis == "iter" and len(g) > 1:
            lines.append(f"  {axis:<6} best of {len(g)} counts in [{g[0]}, {g[-1]}] "
                         f"(oracle early stopping, one run of {g[-1]} per point)")
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


def metric_view(x):
    """The part of an image the metrics are computed on: 3D data (B, 2, H, W) -> plane
    REG_PLANE (B, 1, H, W), the in-focus one, also the only regularized one (PlanePrior);
    2D (one channel) unchanged."""
    return x[:, REG_PLANE:REG_PLANE + 1] if x.ndim == 4 and x.shape[1] > 1 else x


class RegPlanePSNR(torch.nn.Module):
    """deepinv's per-iteration PSNR (BaseOptim.psnr, metrics["psnr"]) on plane REG_PLANE."""

    def forward(self, x, x_gt):
        return psnr_fn(metric_view(x), metric_view(x_gt))


def psnr(x_rec, x_gt):
    """PSNR after scaling the reconstruction to max 1 (as used in the grid search).
    3D: plane REG_PLANE only (metric_view)."""
    x_rec, x_gt = metric_view(x_rec), metric_view(x_gt)
    return psnr_fn(x_rec / x_rec.max().clamp_min(1e-12), x_gt).item()


def compute_final_metrics(x_gt, x_rec):
    """PSNR / SSIM / MicroSSIM of the best reconstruction. 3D: plane REG_PLANE only (metric_view)."""
    x_gt, x_rec = metric_view(x_gt), metric_view(x_rec)
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


# Per-iteration progress bar inside every reconstruction (iteration, relative step, early-
# stop counter). Off by default: evaluate_methods.py keeps only its bar over the search
# points. single_experiment.py sets it to True.
PROGRESS_BAR = False


def _with_progress(stop_fn, bar, iteration, cfg, every=20):
    """Wrap deepinv's check_conv_fn (called at every iteration from it = 2 on) to move
    the bar; returns stop_fn's answer (False without early stopping)."""
    def fn(it, X_prev, X):
        bar.update(it + 1 - bar.n)
        if it % every == 0:
            a, b = X_prev["est"][0], X["est"][0]
            post = {"rel_step": f"{float((b - a).norm() / a.norm().clamp_min(1e-12)):.2e}"}
            if stop_fn is not None and hasattr(iteration, "_n_small"):
                post["stop"] = f"{iteration._n_small}/{cfg.stop_patience} < {cfg.stop_tol:g}"
            bar.set_postfix(post, refresh=False)
        return stop_fn(it, X_prev, X) if stop_fn is not None else False
    return fn


def solve(x, y, physics, prior, data_fidelity, tau, sigma, lam, iters, algo, cfg, L_max,
          log_cost=True, metrics=True):
    """Reconstruct with one parameter set. Returns ({n: (x after n iterations, deepinv
    metrics)} for every n in iters, the step-size history of the backtracking or None).
    log_cost: metrics["cost"] = full F = f + lam R in float64 (custom explicit-prior
    iterations, ~45% slower); False: the float32 smooth part of F.
    metrics: False = no per-iteration history at all (deepinv PSNR / residual / cost, 3
    GPU->CPU syncs per iteration; metrics is then None): for the points of a search, whose
    histories are never used (the chosen point is rerun with them). The iterates are the
    same either way."""
    explicit = getattr(prior, "explicit_prior", False)
    deepinv_bt = explicit and cfg.backtracking == "deepinv"
    bt_kwargs = {}
    if explicit and cfg.backtracking == "custom":
        bt_kwargs = dict(backtracking=True, s=cfg.bt_s, eta=cfg.bt_eta,
                         delta=cfg.bt_delta, max_bt=cfg.bt_max, L_max=L_max)
    stop = explicit and getattr(cfg, "stop_tol", None) is not None
    if stop:
        bt_kwargs.update(stop_tol=cfg.stop_tol, stop_patience=cfg.stop_patience,
                         stop_min_iter=getattr(cfg, "stop_min_iter", 0))

    def run(iteration, n_iter):
        stop_fn = iteration.check_stop if stop and hasattr(iteration, "check_stop") else None
        bar = None
        if PROGRESS_BAR:
            bar = tqdm(total=n_iter, desc="iterations", leave=False, mininterval=1.0)
            stop_fn = _with_progress(stop_fn, bar, iteration, cfg)
        model = build_model(prior, data_fidelity, iteration, tau, sigma, lam, n_iter,
                            cfg.init_choice, deepinv_bt, stop_fn=stop_fn,
                            x_max=getattr(cfg, "x_max", 1.0))
        if getattr(cfg, "is_3d", False) and hasattr(model, "psnr"):
            model.psnr = RegPlanePSNR()        # per-iteration PSNR on plane REG_PLANE (3D)
        try:
            with torch.no_grad():
                if metrics:
                    x_out, hist = model(y, physics, x_gt=x, compute_metrics=True)
                else:
                    x_out, hist = model(y, physics, compute_metrics=False), None
        finally:
            if bar is not None:
                bar.close()
        return x_out.detach().cpu(), hist

    log_cost = log_cost and metrics          # the float64 history is part of the metrics
    iteration = get_iteration(algo, snap_iters=iters, log_cost=log_cost,
                              x_max=getattr(cfg, "x_max", 1.0), **bt_kwargs)
    if isinstance(iteration, SnapshotIterator):
        # one run of max(iters) iterations; the other counts come from the snapshots
        max_it = max(iters)
        x_out, mets = run(iteration, max_it)
        hist = getattr(iteration, "history", None)     # explicit priors with log_cost
        if mets is not None and hist and hist["step_norm"]:
            mets.update({k: [v] for k, v in hist.items()})
        snaps = {**iteration.snapshots, max_it: x_out}
        # stopped early: the counts after the stop get the last iterate
        recs = {n: (snaps.get(n, x_out), mets) for n in iters}
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
    One grid point: run with every per-iteration history. Several: the points run
    without histories (only their PSNR is used), the best one is rerun with them.
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

    def entry(p, sigma, step, n, lam, k, tau, recs, L_hist):
        metrics = recs[n][1]
        return {"psnr": p, "sigma": sigma, "step_scale": step, "iter": n, "lambda": lam,
                "lam_idx": k, "iter_run": iterations_run(metrics, n),
                "stepsize": tau, "x": recs[n][0].clone(),
                "metrics": None if metrics is None else
                {key: [v[:n] for v in val] for key, val in metrics.items()},
                "L_history": L_hist[:n] if L_hist else None}

    combos = list(itertools.product(enumerate(sigmas), enumerate(steps), enumerate(lambdas)))
    full = len(combos) == 1
    for (i, sigma), (t, step), (k, lam) in tqdm(combos, desc=desc, leave=False):
        if (i, t, k) in done:
            continue
        sigma, step, lam = float(sigma), float(step), float(lam)
        tau = base_steps[k] * step
        # recs[n] = (reconstruction after n iterations, deepinv metrics)
        recs, L_hist = solve(x, y, physics, prior, data_fidelity, tau, sigma, lam, iters,
                             algo, cfg, L_safe[k], log_cost=full, metrics=full)

        line = []
        for j, n in enumerate(iters):
            x_n = recs[n][0]
            if x_n is None or not torch.isfinite(x_n).all():
                line.append(f"{n}:div")
                continue
            p = psnr(x_n, x_cpu)
            psnr_map[i, t, j, k] = p
            line.append(f"{n}:{p:.2f}")
            if p > best["psnr"]:
                best = entry(p, sigma, step, n, lam, k, tau, recs, L_hist)

        if verbose:
            tqdm.write(f"sigma={sigma:.4g}  step x{step:.3g} (tau={tau:.3e})  lambda={lam:.4g}  "
                       f"PSNR by iter: " + "  ".join(line))

        done.add((i, t, k))
        if partial_path:
            torch.save({"psnr_map": psnr_map, "best": best, "done": sorted(done)}, partial_path)

    if not full and best["x"] is not None and best.get("metrics") is None:
        # rerun the chosen point with every per-iteration history (same iterates)
        k = best.get("lam_idx")
        if k is None:            # partial file written before lam_idx existed
            k = int(torch.argmin((torch.as_tensor(lambdas).double() - best["lambda"]).abs()))
        recs, L_hist = solve(x, y, physics, prior, data_fidelity, best["stepsize"],
                             best["sigma"], best["lambda"], iters, algo, cfg, L_safe[k])
        best = entry(best["psnr"], best["sigma"], best["step_scale"], best["iter"],
                     best["lambda"], k, best["stepsize"], recs, L_hist)
    return best, psnr_map


def adaptive_param_search(x, meas, prior, fidelity_name, iters, algo, cfg,
                          desc="adaptive search", verbose=False, partial_path=None):
    """
    Coarse-to-fine search of the searched continuous axes among sigma, lambda, step
    (at most 2; adaptive_search.coarse_to_fine). Each point is one run of max(iters)
    iterations, scored by the best PSNR over the counts in iters (one count: no early
    stopping). Returns (best, evaluations, search) where evaluations is a list of
    {"sigma", "lambda", "step", "iter", "psnr"} in the order they were computed.
    If partial_path is given, every evaluation is saved and replayed on restart.
    """
    searched, _ = searched_axes(cfg, algo)
    iters = sorted(int(n) for n in torch.as_tensor(iters).flatten())
    ranges = {}
    for a in ("sigma", "lambda", "step"):
        if a not in searched:
            continue
        r = getattr(cfg, RANGE_FIELDS[a])
        if r is None or len(r) != 2:
            raise ValueError(f"adaptive search of '{a}': set cfg.{RANGE_FIELDS[a]} = "
                             f"(log10 lo, log10 hi)"
                             + (" or use backtracking" if a == "step" else ""))
        ranges[a] = tuple(float(v) for v in r)

    x_cpu = x.detach().float().cpu()
    cache, evaluations = {}, []
    if partial_path and os.path.exists(partial_path):
        state = torch.load(partial_path, weights_only=False)
        cache, evaluations = state["cache"], state["evaluations"]
        tqdm.write(f"resuming {desc}: {len(cache)} points already done")
    bar = tqdm(desc=desc, leave=False, initial=len(cache))
    recs = {}           # point -> reconstruction of this session (resumed points: rerun)

    def reconstruct(point, full=False):
        """full=False: a search point, only its PSNR is used (no per-iteration history);
        full=True: the chosen point, with every history (convergence plots)."""
        sigma = 10 ** point["sigma"] if "sigma" in point else FIXED_VALUES["sigma"]
        lam = 10 ** point["lambda"] if "lambda" in point else FIXED_VALUES["lambda"]
        step = 10 ** point["step"] if "step" in point else FIXED_VALUES["step"]
        df, tau, L_safe = fidelity_and_steps(fidelity_name, meas, lam, prior, cfg)
        tau = tau * step
        out, L_hist = solve(x, meas["y"], meas["physics"], prior, df, tau, sigma, lam,
                            iters, algo, cfg, L_safe, log_cost=full, metrics=full)
        p, n = -float("inf"), iters[-1]           # best iteration count (early stopping)
        for k in iters:
            x_k = out[k][0]
            if x_k is not None and torch.isfinite(x_k).all():
                p_k = psnr(x_k, x_cpu)
                if p_k > p:
                    p, n = p_k, k
        x_n, metrics = out[n]
        if metrics is not None and len(iters) > 1:
            metrics = {key: [v[:n] for v in val] for key, val in metrics.items()}
        recs[tuple(sorted(point.items()))] = {
            "psnr": p, "sigma": sigma, "step_scale": step, "iter": n, "lambda": lam,
            "iter_run": iterations_run(metrics, n),
            "stepsize": tau, "x": x_n if p > -float("inf") else None, "metrics": metrics,
            "L_history": L_hist[:n] if L_hist else None}
        return p, sigma, lam, step, n

    def score(point):
        p, sigma, lam, step, n = reconstruct(point)
        evaluations.append({"sigma": sigma, "lambda": lam, "step": step, "iter": n, "psnr": p})
        if verbose:
            tqdm.write(f"sigma={sigma:.4g}  lambda={lam:.4g}  step x{step:.3g}  "
                       f"iter={n}  PSNR={p:.2f}")
        bar.update(1)
        return p

    def save(point, value):
        if partial_path:
            torch.save({"cache": cache, "evaluations": evaluations}, partial_path)

    search = coarse_to_fine(score, ranges, n_points=cfg.search_points, tol=cfg.search_tol,
                            max_expand=cfg.search_max_expand, cache=cache, on_eval=save,
                            bounds={"sigma": (None, cfg.sigma_max)})
    bar.close()
    # The search runs without any per-iteration history (only the PSNR of each point is
    # used); rerun the chosen point with all of them: deepinv PSNR / residual / cost and,
    # for explicit priors, the float64 F = f + lam R, step and iterate norms, criticality.
    key = tuple(sorted(search["best"].items()))
    reconstruct(search["best"], full=True)
    best = recs[key]
    if len(iters) > 1:     # best iteration = last count: no early stopping happened
        search["hit_bound"]["iter"] = best["iter"] == iters[-1]
    return best, evaluations, search


# ---------------------------------------------------------------------------
# One experiment
# ---------------------------------------------------------------------------

BKG_FIDELITIES = ("KL", "L2")   # fidelities with the measurement background (Config.fid_bkg)

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
                      lam_grid=grids["lambda"], step_grid=grids["step"],
                      fid_bkg=fidelity_name.upper() in BKG_FIDELITIES)
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
    prior = make_prior(denoiser, algo, cfg.is_3d)   # 3D: only plane REG_PLANE is regularized

    report(f"start {tag} in {run_cfg.run_dir}\n{search_plan(cfg, algo, grids, notes)}")
    t0 = time.time()
    partial = partial_path if save else None
    evaluations, search = None, None
    if cfg.search == "adaptive":
        best, evaluations, search = adaptive_param_search(
            x, meas, prior, fidelity_name, grids["iter"], algo, cfg, desc=tag,
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
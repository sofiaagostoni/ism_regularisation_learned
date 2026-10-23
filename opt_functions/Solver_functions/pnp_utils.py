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

    def grid_signature(self):
        """Everything that changes the results of a grid search."""
        def vals(g, cast):
            return None if g is None else [cast(v) for v in torch.as_tensor(g).flatten()]

        def rnd(v):
            return round(float(v), 6)

        return {
            "prior_name": self.prior_name, "tv_eps": self.tv_eps,
            "sigma_grid": vals(self.sigma_grid, rnd), "iter_grid": vals(self.iter_grid, int),
            "lam_grid": vals(self.lam_grid, rnd), "step_grid": vals(self.step_grid, rnd),
            "kl_step_scale": self.kl_step_scale, "l2_step_scale": self.l2_step_scale,
            "init_choice": self.init_choice, "backtracking": self.backtracking,
            "bt": [self.bt_s, self.bt_eta, self.bt_delta, self.bt_max],
            "seed": self.seed, "Nx": self.Nx, "Nz": self.Nz, "pxsizex": self.pxsizex,
        }

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
    for axis, field_name in GRID_FIELDS.items():
        g = given[axis] if given[axis] is not None else getattr(cfg, field_name)
        g = None if g is None else torch.as_tensor(g).flatten()
        if axis in searched:
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
    for axis in ("sigma", "lambda", "step", "iter"):
        g = grids[axis].tolist()
        if axis in searched:
            shown = g if len(g) <= 7 else g[:3] + ["..."] + g[-2:]
            lines.append(f"  {axis:<6} searched ({len(g)}): {shown}  <- {searched[axis]}")
        else:
            lines.append(f"  {axis:<6} fixed = {g[0]:g}  <- {fixed[axis]}")
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


def grid_search(x, y, physics, prior, data_fidelity, base_steps, L_safe, grids, algo, cfg,
                desc="grid search", verbose=False, partial_path=None):
    """
    Search sigma x step x iter x lambda. For lambda index k the step used is
    base_steps[k] * step. Returns (best, psnr_map). If partial_path is given,
    progress is saved after every grid point and resumed on restart.
    """
    sigmas, steps, lambdas = grids["sigma"], grids["step"], grids["lambda"]
    iters = [int(n) for n in grids["iter"]]
    max_it = max(iters)
    psnr_map = torch.full((len(sigmas), len(steps), len(iters), len(lambdas)), float("nan"))
    best = {"psnr": -float("inf"), "x": None}
    done = set()
    x_cpu = x.detach().float().cpu()

    if partial_path and os.path.exists(partial_path):
        state = torch.load(partial_path, weights_only=False)
        if state["psnr_map"].shape == psnr_map.shape:
            psnr_map, best, done = state["psnr_map"], state["best"], set(map(tuple, state["done"]))
            tqdm.write(f"resuming {desc}: {len(done)} grid points already done")

    explicit = getattr(prior, "explicit_prior", False)
    custom_bt = explicit and cfg.backtracking == "custom"
    deepinv_bt = explicit and cfg.backtracking == "deepinv"

    combos = list(itertools.product(enumerate(sigmas), enumerate(steps), enumerate(lambdas)))
    for (i, sigma), (t, step), (k, lam) in tqdm(combos, desc=desc, leave=False):
        if (i, t, k) in done:
            continue
        sigma, step, lam = float(sigma), float(step), float(lam)
        tau = base_steps[k] * step
        bt_kwargs = {}
        if custom_bt:
            bt_kwargs = dict(backtracking=True, s=cfg.bt_s, eta=cfg.bt_eta,
                             delta=cfg.bt_delta, max_bt=cfg.bt_max, L_max=L_safe[k])

        def run(iteration, n_iter):
            model = build_model(prior, data_fidelity, iteration, tau, sigma, lam, n_iter,
                                cfg.init_choice, deepinv_bt)
            with torch.no_grad():
                x_out, metrics = model(y, physics, x_gt=x, compute_metrics=True)
            return x_out.detach().cpu(), metrics

        # recs[n] = (reconstruction after n iterations, deepinv metrics)
        iteration = get_iteration(algo, snap_iters=iters, **bt_kwargs)
        if isinstance(iteration, SnapshotIterator):
            # one run of max_it iterations; the other counts come from the snapshots
            x_out, metrics = run(iteration, max_it)
            snaps = {**iteration.snapshots, max_it: x_out}
            recs = {n: (snaps.get(n), metrics) for n in iters}
        else:
            # deepinv built-in iteration: one run per iteration count
            recs = {n: run(iteration, n) for n in iters}

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
                L_hist = getattr(iteration, "L_history", None)
                best = {"psnr": p, "sigma": sigma, "step_scale": step, "iter": n, "lambda": lam,
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

    grids, notes = resolve_grids(cfg, algo, sigmas, iters, lambdas, steps)
    tag = f"Fidelity{fidelity_name}_{name}_flux{flux}_{cfg.prior_name}_loss{loss_tag}_{algo}"

    # the grids actually used define the run folder
    run_cfg = replace(cfg, sigma_grid=grids["sigma"], iter_grid=grids["iter"],
                      lam_grid=grids["lambda"], step_grid=grids["step"])
    out_path = os.path.join(run_cfg.run_dir, f"best_rec_{tag}.pth")
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

    steps_info = [fidelity_and_steps(fidelity_name, meas, float(l), prior, cfg)
                  for l in grids["lambda"]]
    data_fidelity = steps_info[0][0]
    base_steps = [s[1] for s in steps_info]
    L_safe = [s[2] for s in steps_info]

    report(f"start {tag} in {run_cfg.run_dir}\n{search_plan(cfg, algo, grids, notes)}")
    t0 = time.time()
    best, psnr_map = grid_search(x, meas["y"], meas["physics"], prior, data_fidelity,
                                 base_steps, L_safe, grids, algo, cfg, desc=tag,
                                 verbose=verbose, partial_path=partial_path if save else None)

    nan = float("nan")
    final = (compute_final_metrics(x, best["x"]) if best["x"] is not None
             else {"psnr": nan, "ssim": nan, "microssim": nan})

    result = {
        "x_rec": best["x"], "x_gt": x.cpu(),
        "sigma": best.get("sigma"), "iter": best.get("iter"), "lambda": best.get("lambda"),
        "step_scale": best.get("step_scale"), "stepsize": best.get("stepsize"),
        "psnr": best["psnr"], "ssim": final["ssim"], "microssim": final["microssim"],
        "metrics": best.get("metrics"), "L_history": best.get("L_history"),
        "psnr_map": psnr_map, "grid_axes": GRID_AXES,
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
        report(f"done {tag}: sigma={best['sigma']:.4g}, step x{best['step_scale']:.3g}, "
               f"iter={best['iter']}, lambda={best['lambda']:.4g}  PSNR={best['psnr']:.2f} dB  "
               f"SSIM={final['ssim']:.4f}  MicroSSIM={final['microssim']:.4f}  "
               f"({(time.time() - t0) / 60:.1f} min)")
    return result
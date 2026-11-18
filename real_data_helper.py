"""
PnP / RED reconstructions of real data: no ground truth, no metrics.

The iterations, priors, step-size rules and grids are imported from the
simulation helpers, so a parameter found on simulated data (sigma, step
multiplier, lambda, iterations) means exactly the same thing here.

Two ways to use it:
  reconstruct(...)  one reconstruction with given parameters, e.g. the best
                    ones of the simulation (see params_from_simulation)
  sweep(...)        every combination of a grid, all saved, to compare them
                    by eye. Nothing is ranked: without ground truth there is
                    no score to rank by.

Results layout (same idea as the simulation, nothing is overwritten):
  save_dir/
    summary_real.csv          <- one row per saved reconstruction, all runs
    <run_id>/                 <- one folder per grid/settings combination
      config.json
      log.txt
      summary.csv
      rec_<tag>_sigma.._step.._lam...pth   <- one file per (sigma, step, lambda),
                                              holding every iteration count
"""
import os
import csv
import time
import itertools
from dataclasses import dataclass, replace

import pandas as pd
import torch
from tqdm.auto import tqdm

from opt_functions.Solver_functions import pnp_utils as gs   # <- name of your simulation helpers file


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RealConfig(gs.Config):
    """Same fields as the simulation Config; only the default paths change.
    Set Nx, Nz, pxsizex to the values of the real acquisition."""
    data_dir: str = "Data/Real_data"
    save_dir: str = "Results/results_real"

    @property
    def global_csv_path(self):
        return os.path.join(self.save_dir, "summary_real.csv")


# ---------------------------------------------------------------------------
# Data and forward model
# ---------------------------------------------------------------------------

MEAS_REQUIRED = ("y", "physics", "df_kl", "df_l2", "L_kl", "L_l2")


def load_real_data(name, cfg):
    """{data_dir}/{name}.pth: either the measurement tensor y, or a dict with
    'y' and whatever else the forward model needs (e.g. 'PSF')."""
    d = torch.load(os.path.join(cfg.data_dir, f"{name}_data.pth"), map_location=cfg.device,
                   weights_only=False)
    if torch.is_tensor(d):
        d = {"y": d}
    d["y"] = d["y"].float().to(cfg.device)
    return d


def make_real_measurements(data, cfg):
    """
    Forward model and data fidelities for the real measurement data["y"].

    Must return a dict with the keys of MEAS_REQUIRED, built exactly as in
    generate_meas_ism (same physics, same KL / L2 fidelities, same formulas
    for L_kl and L_l2), only without simulating y. Otherwise the step
    multipliers and sigmas found on simulated data do not transfer.

    Intensity scale: the iterates and the denoiser output are clamped to
    [0, 1], like the simulated ground truth. The fidelity must therefore
    contain the photon scale of the data (the equivalent of the simulation
    flux, e.g. a gain), so that the solution x lies in [0, 1].
    """
    raise NotImplementedError(
        "make_real_measurements: build physics, df_kl, df_l2, L_kl, L_l2 from "
        "data['y'] as generate_meas_ism does (see the docstring)")


def _check_meas(meas):
    missing = [k for k in MEAS_REQUIRED if k not in meas]
    if missing:
        raise ValueError(f"meas is missing {missing} (needs {MEAS_REQUIRED})")


# ---------------------------------------------------------------------------
# Parameters from the simulation
# ---------------------------------------------------------------------------

def params_from_simulation(csv_path, data, flux, fidelity, prior, algo, loss="none",
                           run_ids=None):
    """
    Best parameters (by PSNR) of one simulated experiment, from summary_all.csv,
    as keyword arguments for reconstruct():
        reconstruct(cfg, "cell1", "KL", **params_from_simulation(...))
    Choose the simulated data and flux closest to the real acquisition.
    """
    df = pd.read_csv(csv_path)
    if run_ids is not None:
        df = df[df.run_id.isin(run_ids)]
    sel = df[(df.data == data) & (df.flux == flux)
             & (df.fidelity.astype(str).str.upper() == fidelity.upper())
             & (df.prior.astype(str) == prior)
             & (df.algo.astype(str).str.upper() == algo.upper())
             & (df.loss.fillna("none").astype(str) == gs.effective_loss(loss, prior))]
    if sel.empty:
        raise ValueError(f"no simulation result for data={data}, flux={flux}, "
                         f"fidelity={fidelity}, prior={prior}, algo={algo}, loss={loss}")
    r = sel.loc[sel.psnr.idxmax()]
    params = {"sigma": float(r.sigma), "n_iter": int(r.iter),
              "step_scale": 1.0 if pd.isna(r.get("step_scale")) else float(r.step_scale),
              "lam": 1.0 if pd.isna(r["lambda"]) else float(r["lambda"])}
    print(f"from simulation (run {r.run_id}, PSNR {r.psnr:.2f}): {params}")
    return params


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _run(model, y, physics):
    with torch.no_grad():
        out = model(y, physics)          # no x_gt, no metrics
    return out[0] if isinstance(out, tuple) else out


def _setup(cfg, loss, fidelity_name, algo, lambdas, meas, denoiser):
    """Prior, data fidelity, base step and safe Lipschitz bound for each lambda."""
    explicit = cfg.prior_name in gs.EXPLICIT_PRIORS
    if denoiser is None:
        denoiser = gs.get_denoiser(loss, cfg)
    prior = denoiser if explicit else gs.wrap_prior(denoiser, algo)
    data_fidelity, _ = gs.get_fidelity(fidelity_name, meas, cfg, float(lambdas[0]), prior)
    base_steps = [gs.get_fidelity(fidelity_name, meas, cfg, float(l), prior)[1]
                  for l in lambdas]
    L_max = [gs.lipschitz_bound(fidelity_name, meas, float(l), prior) for l in lambdas]
    return prior, data_fidelity, base_steps, L_max


def _solve(prior, data_fidelity, y, physics, tau, sigma, iters, lam, algo, cfg, L_max):
    """{n_iter: reconstruction} for every n in iters. The custom iterations
    do one run of max(iters) and record the intermediate iterates."""
    iters = sorted({int(n) for n in iters})
    max_it = iters[-1]
    kw = {}
    if gs.bt_mode(cfg) == "custom" and getattr(prior, "explicit_prior", False):
        kw = dict(backtracking=True, s=cfg.bt_s, eta=cfg.bt_eta, delta=cfg.bt_delta,
                  max_bt=cfg.bt_max, L_max=L_max)

    iteration = gs.get_iteration(algo, snap_iters=[n for n in iters if n < max_it], **kw)
    if isinstance(iteration, gs.SnapshotIterator):
        model = gs.build_model(prior, data_fidelity, iteration, tau, sigma, max_it, lam, cfg)
        recs = dict(iteration.snapshots)
        recs[max_it] = _run(model, y, physics).detach().cpu()
        L_hist = getattr(iteration, "L_history", None)
        return {n: recs.get(n) for n in iters}, (list(L_hist) if L_hist else None)

    recs = {}                            # deepinv built-in iteration: one run per count
    for n in iters:
        model = gs.build_model(prior, data_fidelity, gs.get_iteration(algo, **kw),
                               tau, sigma, n, lam, cfg)
        recs[n] = _run(model, y, physics).detach().cpu()
    return recs, None


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

CSV_FIELDS = ["run_id", "data", "fidelity", "loss", "prior", "algo", "sigma",
              "step_scale", "stepsize", "lambda", "iter", "diverged", "file"]


def _append_csv(path, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Sweep and single reconstruction
# ---------------------------------------------------------------------------

def sweep(cfg, name, fidelity_name, loss="none", algo=None,
          sigmas=None, iters=None, lambdas=None, steps=None,
          data=None, meas=None, denoiser=None, save=True, overwrite=False):
    """
    Reconstruct the real data `name` for every combination of the grids.

    Which grids are searched depends on the prior and algorithm, exactly as in
    the simulation (gs.searched_axes); arguments override the grids in cfg.
    data / meas / denoiser : pre-computed objects, to avoid reloading in loops.
    save      : one .pth per (sigma, step, lambda), with every iteration count,
                and one csv row per reconstruction. Already saved combinations
                are skipped, so an interrupted sweep resumes.
    overwrite : recompute combinations that are already saved.

    Returns a list with one dict per (sigma, step, lambda):
        recs     {n_iter: image}, None if skipped (then load "file")
        x_rec    the image at the largest iteration count
        diverged {n_iter: bool}
    """
    algo = (cfg.algo if algo is None else algo).upper()
    loss_tag = gs.effective_loss(loss, cfg.prior_name)
    grids, notes = gs.resolve_grids(cfg, algo, sigmas=sigmas, iters=iters,
                                    lambdas=lambdas, steps=steps)
    sigmas, iters, lambdas, steps = (grids["sigma"], grids["iter"],
                                     grids["lambda"], grids["step"])

    run_cfg = replace(cfg, sigma_grid=sigmas, iter_grid=iters,
                      lam_grid=lambdas, step_grid=steps,
                      fid_bkg=fidelity_name.upper() in gs.BKG_FIDELITIES)  # as experiment_paths
    if save:
        run_cfg.init_run_dir()
    say = (lambda msg: gs.log(msg, run_cfg)) if save else tqdm.write
    tag = f"Fidelity{fidelity_name}_{name}_{cfg.prior_name}_loss{loss_tag}_{algo}"

    def file_for(s, m, l):
        return os.path.join(run_cfg.run_dir, f"rec_{tag}_sigma{s:g}_step{m:g}_lam{l:g}.pth")

    combos = list(itertools.product(sigmas.tolist(), steps.tolist(), lambdas.tolist()))
    results, todo = [], []
    for s, m, l in combos:
        if save and not overwrite and os.path.exists(file_for(s, m, l)):
            results.append({"sigma": s, "step_scale": m, "lambda": l,
                            "file": file_for(s, m, l), "recs": None, "x_rec": None})
        else:
            todo.append((s, m, l))
    if results:
        say(f"{tag}: {len(results)} combinations already saved, skipped")
    if not todo:
        return results

    if meas is None:
        if data is None:
            data = load_real_data(name, cfg)
        meas = make_real_measurements(data, cfg)
    _check_meas(meas)
    prior, data_fidelity, base_steps, L_max = _setup(cfg, loss, fidelity_name, algo,
                                                     lambdas, meas, denoiser)
    say(f"start {tag} in {run_cfg.run_dir}\n{gs.search_plan(cfg, algo, grids, notes)}")

    lam_index = {l: k for k, l in enumerate(lambdas.tolist())}
    for s, m, l in tqdm(todo, desc=tag, leave=False):
        k = lam_index[l]
        tau = base_steps[k] * m
        t0 = time.time()
        recs, L_hist = _solve(prior, data_fidelity, meas["y"], meas["physics"], tau, s,
                              iters.tolist(), l, algo, cfg, L_max[k])
        diverged = {n: x is None or not bool(torch.isfinite(x).all()) for n, x in recs.items()}
        entry = {"sigma": s, "step_scale": m, "lambda": l, "stepsize": tau,
                 "recs": recs, "x_rec": recs[max(recs)], "diverged": diverged,
                 "L_history": L_hist, "file": file_for(s, m, l)}

        if save:
            torch.save({**entry, "name": name, "fidelity": fidelity_name, "loss": loss_tag,
                        "prior": cfg.prior_name, "algo": algo, "run_id": run_cfg.run_id,
                        "config": run_cfg.grid_signature()}, entry["file"])
            for n in recs:
                row = {"run_id": run_cfg.run_id, "data": name, "fidelity": fidelity_name,
                       "loss": loss_tag, "prior": cfg.prior_name, "algo": algo,
                       "sigma": s, "step_scale": m, "stepsize": tau, "lambda": l,
                       "iter": n, "diverged": diverged[n], "file": entry["file"]}
                _append_csv(run_cfg.csv_path, row)
                _append_csv(run_cfg.global_csv_path, row)

        div = [n for n, d in diverged.items() if d]
        say(f"sigma={s:.4g}  step x{m:.3g} (tau={tau:.3e})  lambda={l:.4g}  "
            f"iters {sorted(recs)}" + (f"  DIVERGED at {div}" if div else "")
            + f"  ({(time.time() - t0) / 60:.1f} min)")
        results.append(entry)
    return results


def reconstruct(cfg, name, fidelity_name, sigma, n_iter, step_scale=1.0, lam=1.0,
                loss="none", algo=None, **kwargs):
    """
    One reconstruction with given parameters. n_iter: an int, or a list to
    keep several iteration counts from the same run. Extra keyword arguments
    (data, meas, denoiser, save, overwrite) are passed to sweep().
    As in the simulation, lambda is ignored for PnP and the step multiplier
    is ignored for WCRR with backtracking.
    Returns the dict of sweep(); x_rec is the image at the largest n_iter.
    """
    iters = torch.as_tensor(n_iter).flatten().tolist()
    out = sweep(cfg, name, fidelity_name, loss=loss, algo=algo, sigmas=[sigma],
                iters=iters, lambdas=[lam], steps=[step_scale], **kwargs)[0]
    if out["recs"] is None:                         # already saved: load it
        out = torch.load(out["file"], weights_only=False)
    return out


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = RealConfig(prior_name="Drunet_finetune", algo="PGD",
                     Nx=3000, pxsizex=40)            # values of the real acquisition

    # 1) parameters of the closest simulated case
    p = params_from_simulation("Results/results_gridsearch/summary_all.csv",
                               data="tub_level", flux=5, fidelity="KL",
                               prior="Drunet_finetune", algo="PGD", loss="L2")
    res = reconstruct(cfg, "my_cell", "KL", loss="L2", **p)

    # 2) or a small sweep around them, to compare by eye
    cfg.sigma_grid = torch.tensor([0.5, 1.0, 2.0]) * p["sigma"]
    cfg.step_grid = torch.tensor([p["step_scale"]])
    cfg.iter_grid = torch.tensor([p["n_iter"] // 2, p["n_iter"], 2 * p["n_iter"]])
    results = sweep(cfg, "my_cell", "KL", loss="L2")
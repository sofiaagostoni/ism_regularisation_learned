"""
Rerun only the chosen parameters of finished evaluate_methods.py results, to add the
convergence histories without redoing the parameter searches (1 run instead of ~15-60).

For every existing SAVE_DIR/<run_id>/best_rec_<tag>.pth of METHODS x jobs() that has no
float64 history yet, the same measurement (same seed) is reconstructed with the saved
(sigma, lambda) and N_ITER iterations, and the file is updated with
    metrics["cost"]          F(x_k) = f + lam R, float64
    metrics["step_norm"]     ||x_{k+1} - x_k||       (unclamped iterates)
    metrics["iterate_norm"]  ||x_{k+1}||
    metrics["criticality"]   ||x_k - max(0, x_k - grad F(x_k))||
    L_history, x_rec, psnr / ssim / microssim of this run (psnr_search: the old PSNR;
    equal for TV / L1, slightly different for WCRR, whose GPU runs are not bitwise
    reproducible), cost_def = "F_fp64".
The old file is kept in SAVE_DIR/_pre_fp64/<run_id>/. evaluation*.csv are not changed.

    python refresh_histories.py --plan              # list the files to update
    python refresh_histories.py --shard 0/2         # e.g. one shard per GPU
    python refresh_histories.py --methods L1 --force
    python refresh_histories.py --image tub_balls --flux 40   # one (image, flux) only
"""
import argparse
import gc
import os
import shutil
from dataclasses import replace

import torch
from tqdm.auto import tqdm

from evaluate_methods import FIDELITY, LOSS, METHODS, SAVE_DIR, jobs
from opt_functions.data_preparation import load_data, make_measurements
from opt_functions.Solver_functions.prior_config import get_denoiser, make_prior
from opt_functions.Solver_functions.pnp_utils import (compute_final_metrics, experiment_paths,
                                                      fidelity_and_steps, psnr, solve)

HISTORY_KEYS = ("cost", "step_norm", "iterate_norm", "criticality")


def up_to_date(res):
    m = res.get("metrics") or {}
    return res.get("cost_def") == "F_fp64" and all(k in m for k in HISTORY_KEYS)


def refresh(res, x, meas, prior, cfg, fidelity, algo):
    lam, sigma, n_iter = res["lambda"], res["sigma"], int(res["iter"])
    df, tau, L_safe = fidelity_and_steps(fidelity, meas, lam, prior, cfg)
    if res.get("stepsize") is not None and abs(tau - res["stepsize"]) > 1e-9 * abs(tau):
        raise RuntimeError(f"step {tau:.6e} != saved {res['stepsize']:.6e}: config changed?")
    recs, L_hist = solve(x, meas["y"], meas["physics"], prior, df, tau, sigma, lam,
                         [n_iter], algo, cfg, L_safe, log_cost=True)
    x_n, metrics = recs[n_iter]
    final = compute_final_metrics(x, x_n)
    res.update({
        "psnr_search": res["psnr"], "psnr": psnr(x_n, x.detach().float().cpu()),
        "ssim": final["ssim"], "microssim": final["microssim"],
        "x_rec": x_n, "metrics": metrics,
        "L_history": L_hist[:n_iter] if L_hist else None, "cost_def": "F_fp64",
    })
    return res


def main(shard, n_shards, plan_only, only, force, images=None, fluxes=None):
    methods = METHODS if only is None else {k: METHODS[k] for k in only}
    todo = []
    for image, flux, seed in jobs()[shard::n_shards]:
        if (images and image not in images) or (fluxes and flux not in fluxes):
            continue
        for method, m in methods.items():
            cfg = replace(m["cfg"], seed=seed)
            fidelity = m.get("fidelity", FIDELITY)
            *_, path = experiment_paths(cfg, image, flux, LOSS, fidelity, m["algo"])
            if not os.path.exists(path):
                continue
            if not force and up_to_date(torch.load(path, map_location="cpu", weights_only=False)):
                continue
            todo.append((image, flux, seed, method, cfg, fidelity, m["algo"], path))
    for t in todo:
        print(f"{t[0]:>15s} flux={t[1]:<3} seed={t[2]} {t[3]:<13s} {t[7]}")
    print(f"{len(todo)} files to update")
    if plan_only:
        return

    denoisers, key, x, meas = {}, None, None, None
    for image, flux, seed, method, cfg, fidelity, algo, path in tqdm(todo, desc="refresh"):
        if (image, flux, seed) != key:          # measurement shared by the methods
            key = (image, flux, seed)
            x = load_data(image, cfg)
            meas = make_measurements(x, flux, cfg)
        if method not in denoisers:
            denoisers[method] = get_denoiser(LOSS, cfg)
        prior = make_prior(denoisers[method], algo, cfg.is_3d)
        res = torch.load(path, map_location="cpu", weights_only=False)
        res = refresh(res, x, meas, prior, cfg, fidelity, algo)

        backup = os.path.join(SAVE_DIR, "_pre_fp64", os.path.relpath(path, SAVE_DIR))
        if not os.path.exists(backup):
            os.makedirs(os.path.dirname(backup), exist_ok=True)
            shutil.copy2(path, backup)
        tmp = path + ".tmp"
        torch.save(res, tmp)
        os.replace(tmp, path)
        tqdm.write(f"{image} flux={flux} {method}: PSNR {res['psnr_search']:.3f} -> "
                   f"{res['psnr']:.3f}, F[-1] = {res['metrics']['cost'][0][-1]:.6f}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th (image, flux, seed)")
    parser.add_argument("--plan", action="store_true", help="list the files only")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS))
    parser.add_argument("--force", action="store_true", help="also files already up to date")
    parser.add_argument("--image", nargs="+", help="only these images")
    parser.add_argument("--flux", nargs="+", type=int, help="only these fluxes")
    args = parser.parse_args()
    shard, n_shards = map(int, args.shard.split("/"))
    main(shard, n_shards, args.plan, args.methods, args.force, args.image, args.flux)

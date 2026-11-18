#%%
"""
3D data reconstructed with the parameters of the 2D searches: for every 3D image, flux and
method, one reconstruction (no search) with the sigma / lambda (DRUNet: + step, iteration
count) that the 2D adaptive search of evaluate_methods.py found for the same method and
flux on --params-from (default: tub).

3D: 2 axial planes, plane 0 out of focus, plane 1 in focus; the regularizer acts on plane 1
only and every metric is computed on plane 1 (prior_config.REG_PLANE).

    python run_3d_from_2d.py                                    # everything (defaults below)
    CUDA_VISIBLE_DEVICES=1 python run_3d_from_2d.py --fluxes 20 # one flux, on GPU 1
    python run_3d_from_2d.py --images tub_3D --methods WCRR TV_smooth --params-from tub_level
    python run_3d_from_2d.py --plan                             # what would run / is missing
    python run_3d_from_2d.py --plot                             # + a figure per run (as single_experiment)

Order: image -> flux (in the order given) -> method, i.e. a whole flux is done before the
next one. A (method, flux) whose 2D search is not finished yet is skipped (listed at the
end): run the script again later, finished runs are skipped too (the result file exists),
so only the new ones are computed.

Saved as single runs (single_experiment.py), so compare_results.py sees them as the method
on that 3D image: SAVE_DIR/single_<hash>/best_rec_<tag>.pth + summary_all.csv; plus a table
SAVE_DIR/3d_from_2d_<params-from>.csv with the 3D and 2D metrics side by side.
"""
import argparse
import csv
import os
import time

import torch

import evaluate_methods as em
import single_experiment as se
from opt_functions.data_preparation import load_data, make_measurements
from opt_functions.Solver_functions import pnp_utils
from opt_functions.Solver_functions.prior_config import REG_PLANE, get_denoiser
from opt_functions.Solver_functions.pnp_utils import experiment_paths, run_experiment

# TV (non-smooth, prox by FGP: the slowest) left out; add it with --methods
DEFAULT_METHODS = ["WCRR", "WCRR_ISM", "TV_smooth", "L1",
                   "WCRR_L2", "WCRR_ISM_L2", "TV_smooth_L2", "L1_L2"]
DEFAULT_FLUXES = [20, 5, 10, 40]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--images", nargs="+", default=se.IMAGES_3D, choices=se.IMAGES_3D)
    p.add_argument("--fluxes", nargs="+", type=float, default=DEFAULT_FLUXES)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=list(em.METHODS))
    p.add_argument("--params-from", default="tub", choices=em.IMAGES,
                   help="2D image whose search results give the parameters")
    p.add_argument("--iter", nargs="+", type=int, default=None,
                   help="iterations (default: the method's, N_ITER with early stopping)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=em.SAVE_DIR)
    p.add_argument("--overwrite", action="store_true", help="rerun finished runs")
    p.add_argument("--plan", action="store_true", help="only list what would run / is missing")
    p.add_argument("--plot", action="store_true", help="a figure per run (as single_experiment)")
    p.add_argument("--progress", action="store_true", help="per-iteration progress bar")
    p.add_argument("--stacked", action="store_true", help="25 separate detector operators")
    a = p.parse_args()
    pnp_utils.PROGRESS_BAR = a.progress
    fluxes = [int(f) if f == int(f) else f for f in a.fluxes]

    rows, missing, denoisers = [], [], {}
    t_all = time.time()
    for image in a.images:
        for flux in fluxes:
            x = meas = None                              # simulated once per (image, flux)
            for method in a.methods:
                cfg, algo, fid = se.method_config(method, a.seed, a.save_dir,
                                                  batched=not a.stacked, is_3d=True)
                try:
                    given, src = se.params_from_2d(method, a.params_from, flux, a.seed, cfg, algo)
                except SystemExit:
                    missing.append((image, flux, method))
                    continue
                if a.iter is not None:
                    given["iter"] = a.iter
                kw = dict(sigmas=given["sigma"], lambdas=given["lambda"], steps=given["step"],
                          iters=given["iter"])
                *_, tag, out = experiment_paths(cfg, image, flux, em.LOSS, fid, algo, **kw)
                pars = "  ".join(f"{k}={v[0]:.4g}" for k, v in given.items() if v is not None)
                done = os.path.exists(out) and not a.overwrite
                if a.plan:
                    print(f"{'done' if done else 'RUN '}  {image} flux {flux:<3g} {method:<13} {pars}")
                    continue
                if done:
                    res = torch.load(out, map_location="cpu", weights_only=False)
                else:
                    print(f"\n=== {image} flux {flux} {method} ({pars}, from the 2D search on "
                          f"{a.params_from})", flush=True)
                    if x is None:
                        x = load_data(image, cfg)
                        meas = make_measurements(x, flux, cfg)
                    if method not in denoisers:
                        denoisers[method] = get_denoiser(em.LOSS, cfg)
                    t0 = time.time()
                    res = run_experiment(cfg, image, flux, em.LOSS, fid, algo=algo, x=x, meas=meas,
                                         denoiser=denoisers[method], overwrite=a.overwrite,
                                         verbose=False, **kw)
                    if res is None:
                        res = torch.load(out, map_location="cpu", weights_only=False)
                    print(f"    3D plane {REG_PLANE}: PSNR {res['psnr']:.2f}  SSIM {res['ssim']:.4f}  "
                          f"({res['iter_run']} it, {(time.time() - t0) / 60:.1f} min)   |   2D "
                          f"{a.params_from}: PSNR {src['psnr']:.2f}  SSIM {src['ssim']:.4f}", flush=True)
                    if a.plot and res["x_rec"] is not None:
                        png = os.path.join(os.path.dirname(out), f"{tag}.png")
                        se.plot_result(res, None, meas["sum_y"],
                                       f"{method} | {image} | flux {flux:g} | plane {REG_PLANE} "
                                       f"(in focus) | parameters of the 2D search on {a.params_from}",
                                       png)
                        print(f"    figure: {png}")
                rows.append(dict(image=image, flux=flux, method=method, fidelity=fid, params=pars,
                                 psnr_3d=res["psnr"], ssim_3d=res["ssim"],
                                 microssim_3d=res["microssim"], iter_run=res["iter_run"],
                                 psnr_2d=src["psnr"], ssim_2d=src["ssim"],
                                 microssim_2d=src["microssim"], run_id=res["run_id"]))

    if a.plan:
        print(f"\n{len(missing)} skipped: no 2D result yet")
    if rows:
        path = os.path.join(a.save_dir, f"3d_from_2d_{a.params_from}.csv")
        old = []
        if os.path.exists(path):                 # keep the rows of other runs of the script
            keys = {(r["image"], r["flux"], r["method"]) for r in rows}
            old = [r for r in csv.DictReader(open(path))
                   if (r["image"], float(r["flux"]), r["method"]) not in
                   {(i, float(f), m) for i, f, m in keys}]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(old + rows)
        print(f"\n{'image':<11}{'flux':>5}  {'method':<13}{'PSNR 3D':>9}{'PSNR 2D':>9}"
              f"{'SSIM 3D':>9}{'SSIM 2D':>9}{'iters':>7}")
        for r in rows:
            print(f"{r['image']:<11}{r['flux']:>5g}  {r['method']:<13}{r['psnr_3d']:>9.2f}"
                  f"{r['psnr_2d']:>9.2f}{r['ssim_3d']:>9.4f}{r['ssim_2d']:>9.4f}{r['iter_run']:>7}")
        print(f"\ntable: {path}   ({(time.time() - t_all) / 60:.1f} min)")
    if missing and not a.plan:
        print(f"\nskipped (2D search on {a.params_from} not finished yet), rerun the script later:")
        for image, flux, method in missing:
            print(f"   {image} flux {flux:g} {method}")


if __name__ == "__main__":
    main()

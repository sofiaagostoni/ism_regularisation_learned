"""
Print the functional F(x_k) = f + lam R saved per iteration by evaluate_methods.py.

    python print_functional.py                          # every result in Results/evaluation
    python print_functional.py --image tub --flux 10    # filter
    python print_functional.py --method WCRR L1_MD      # filter on prior / prior_algo
    python print_functional.py --iters 1 10 100 1000 2000

F* = min over the saved iterates; F - F* is relative to it.
Files saved before the float64 fix (no "cost_def") are marked OLD: there the cost is the
float32 smooth part (for L1 / PROX_GD the data term only).
"""
import argparse
import glob
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="Results/evaluation")
parser.add_argument("--image", nargs="+")
parser.add_argument("--flux", nargs="+", type=float)
parser.add_argument("--method", nargs="+", help="prior (e.g. WCRR) or prior_algo (e.g. WCRR_MD_BURG)")
parser.add_argument("--iters", nargs="+", type=int, default=[1, 10, 100, 500, 1000, 2000])
args = parser.parse_args()

paths = sorted(glob.glob(os.path.join(args.dir, "**", "best_rec_*.pth"), recursive=True))
for path in paths:
    r = torch.load(path, map_location="cpu", weights_only=False)
    method = f"{r['prior']}_{r['algo']}"
    if args.image and r["name"] not in args.image:
        continue
    if args.flux and float(r["flux"]) not in args.flux:
        continue
    if args.method and r["prior"] not in args.method and method not in args.method:
        continue
    cost = (r.get("metrics") or {}).get("cost")
    if not cost or not len(cost[0]):
        print(f"{r['name']} flux={r['flux']} {method}: no cost history")
        continue

    F = torch.tensor(cost[0], dtype=torch.float64)
    F_star = F.min()
    old = "" if r.get("cost_def") == "F_fp64" else "  [OLD: float32 smooth part]"
    sigma = f"{r['sigma']:.3g}" if r["sigma"] is not None else "-"
    print(f"\n{r['name']}  flux={r['flux']}  {method} ({r['fidelity']})  "
          f"lambda={r['lambda']:.3g} sigma={sigma}  PSNR={r['psnr']:.2f}{old}")
    print(f"  {'k':>6}  {'F(x_k)':>20}  {'F - F*':>12}  {'(F - F*)/|F*|':>14}")
    for k in [k for k in args.iters if 1 <= k <= len(F)]:
        Fk = F[k - 1]                           # cost[0][k-1] = F after k iterations
        print(f"  {k:>6}  {Fk:>20.10f}  {Fk - F_star:>12.4e}  {(Fk - F_star) / F_star.abs():>14.4e}")
    n_up = int((F.diff() > 0).sum())
    print(f"  F* = {F_star:.10f} at k = {int(F.argmin()) + 1} / {len(F)};  increases: {n_up}")

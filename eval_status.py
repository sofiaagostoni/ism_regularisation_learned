#%%
"""
Status of evaluate_methods.py: every (image, flux, seed, method) of its current settings,
finished / running / not started, plus the evaluate_methods processes alive.

    python eval_status.py
    watch -n 60 python eval_status.py        # refresh every minute (Ctrl+C to quit)

Details of a running search: python search_progress.py
"""
import os
import subprocess
import time
from dataclasses import replace

import torch

from evaluate_methods import FIDELITY, LOSS, METHODS, jobs
from opt_functions.Solver_functions.pnp_utils import experiment_paths


def status_rows():
    rows = []
    for image, flux, seed in jobs():
        for method, m in METHODS.items():
            cfg = replace(m["cfg"], seed=seed)
            fidelity = m.get("fidelity", FIDELITY)          # MD methods: 'KL0'
            _, _, _, tag, out_path = experiment_paths(cfg, image, flux, LOSS, fidelity, m["algo"])
            partial = os.path.join(os.path.dirname(out_path), f"partial_{tag}.pth")
            if os.path.exists(out_path):
                res = torch.load(out_path, map_location="cpu", weights_only=False)
                n = len(res.get("evaluations") or [])
                it = res.get("iter_run") or res.get("iter")
                state, info = "done", f"PSNR {res['psnr']:.2f}  ({n} runs, {it} iterations)"
            elif os.path.exists(partial):
                ev = torch.load(partial, map_location="cpu", weights_only=False)["evaluations"]
                best = max((e["psnr"] for e in ev), default=float("nan"))
                age = (time.time() - os.path.getmtime(partial)) / 60
                state = "running" if age < 30 else "stalled?"
                info = f"{len(ev)} runs, best so far {best:.2f}, last update {age:.0f} min ago"
            else:
                state, info = "-", ""
            rows.append((image, flux, seed, method, state, info))
    return rows


if __name__ == "__main__":
    procs = subprocess.run(["ps", "-u", os.environ.get("USER", ""), "-o", "pid,etime,args"],
                           capture_output=True, text=True).stdout.splitlines()
    procs = [p for p in procs if "evaluate_methods.py" in p]
    print("processes:" if procs else "processes: none running")
    for p in procs:
        print("  " + p.strip())

    rows = status_rows()
    print(f"\n{'image':<16}{'flux':>5}{'seed':>5}  {'method':<14}{'status':<10}info")
    for image, flux, seed, method, state, info in rows:
        print(f"{image:<16}{flux:>5}{seed:>5}  {method:<14}{state:<10}{info}")

    print()
    for method in METHODS:
        s = [r[4] for r in rows if r[3] == method]
        print(f"{method:<14} {s.count('done'):>3} done, {s.count('running'):>2} running, "
              f"{s.count('stalled?'):>2} stalled?, {s.count('-'):>3} not started  (of {len(s)})")
    print("\n'stalled?': no update for 30 min (process stopped? restart it to resume)")

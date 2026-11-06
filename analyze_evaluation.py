#%%
"""
Statistics of the results of evaluate_methods.py.

    python analyze_evaluation.py [--ref WCRR] [--dir Results/evaluation]

1. Oracle (per-image tuned) metrics, one table per flux: image x method, best per image
   marked, and the mean rank of each method. The metrics of different images are not
   comparable (an easy image scores much higher), so they are never averaged together.
2. Paired comparison with a reference method, image by image:
   mean difference, 95% bootstrap CI over images, win rate, Wilcoxon signed-rank p.
3. Fixed parameters, leave-one-image-out: for each image the parameters are chosen
   from the *other* images only, which is what a method gives without the ground truth
   of the test image. The gap to the oracle shows how sensitive a method is to its
   parameters. Adaptive search: evaluation_loo*.csv (evaluate_methods.py --loo, median
   of the other images' best parameters); grid search: from the saved PSNR maps.
4. Warnings: best parameter on the border of its grid / at the expansion limit.

Noise realizations (seeds) are averaged per image first: images are the samples.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

METRICS = ["psnr", "ssim", "microssim"]
N_BOOT = 10000


def load(save_dir, loo=False):
    """Per-image tuned rows (loo=False) or leave-one-image-out rows (loo=True)."""
    files = [f for f in sorted(glob.glob(os.path.join(save_dir, "evaluation*.csv")))
             if ("_loo" in os.path.basename(f)) == loo]
    if not files:
        if loo:
            return None
        raise FileNotFoundError(f"no evaluation*.csv in {save_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(subset=["image", "flux", "seed", "method"], keep="last")


def per_image(df, metrics=METRICS):
    """Average over seeds: one value per (method, flux, image)."""
    return df.groupby(["method", "flux", "image"], as_index=False)[metrics].mean()


def bootstrap_ci(d, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


def paired_table(img, ref, metric):
    """Every method vs ref on the images both have: positive = better than ref."""
    rows = []
    for flux, g in img.groupby("flux"):
        wide = g.pivot(index="image", columns="method", values=metric)
        for method in wide.columns.drop(ref, errors="ignore"):
            pair = wide[[ref, method]].dropna()
            d = (pair[method] - pair[ref]).to_numpy()
            if len(d) == 0:
                continue
            lo, hi = bootstrap_ci(d) if len(d) > 1 else (np.nan, np.nan)
            p = wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else np.nan
            rows.append({"flux": flux, "method": method, "vs": ref, "n_images": len(d),
                         f"mean_diff_{metric}": d.mean(), "ci95_low": lo, "ci95_high": hi,
                         "win_rate": (d > 0).mean(), "wilcoxon_p": p})
    return pd.DataFrame(rows)


def leave_one_out(df):
    """PSNR with parameters fixed from the other images (per method, flux, seed)."""
    rows = []
    for (method, flux, seed), g in df.groupby(["method", "flux", "seed"]):
        if any(torch.load(r.path, map_location="cpu", weights_only=False)["psnr_map"] is None
               for r in g.itertuples()):
            continue                                            # adaptive search: no map
        maps = {r.image: torch.load(r.path, map_location="cpu", weights_only=False)["psnr_map"]
                for r in g.itertuples()}
        if len({m.shape for m in maps.values()}) > 1:
            print(f"skip LOO {method} flux={flux}: different grids across images")
            continue
        stack = torch.stack([maps[k] for k in maps])            # (n_images, *grid)
        for i, image in enumerate(maps):
            if len(maps) < 2:
                break
            others = torch.cat([stack[:i], stack[i + 1:]]).nanmean(0)
            others = torch.nan_to_num(others, nan=-float("inf"))
            k = int(others.flatten().argmax())
            fixed = float(stack[i].flatten()[k])
            oracle = float(torch.nan_to_num(stack[i], nan=-float("inf")).max())
            rows.append({"method": method, "flux": flux, "seed": seed, "image": image,
                         "psnr_fixed": fixed, "psnr_oracle": oracle,
                         "gap": oracle - fixed})
    return pd.DataFrame(rows)


def per_image_tables(img, metric):
    """Per flux: image x method table of one metric, * = best on that image, and a last
    row with the mean rank of each method over the images (1 = best)."""
    fmt = "{:.2f}" if "psnr" in metric else "{:.3f}"
    for flux, g in img.groupby("flux"):
        wide = g.pivot(index="image", columns="method", values=metric)
        rank = wide.rank(axis=1, ascending=False)
        order = rank.mean().sort_values().index
        wide, rank = wide[order], rank[order]
        best = wide.max(axis=1)
        shown = pd.DataFrame({m: ["-" if pd.isna(v) else fmt.format(v) + ("*" if v == b else " ")
                                  for v, b in zip(wide[m], best)] for m in wide},
                             index=wide.index)
        shown.loc["mean rank"] = [f"{v:.2f}" for v in rank.mean()]
        print(f"--- {metric} | flux {flux}   (* = best on that image) ---")
        print(shown.to_string(), "\n")


def mean_std(img, metrics):
    agg = img.groupby(["flux", "method"])[metrics].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        fmt = "{:.2f} ± {:.2f}" if "psnr" in m else "{:.3f} ± {:.3f}"   # dB vs [0, 1]
        out[m] = [fmt.format(a, b) for a, b in zip(agg[(m, "mean")], agg[(m, "std")].fillna(0))]
    return out


def main(save_dir, ref):
    pd.set_option("display.width", 200, "display.max_columns", 20)
    df = load(save_dir)
    img = per_image(df)
    print(f"{len(df)} rows: {df.image.nunique()} images, fluxes {sorted(df.flux.unique().tolist())}, "
          f"seeds {sorted(df.seed.unique().tolist())}, methods {sorted(df.method.unique())}\n")

    print("=== Oracle (per-image tuned), per image ===")
    for metric in METRICS:
        per_image_tables(img, metric)

    if ref not in set(df.method):
        ref = sorted(df.method.unique())[0]
    for metric in METRICS:
        t = paired_table(img, ref, metric)
        if len(t):
            print(f"=== Paired {metric} vs {ref} (positive = better than {ref}) ===")
            print(t.round(4).to_string(index=False), "\n")

    loo_runs = load(save_dir, loo=True)
    if loo_runs is not None:
        loo = (per_image(loo_runs, ["psnr"]).rename(columns={"psnr": "psnr_fixed"})
               .merge(img[["method", "flux", "image", "psnr"]]
                      .rename(columns={"psnr": "psnr_oracle"}), on=["method", "flux", "image"]))
        loo["gap"] = loo.psnr_oracle - loo.psnr_fixed
    else:
        loo = leave_one_out(df)
    if len(loo):
        loo_img = loo.groupby(["method", "flux", "image"], as_index=False)[
            ["psnr_fixed", "psnr_oracle", "gap"]].mean()
        print("=== Fixed parameters (leave-one-image-out) vs oracle, PSNR ===")
        print(mean_std(loo_img, ["psnr_fixed", "psnr_oracle"])
              .join(loo_img.groupby(["flux", "method"]).gap.mean().round(2)), "\n")
        t = paired_table(loo_img.rename(columns={"psnr_fixed": "psnr"}), ref, "psnr")
        if len(t):
            print(f"=== Paired fixed-parameter PSNR vs {ref} ===")
            print(t.round(4).to_string(index=False), "\n")

    edge = df[df.lambda_on_edge | df.sigma_on_edge]
    if len(edge):
        print("=== WARNING: best parameter on the grid border (widen the grid) ===")
        print(edge[["image", "flux", "seed", "method", "lambda", "sigma",
                    "lambda_on_edge", "sigma_on_edge"]].to_string(index=False), "\n")

    if df.image.nunique() < 10:
        print(f"Note: only {df.image.nunique()} images; the CIs are wide and the Wilcoxon "
              f"p-value cannot go below ~{2 / 2 ** df.image.nunique():.3f}.")
    return df, img, loo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="Results/evaluation")
    parser.add_argument("--ref", default="WCRR", help="method the others are compared to")
    args = parser.parse_args()
    main(args.dir, args.ref)

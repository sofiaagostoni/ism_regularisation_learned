#!/usr/bin/env python3
"""Explore an HDF5 dataset: structure, stats, and a grid of sample images.

Works with both common layouts:
  - one big dataset of samples, e.g. "images" with shape (N, 1, 40, 40)
  - one small dataset per sample, e.g. keys "0", "1", ... each (1, 40, 40)
    (the layout used by the WCRR training code)

Usage:
    python explore_h5.py                          # data/train.h5: structure + stats + 16 images
    python explore_h5.py data/validation.h5
    python explore_h5.py --k 36 --out grid.png    # 36 random images
    python explore_h5.py --key images             # choose a dataset/group explicitly
    python explore_h5.py --print 0                # also print the pixel values of sample 0
    python explore_h5.py --no-plot                # text only
"""
import argparse
from collections import Counter

import h5py
import numpy as np

MAX_LIST = 10  # list at most this many members per group, summarise the rest


def fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def print_attrs(obj, indent):
    for k, v in obj.attrs.items():
        print(f"{indent}  @{k} = {v!r}"[:200])


def describe(group, indent=""):
    """Print the file tree; groups with many members are summarised."""
    keys = list(group.keys())
    for name in keys[:MAX_LIST]:
        obj = group[name]
        if isinstance(obj, h5py.Group):
            print(f"{indent}[group] {name}/  ({len(obj)} members)")
            print_attrs(obj, indent)
            describe(obj, indent + "  ")
        else:
            info = (f"shape={obj.shape}  dtype={obj.dtype}  "
                    f"size={fmt_bytes(obj.size * obj.dtype.itemsize)}")
            if obj.chunks:
                info += f"  chunks={obj.chunks}"
            if obj.compression:
                info += f"  compression={obj.compression}"
            print(f"{indent}[data]  {name}  {info}")
            print_attrs(obj, indent)
    if len(keys) > MAX_LIST:
        kinds = Counter()
        for name in keys[MAX_LIST:]:
            obj = group[name]
            kinds["groups" if isinstance(obj, h5py.Group) else f"datasets {obj.shape} {obj.dtype}"] += 1
        summary = ", ".join(f"{c} {k}" for k, c in kinds.items())
        print(f"{indent}... and {len(keys) - MAX_LIST} more: {summary}")


def natural_key(k):
    return (0, int(k), "") if k.isdigit() else (1, 0, k)


class Samples:
    """Uniform access to individual samples, whatever the file layout."""

    def __init__(self, f, key=None):
        obj = f[key] if key else f
        # descend through wrapper groups that contain a single subgroup and no data
        while isinstance(obj, h5py.Group):
            members = list(obj.values())
            if len(members) == 1 and isinstance(members[0], h5py.Group):
                obj = members[0]
            else:
                break

        if isinstance(obj, h5py.Group):
            names = [k for k in obj.keys() if isinstance(obj[k], h5py.Dataset)]
            if not names:
                raise ValueError("No datasets found here; pick one with --key")
            if len(names) == 1 and obj[names[0]].ndim >= 3:
                obj = obj[names[0]]  # a single big array of samples
            else:
                self.group, self.keys, self.ds = obj, sorted(names, key=natural_key), None
                self.desc = f"{len(self.keys)} datasets in '{obj.name}', one per sample"
                return

        self.ds, self.keys = obj, None
        if obj.ndim <= 2:  # a single image or signal
            self.desc = f"dataset '{obj.name}' (single sample)"
        else:
            self.desc = f"dataset '{obj.name}', {obj.shape[0]} samples along axis 0"

    def __len__(self):
        if self.keys is not None:
            return len(self.keys)
        return 1 if self.ds.ndim <= 2 else self.ds.shape[0]

    def __getitem__(self, i):
        if self.keys is not None:
            return np.asarray(self.group[self.keys[i]][()])
        return np.asarray(self.ds[()] if self.ds.ndim <= 2 else self.ds[i])

    def name(self, i):
        if self.keys is None:
            return str(i)
        z = self.group[self.keys[i]].attrs.get("scale")   # zoom salvato da export_wcrr_h5.py
        return self.keys[i] if z is None else f"{self.keys[i]} z={float(z):.2f}"


def stats(samples, n, rng):
    """Stats over n random samples."""
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    shapes, dtypes, vals = Counter(), Counter(), []
    for i in idx:
        x = samples[i]
        shapes[x.shape] += 1
        dtypes[str(x.dtype)] += 1
        if np.iscomplexobj(x):
            x = np.abs(x)
        vals.append(x.astype(np.float64).ravel())
    v = np.concatenate(vals)
    finite = np.isfinite(v)
    f = v[finite]

    print(f"Samples: {len(samples)}  ({samples.desc})")
    print(f"Stats on {len(idx)} random samples:")
    print(f"  shapes: {dict(shapes)}")
    print(f"  dtypes: {dict(dtypes)}")
    if f.size:
        p1, p50, p99 = np.percentile(f, [1, 50, 99])
        print(f"  min={f.min():.4g}  max={f.max():.4g}  mean={f.mean():.4g}  std={f.std():.4g}")
        print(f"  percentiles 1/50/99: {p1:.4g} / {p50:.4g} / {p99:.4g}")
    print(f"  NaN/inf values: {(~finite).sum()}")
    if f.size and (f.min() < -0.05 or f.max() > 1.05):
        print("  NOTE: values are outside [0, 1]; the WCRR training code assumes images in [0, 1].")


def to_image(item):
    """Turn one sample into something imshow can display."""
    item = np.asarray(item)
    if np.iscomplexobj(item):
        item = np.abs(item)
    item = np.squeeze(item)
    if item.ndim == 3:
        if item.shape[-1] in (3, 4):        # H, W, C
            pass
        elif item.shape[0] in (3, 4):       # C, H, W
            item = np.moveaxis(item, 0, -1)
        else:                               # volume: take middle slice
            item = item[item.shape[0] // 2]
    while item.ndim > 3:
        item = item[item.shape[0] // 2]
    if item.ndim == 3:                      # normalise RGB to [0, 1]
        item = (item - item.min()) / (np.ptp(item) + 1e-12)
    return item


def plot_grid(samples, k, out, rng, per_image):
    import matplotlib
    matplotlib.use("Agg")  # works on clusters without a display
    import matplotlib.pyplot as plt

    idx = np.sort(rng.choice(len(samples), size=min(k, len(samples)), replace=False))
    imgs = [to_image(samples[i]) for i in idx]
    gray = [im for im in imgs if im.ndim == 2]
    vmin = min(im.min() for im in gray) if gray else None
    vmax = max(im.max() for im in gray) if gray else None

    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows),
                             squeeze=False, constrained_layout=True)
    last = None
    for ax, i, img in zip(axes.flat, idx, imgs):
        if img.ndim == 1:
            ax.plot(img)
        elif img.ndim == 2:
            last = ax.imshow(img, cmap="hot",
                             vmin=None if per_image else vmin,
                             vmax=None if per_image else vmax)
        else:
            ax.imshow(img)
        ax.set_title(f"#{samples.name(i)}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(imgs):]:
        ax.axis("off")
    if last is not None and not per_image:
        fig.colorbar(last, ax=axes, fraction=0.025, label="shared intensity scale")
    fig.suptitle(f"{len(imgs)} random samples of {len(samples)}"
                 + ("  (each image scaled separately)" if per_image else ""))
    fig.savefig(out, dpi=120)
    print(f"Saved {len(imgs)} images to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", default="data/train.h5")
    p.add_argument("--key", help="dataset or group holding the samples (auto-detected if omitted)")
    p.add_argument("--n", type=int, default=500, help="random samples used for stats (default 500)")
    p.add_argument("--k", type=int, default=16, help="number of images to plot (default 16)")
    p.add_argument("--out", default="h5_preview.png", help="output image file")
    p.add_argument("--per-image", action="store_true",
                   help="scale each image separately instead of a shared intensity scale")
    p.add_argument("--print", type=int, metavar="I", dest="print_idx",
                   help="print the pixel values of sample I")
    p.add_argument("--no-plot", action="store_true", help="skip saving images")
    p.add_argument("--seed", type=int, default=0, help="random seed for sample selection")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.path, "r") as f:
        print(f"File: {args.path}")
        print_attrs(f, "")
        print("\nStructure:")
        describe(f, "  ")
        print()

        samples = Samples(f, args.key)
        stats(samples, args.n, rng)

        if args.print_idx is not None:
            x = np.squeeze(samples[args.print_idx])
            print(f"\nSample #{samples.name(args.print_idx)}  shape={x.shape}  dtype={x.dtype}:")
            print(np.array2string(x, precision=3, suppress_small=True,
                                  max_line_width=400, threshold=10_000))

        if not args.no_plot:
            print()
            plot_grid(samples, args.k, args.out, rng, args.per_image)
"""
Coarse-to-fine maximization of an expensive score over 1 or 2 parameters, in log10 space.

    1. coarse scan: n_points per axis over the initial range (product grid in 2D)
    2. expansion:   while the best point is on the border of the scanned range, add one
                    more point (row in 2D) beyond that border, up to max_expand decades
                    past the initial range. A best point on the final border is reported
                    in hit_bound: the range should be widened.
    3. refinement:  1D: golden-section search inside the bracket around the best point
                    2D: pattern search, a 3 x 3 stencil around the best point whose
                        spacing is halved at every level (follows diagonal valleys,
                        e.g. the sigma-lambda ridge of WCRR, better than searching one
                        axis at a time)
    Both stop when the resolution is below tol decades.

The score is assumed unimodal near its maximum only; the coarse scan over a wide range
is what protects against local maxima. A point replaces the current best only if it is
better by more than min_gain, so plateaus (e.g. lambda so small that R does nothing)
do not drag the search along. Every point is evaluated at most once (cache),
so a search restarted with the cache of a previous run replays it without cost.
"""
import itertools
import math

INV_PHI = (math.sqrt(5) - 1) / 2          # 0.618...
NDIGITS = 6                               # rounding of log10 values in the cache keys


def coarse_to_fine(score, ranges, n_points=5, tol=0.05, max_expand=3.0, min_gain=1e-3,
                   cache=None, on_eval=None):
    """
    score      : function {axis: log10 value} -> float to maximize (-inf = failed)
    ranges     : {axis: (log10 lo, log10 hi)}, 1 or 2 axes
    cache      : dict {tuple of log10 values: score}, filled in place (resume)
    on_eval    : called as on_eval(point, value) after every new evaluation
    Returns {"best": {axis: log10 value}, "score", "hit_bound": {axis: bool},
             "explored": {axis: (min, max)}, "n_evals": new evaluations}.
    """
    axes = list(ranges)
    if len(axes) not in (1, 2):
        raise ValueError(f"1 or 2 axes supported, got {axes}")
    cache = {} if cache is None else cache
    n_new = 0

    def f(point):
        nonlocal n_new
        key = tuple(round(float(point[a]), NDIGITS) for a in axes)
        if key not in cache:
            value = float(score(dict(zip(axes, key))))
            cache[key] = value if math.isfinite(value) else -math.inf
            n_new += 1
            if on_eval is not None:
                on_eval(dict(zip(axes, key)), cache[key])
        return cache[key]

    def best_of(points, best=None):
        """Best of points; the current best is kept unless beaten by more than min_gain."""
        for p in points:
            if best is None or f(p) > f(best) + min_gain:
                best = p
        return best

    # 1. coarse scan
    lines = {a: [lo + (hi - lo) * i / (n_points - 1) for i in range(n_points)]
             for a, (lo, hi) in ranges.items()}
    spacing = {a: lines[a][1] - lines[a][0] for a in axes}

    def grid_points():
        return [dict(zip(axes, v)) for v in itertools.product(*(lines[a] for a in axes))]

    best = best_of(grid_points())

    # 2. expansion
    limits = {a: (lo - max_expand, hi + max_expand) for a, (lo, hi) in ranges.items()}
    hit_bound = {a: False for a in axes}
    moved = True
    while moved:
        moved = False
        for a in axes:
            for side in (0, -1):
                if not math.isclose(best[a], lines[a][side]):
                    continue
                new = lines[a][side] + (-1 if side == 0 else 1) * spacing[a]
                if (new < limits[a][0] - 1e-9) or (new > limits[a][1] + 1e-9):
                    hit_bound[a] = True
                    continue
                lines[a] = [new] + lines[a] if side == 0 else lines[a] + [new]
                best = best_of(grid_points(), best)
                moved = True

    # 3. refinement
    if len(axes) == 1:
        best = _golden_refine(f, axes[0], lines[axes[0]], best, tol, min_gain)
    else:
        step = dict(spacing)
        while max(step.values()) > tol:
            step = {a: s / 2 for a, s in step.items()}
            stencil = [{a: best[a] + d * step[a] for a, d in zip(axes, ds)}
                       for ds in itertools.product((-1, 0, 1), repeat=2)]
            best = best_of(stencil, best)

    explored = {a: (min(k[i] for k in cache), max(k[i] for k in cache))
                for i, a in enumerate(axes)}
    return {"best": best, "score": f(best), "hit_bound": hit_bound,
            "explored": explored, "n_evals": n_new}


def _golden_refine(f, a, line, best, tol, min_gain):
    """Golden-section search on the bracket (left neighbour, best, right neighbour)."""
    i = min(range(len(line)), key=lambda j: abs(line[j] - best[a]))
    if i == 0 or i == len(line) - 1:      # best on the final border: nothing to bracket
        return best
    lo, mid, hi = line[i - 1], line[i], line[i + 1]
    fm = f({a: mid})
    while hi - lo > 2 * tol:
        # new point in the larger of the two segments, at the golden ratio from mid
        if hi - mid > mid - lo:
            x = mid + (1 - INV_PHI) * (hi - mid)
            fx = f({a: x})
            if fx > fm + min_gain:
                lo, mid, fm = mid, x, fx
            else:
                hi = x
        else:
            x = mid - (1 - INV_PHI) * (mid - lo)
            fx = f({a: x})
            if fx > fm + min_gain:
                hi, mid, fm = mid, x, fx
            else:
                lo = x
    return {a: mid}

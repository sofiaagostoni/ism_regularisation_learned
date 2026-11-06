"""
Custom optimization iterations and the model builder.

Custom iterations (they store intermediate iterates, so ONE run of max(iters)
iterations gives the result for every iteration count of the grid):
    'PGD'      PnP proximal gradient:  x <- D(x - tau grad f(x))
    'PNP_MD'   PnP mirror descent (Burg entropy), then D
    'PROJ_GD'  explicit priors, x >= 0, optional adaptive backtracking:
    'PROX_GD'  gradient step on R if smooth (WCRR, TV_smooth), prox step on R
               if not (L1). The two names are the same iteration.
    'MD_BURG'  (MD ADDED) explicit priors, mirror descent with the Burg entropy and
               backtracking, from ism_regularisation. Use with fidelity 'KL0'.
Any other algo name is passed to deepinv as a built-in iteration.

PROJ_GD / PROX_GD / MD_BURG with log_cost=True (default) return as "cost" the full
functional F = f + lam R at the new iterate, in float64; deepinv stores it per iteration
in metrics["cost"]. This costs ~45% more time per iteration; log_cost=False returns the
float32 smooth part S instead (= F except for L1, where S = f).
With log_cost=True they also record, on the actual (unclamped) iterates x_k, in
iteration.history (copied by pnp_utils.solve into metrics):
    step_norm[k]    = ||x_{k+1} - x_k||                  finite length: sum < inf
    iterate_norm[k] = ||x_{k+1}||                        boundedness
    criticality[k]  = ||x_k - max(0, x_k - grad F(x_k))|| KKT residual of
                      min F s.t. x >= 0 (0 iff x_k is critical; for L1, grad F
                      includes lam, the gradient of lam ||x||_1 on x >= 0)
Stopping (stop_tol not None): stop once ||x_{k+1} - x_k|| / ||x_k|| < stop_tol for
stop_patience consecutive iterations (checked on the unclamped iterates; see check_stop).
"""
import copy
import weakref
from functools import partial

import torch
from deepinv.utils import TensorList
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.optim_iterators import OptimIterator
from deepinv.optim.bregman import BurgEntropy
from tqdm.auto import tqdm


class SnapshotIterator(OptimIterator):
    """Stores the iterate after n iterations for every n in snap_iters.
    The iterate after n iterations is the input of call n."""

    def __init__(self, snap_iters=(), **kwargs):
        super().__init__(**kwargs)
        self.snap_iters = {int(n) for n in snap_iters}
        self.snapshots = {}
        self.n_calls = 0

    def _snap(self, x):
        if self.n_calls in self.snap_iters:
            self.snapshots[self.n_calls] = x.detach().clamp(0, 1).cpu().clone()
        self.n_calls += 1


class PnPPGDIteration(SnapshotIterator):
    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs):
        x_prev = X["est"][0]
        self._snap(x_prev)
        z = x_prev - cur_params["stepsize"] * cur_data_fidelity.grad(x_prev, y, physics)
        x = cur_prior.prox(z, cur_params["g_param"])
        return {"est": (x,), "cost": None}


class PnPMDIteration(SnapshotIterator):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.bregman = BurgEntropy()
        self.eps = eps                      # Burg entropy needs x > 0

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs):
        self._snap(X["est"][0])
        x_prev = X["est"][0].clamp_min(self.eps)
        grad_f = cur_data_fidelity.grad(x_prev, y, physics)
        z = self.bregman.grad_conj(self.bregman.grad(x_prev) - cur_params["stepsize"] * grad_f)
        x = cur_prior.prox(z, cur_params["g_param"]).clamp_min(self.eps)
        return {"est": (x,), "cost": None}


# Full functional F = f + lam R in float64, for the convergence history metrics["cost"].
# float32 is not enough: F ~ 1e5 is a sum over ~1e6 pixels (resolution ~1e-2), and the
# WCRR spline integral loses ~1% to rounding.
_PRIORS_FP64 = weakref.WeakKeyDictionary()      # prior -> float64 copy, built once


def _to_float64(module):
    """float64 copy of a module, including tensors stored as plain attributes
    (e.g. the WCRR spline coefficients)."""
    module = copy.deepcopy(module)
    for m in module.modules():
        for name, val in list(vars(m).items()):
            if (torch.is_tensor(val) and val.is_floating_point()
                    and not isinstance(val, torch.nn.Parameter)):
                setattr(m, name, val.double())
    return module.double()


def full_cost_fp64(x, data_fidelity, prior, lam, sigma, y, physics):
    """F(x) = f(x) + lam R(x) in float64, shape (batch,). R(x) = ||x||_1 for L1."""
    if prior not in _PRIORS_FP64:
        _PRIORS_FP64[prior] = _to_float64(prior)
    y64 = (TensorList([t.double() for t in y]) if isinstance(y, TensorList)
           else y.double())
    x64 = x.double()
    with torch.no_grad():
        return data_fidelity(x64, y64, physics) + lam * _PRIORS_FP64[prior](x64, sigma)


class ProxGDIteration(SnapshotIterator):
    """
    Proximal gradient descent on F = f + lam*R over x >= 0, for explicit priors.

    How R is used depends on prior.smooth_prior:
        smooth R (WCRR, TV_smooth):  x+ = max(0, x - tau * (grad f + lam grad R))
        non-smooth R (L1):           x+ = max(0, prox_{tau lam R}(x - tau * grad f))
    'PROJ_GD' and 'PROX_GD' are the same iteration; the prior decides.

    backtracking=False: tau fixed.
    backtracking=True : each iteration first tries a larger step,
        L = 1 / min(tau_k / delta, 1 / s),
    then multiplies L by eta until, with S the smooth part of F,
        S(x+) <= S(x) + <grad S(x), x+ - x> + L/2 ||x+ - x||^2,   tau = 1 / L.
    After max_bt loops, or once L >= L_max, the safe step 1/L_max is taken.
    S(x+) is kept and reused as S(x) by the next iteration.
    The returned "cost" is the full F(x+) = f + lam R in float64 (full_cost_fp64),
    also for L1, where S = f only; with log_cost=False it is S(x+) (cheaper).
    """

    def __init__(self, backtracking=False, s=1.0, eta=2.0, delta=0.9,
                 L_max=float("inf"), max_bt=50, log_cost=True, stop_tol=None,
                 stop_patience=10, **kwargs):
        super().__init__(**kwargs)
        self.log_cost = log_cost
        self.stop_tol, self.stop_patience = stop_tol, stop_patience
        self._n_small = 0                   # consecutive iterations with a small step
        self.backtracking = backtracking
        self.s, self.eta, self.delta = s, eta, delta
        self.L_max, self.max_bt = L_max, max_bt
        self.tau_k = None                   # last accepted step
        self.L_history = []
        self.n_backtracks = []
        self._S_last = None                 # (iterate, S(iterate)) of the last call
        self.history = {"step_norm": [], "iterate_norm": [], "criticality": []}

    def _S_prev(self, x_in, S_fn):
        """S at the input iterate, reused from the last call when x_in is its output."""
        if self._S_last is not None and self._S_last[0] is x_in:
            return self._S_last[1]
        return float(S_fn().sum())

    def check_stop(self, it, X_prev, X):
        """deepinv check_conv_fn: True after stop_patience consecutive iterations with
        ||x_{k+1} - x_k|| / ||x_k|| < stop_tol (computed in _output)."""
        return self.stop_tol is not None and self._n_small >= self.stop_patience

    @staticmethod
    def _norm(v):
        return float(v.double().norm())

    def _criticality(self, x, grad_F):
        """||x - P_{x>=0}(x - grad F(x))||, only computed when log_cost."""
        return self._norm(x - (x - grad_F).clamp_min(0)) if self.log_cost else None

    def _output(self, x, S_new, x_in, crit, cur_data_fidelity, cur_prior, lam, sigma, y,
                physics):
        """Remember S(x) for the next call; "cost" = F(x) in float64 (log_cost) or S(x).
        x_in: input iterate of this call, crit: criticality at x_in."""
        self._S_last = (x, float(S_new.sum()))
        if self.stop_tol is not None:
            rel = float((x - x_in).norm() / x_in.norm().clamp_min(1e-12))
            self._n_small = self._n_small + 1 if rel < self.stop_tol else 0
        if not self.log_cost:
            return {"est": (x,), "cost": S_new}
        self.history["step_norm"].append(self._norm(x - x_in))
        self.history["iterate_norm"].append(self._norm(x))
        self.history["criticality"].append(crit)
        F = full_cost_fp64(x, cur_data_fidelity, cur_prior, lam, sigma, y, physics)
        return {"est": (x,), "cost": F}

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs):
        x_prev = X["est"][0]
        self._snap(x_prev)
        lam, tau, sigma = cur_params["lambda"], cur_params["stepsize"], cur_params["g_param"]
        smooth_R = getattr(cur_prior, "smooth_prior", True)

        def S(x):                       # smooth part of F
            out = cur_data_fidelity(x, y, physics)
            return out + lam * cur_prior(x, sigma) if smooth_R else out

        def step(tau):                  # one (proximal) gradient step, then x >= 0
            z = x_prev - tau * g
            if not smooth_R:
                z = cur_prior.prox(z, sigma, gamma=tau * lam)
            return z.clamp_min(0)

        g = cur_data_fidelity.grad(x_prev, y, physics)
        if smooth_R:
            g = g + lam * cur_prior.grad(x_prev, sigma)
        crit = self._criticality(x_prev, g if smooth_R else g + lam)

        out = partial(self._output, x_in=x_prev, crit=crit, cur_data_fidelity=cur_data_fidelity,
                      cur_prior=cur_prior, lam=lam, sigma=sigma, y=y, physics=physics)
        if not self.backtracking:
            x = step(tau)
            return out(x, S(x))

        S_prev = self._S_prev(x_prev, lambda: S(x_prev))

        if self.tau_k is None:
            self.tau_k = tau
        L = 1.0 / min(self.tau_k / self.delta, 1.0 / self.s)
        n_bt = 0
        while True:
            x = step(1.0 / L)
            S_new = S(x)
            d = x - x_prev
            bound = S_prev + float((g * d).sum()) + 0.5 * L * float((d * d).sum())
            if float(S_new.sum()) <= bound:          # NaN fails the test -> smaller step
                break
            L *= self.eta
            n_bt += 1
            if L >= self.L_max or n_bt > self.max_bt:
                L = self.L_max
                x = step(1.0 / L)
                S_new = S(x)
                break

        self.tau_k = 1.0 / L
        self.L_history.append(L)
        self.n_backtracks.append(n_bt)
        return out(x, S_new)


# >>>>>>>>>> MD ADDED >>>>>>>>>>
def bregman_burg(x, u, eps=1e-8):
    """Bregman distance of the Burg entropy h(x) = -sum log x:
    D_h(x, u) = sum( -log x + log u + (x - u) / u ).  (Bregman_h of ism_regularisation)"""
    return (-torch.log(x + eps) + torch.log(u + eps) + (x - u) / (u + eps)).sum()


class MirrorDescentIteration(ProxGDIteration):
    """
    Mirror descent with the Burg entropy for explicit priors, ported from
    ism_regularisation (Pgd_Backtracking, algorithm 'md'). Use with fidelity 'KL0'.

        x+ = x / (1 + tau * x * g),
        g  = grad f + lam grad R    smooth R (WCRR, TV_smooth)
        g  = grad f + lam           L1: on x > 0, ||x||_1 = sum x, gradient = 1

    Backtracking as in ism_regularisation: L is multiplied by eta until
        S(x+) <= S(x) - (0.8 / tau) D_h(x+, x),       tau = 1 / L,
    S = f + lam R (also for L1; the original code used S = f for L1).
    Same candidate step (tau_k / delta, capped at 1 / s) and fallback (step 1/L_max
    after max_bt loops or when L >= L_max) as ProxGDIteration.
    L_max: from fidelity_and_steps, = ||y||_1 (+ lam * Lip(grad R)) for 'KL0'.
    """

    def __init__(self, x_eps=1e-8, armijo=0.8, **kwargs):
        super().__init__(**kwargs)
        self.x_eps = x_eps              # x must stay > 0 (Burg entropy, multiplicative step)
        self.armijo = armijo

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs):
        x_in = X["est"][0]
        self._snap(x_in)
        x_prev = x_in.clamp_min(self.x_eps)          # adjoint_y init can contain zeros
        lam, tau, sigma = cur_params["lambda"], cur_params["stepsize"], cur_params["g_param"]
        smooth_R = getattr(cur_prior, "smooth_prior", True)

        def S(x):
            # MD: R is always in the test. For L1 the original code tested f only, which
            # forced the fallback step 1/L_max often; on x > 0, ||x||_1 = sum x is linear.
            return cur_data_fidelity(x, y, physics) + lam * cur_prior(x, sigma)

        g = cur_data_fidelity.grad(x_prev, y, physics)
        g = g + (lam * cur_prior.grad(x_prev, sigma) if smooth_R else lam)

        def step(tau):
            return x_prev / (1 + tau * x_prev * g)

        crit = self._criticality(x_prev, g)
        out = partial(self._output, x_in=x_in, crit=crit, cur_data_fidelity=cur_data_fidelity,
                      cur_prior=cur_prior, lam=lam, sigma=sigma, y=y, physics=physics)
        if not self.backtracking:
            x = step(tau)
            return out(x, S(x))

        S_prev = self._S_prev(x_in, lambda: S(x_prev))

        if self.tau_k is None:
            self.tau_k = tau
        L = 1.0 / min(self.tau_k / self.delta, 1.0 / self.s)
        n_bt = 0
        while True:
            x = step(1.0 / L)
            S_new = S(x)
            bound = S_prev - self.armijo * L * float(bregman_burg(x, x_prev))
            if float(S_new.sum()) <= bound:          # NaN / x <= 0 fails -> smaller step
                break
            L *= self.eta
            n_bt += 1
            if L >= self.L_max or n_bt > self.max_bt:
                L = self.L_max
                x = step(1.0 / L)
                S_new = S(x)
                break

        self.tau_k = 1.0 / L
        self.L_history.append(L)
        self.n_backtracks.append(n_bt)
        return out(x, S_new)
# <<<<<<<<<< MD ADDED <<<<<<<<<<


CUSTOM_ITERATIONS = {"PGD": PnPPGDIteration, "PNP_MD": PnPMDIteration,
                     "PROJ_GD": ProxGDIteration, "PROX_GD": ProxGDIteration,
                     "MD_BURG": MirrorDescentIteration}          # MD ADDED: MD_BURG


def get_iteration(algo, snap_iters=(), log_cost=True, **bt_kwargs):
    """Custom iteration object, or the algo name for a deepinv built-in iteration.
    bt_kwargs (backtracking and stopping settings) and log_cost are used by PROJ_GD /
    PROX_GD / MD_BURG only."""
    cls = CUSTOM_ITERATIONS.get(algo.upper())
    if cls is None:
        return algo.upper()
    if cls in (ProxGDIteration, MirrorDescentIteration):         # MD ADDED: MirrorDescentIteration
        return cls(snap_iters=snap_iters, log_cost=log_cost, **bt_kwargs)
    return cls(snap_iters=snap_iters)


def init_fn(observation, physics, init_choice):
    x_adj = physics.A_adjoint(observation)      # right shape, device and dtype
    if init_choice == "adjoint_y":
        x0 = x_adj / x_adj.abs().max().clamp_min(1e-12)
    elif init_choice == "random":
        x0 = torch.randn_like(x_adj)
    elif init_choice == "constant":
        x0 = 0.5 * torch.ones_like(x_adj)
    else:
        raise ValueError(f"Unknown init: {init_choice}")
    return {"est": (x0, x0.clone())}


def build_model(prior, data_fidelity, iteration, stepsize, sigma, lam, n_iter,
                init_choice="adjoint_y", deepinv_backtracking=False, stop_fn=None):
    """stop_fn: check_conv_fn(it, X_prev, X) -> bool for early stopping (None: always
    n_iter iterations)."""
    model = optim_builder(
        iteration=iteration, prior=prior, data_fidelity=data_fidelity, g_first=False,
        params_algo={"stepsize": stepsize, "g_param": sigma, "lambda": lam},
        backtracking=deepinv_backtracking, early_stop=stop_fn is not None, max_iter=n_iter,
        get_output=lambda X: X["est"][0].clamp(0, 1),
        custom_init=partial(init_fn, init_choice=init_choice),
        verbose=False,
    )
    if stop_fn is not None:
        model.fixed_point.check_conv_fn = stop_fn
    return model
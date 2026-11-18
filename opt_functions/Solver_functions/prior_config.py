"""
Priors.

get_denoiser(loss, cfg)    -> a denoiser (TV, DRUNet, fine-tuned DRUNet) or an explicit
                              regularizer (WCRR, smoothed TV, TV, L1)
make_prior(denoiser, algo, is_3d) -> the prior used by the solver

Explicit regularizers have explicit_prior = True and get_mu() (Lipschitz constant
of grad R). smooth_prior tells the PROJ_GD / PROX_GD iteration how to use R:
    smooth_prior = True   R enters the gradient step      (WCRR, smoothed TV)
    smooth_prior = False  R enters the proximal step      (L1, TV_nonsmooth, TV_lhs, HS)
"""
import os
import sys

import torch
import torch.nn.functional as F
import deepinv as dinv

# The WCRR code lives in a sibling repository
_WCRR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                          "wcrr"))
if _WCRR_ROOT not in sys.path:
    sys.path.append(_WCRR_ROOT)
from models import utils as wcrr_utils  # noqa: E402

# Algorithms that take a gradient step on the prior -> RED; all others -> PnP
GRADIENT_ALGOS = {"MD", "GD"}


class AllPassConv(torch.nn.Module):
    """
    Drop-in replacement for the WCRR multi-convolution: every channel gets the same
    zero-mean all-pass filter, |H(f)| = 1 for f != 0 and H(0) = 0, i.e.
        W_c x = (x - mean(x)) / sqrt(C)       (mean over the image)
    so W^T W = I - (projection on constants) and ||W|| = 1, as for the learned filters.
    The regularizer becomes pixel-wise: only the learned activations and scalings act.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.L = torch.tensor(1.)                            # already normalized

    @staticmethod
    def _zero_mean(x):
        return x - x.mean(dim=(-2, -1), keepdim=True)

    def forward(self, x):
        return self._zero_mean(x).repeat(1, self.num_channels, 1, 1) / self.num_channels ** 0.5

    def transpose(self, y):
        return self._zero_mean(y.sum(1, keepdim=True)) / self.num_channels ** 0.5

    def spectral_norm(self, *args, **kwargs):
        return self.L


class WCRRPrior(dinv.optim.Prior):
    """Weakly convex ridge regularizer, with explicit cost and gradient.
    allpass=True: learned filters replaced by AllPassConv (learned activations only)."""

    def __init__(self, device="cpu", mode = "pretrained", n_power_steps=200, allpass=False):
        super().__init__()
        self.explicit_prior = True
        self.smooth_prior = True
        if mode == "pretrained":
            self.model = wcrr_utils.load_model("WCRR-CNN", str(device))
        elif mode == "ISMtrain":
            self.model = wcrr_utils.load_model("WCRR-CNN-ISM", str(device), base_dir="/data/cil/sofia/ism_regularisation_learned/training_wcrr/trained_models")
        elif mode == "ISMtrain_softplus":
            self.model = wcrr_utils.load_model("WCRR-CNN-softplus-ISM", str(device), base_dir="/data/cil/sofia/ism_regularisation_learned/training_wcrr/trained_models")


        # Lipschitz bound of the conv layer, computed once
        with torch.no_grad():
            self.model.conv_layer.spectral_norm(mode="power_method", n_steps=n_power_steps)
        if allpass:
            self.model.conv_layer = AllPassConv(self.model.num_channels)

        # Move the model, including tensors stored as plain attributes (spline grids)
        self.model.to(device)
        for m in self.model.modules():
            for name, val in list(vars(m).items()):
                if torch.is_tensor(val) and not isinstance(val, torch.nn.Parameter):
                    setattr(m, name, val.to(device))

    @staticmethod
    def _sigma(x, sigma):
        """sigma as a (B, 1, 1, 1) tensor matching the batch of x."""
        s = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        return s.reshape(-1, 1, 1, 1).expand(x.shape[0], 1, 1, 1)

    def fn(self, x, sigma, *args, **kwargs):
        return self.model.cost(x, self._sigma(x, sigma))

    def grad(self, x, sigma, *args, **kwargs):
        return self.model.grad(x, self._sigma(x, sigma))

    def get_mu(self):
        """Lipschitz constant of grad R."""
        return self.model.get_mu()


# WCHS (learned Hessian-Schatten, WCRR-based) lives in the sibling repository learned-hs
_LEARNED_HS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                                "learned-hs"))
WCHS_MODELS = {"WCHS_free": "wchs_free",              # trained on BSD (natural images)
               "WCHS_free_smlm": "wchs_free_smlm"}    # trained on SMLM renders


class WCHSPrior(dinv.optim.Prior):
    """
    Weakly convex Hessian-Schatten regularizer of learned-hs (WCvxSpectralNet), loaded with
    learned-hs's own loader (src.priors.learned.WCRRPrior) and exposed with the same
    interface as WCRRPrior above: fn(x, sigma), grad(x, sigma), get_mu().
    Same sigma convention as WCRR (noise std ~ sigma / 255, trained on sigma in [0, 30]).

    model      : folder in learned-hs/pretrained ("wchs_free", "wchs_free_smlm", ...)
    checkpoint : training step (checkpoint_<n>.pth); None = the last one, as in
                 learned-hs experiments/deconv_eval.py
    norm       : "fourier" (as trained and as deconv_eval evaluates it; get_mu includes the
                 norm^2 > 1 of the finite operator) or "power" (as WCRRPrior above)
    """

    def __init__(self, model, device="cpu", checkpoint=None, norm="fourier"):
        super().__init__()
        self.explicit_prior = True
        self.smooth_prior = True
        if _LEARNED_HS_ROOT not in sys.path:
            sys.path.append(_LEARNED_HS_ROOT)
        from src.priors.learned import WCRRPrior as _LearnedHSLoader   # noqa: E402

        root = os.path.join(_LEARNED_HS_ROOT, "pretrained", model)
        if checkpoint is None:
            checkpoint = max(int(f[len("checkpoint_"):-len(".pth")])
                             for f in os.listdir(os.path.join(root, "checkpoints"))
                             if f.startswith("checkpoint_"))
        self.checkpoint = checkpoint
        self._hs = _LearnedHSLoader(os.path.join(root, "checkpoints", f"checkpoint_{checkpoint}.pth"),
                                    os.path.join(root, "config.json"), device=torch.device(device),
                                    norm=norm)
        self.model = self._hs.model
        self._mu = torch.tensor(self._hs.lipschitz_constant())   # mu * op_norm2, fixed

    _sigma = staticmethod(WCRRPrior._sigma)

    def fn(self, x, sigma, *args, **kwargs):
        return self.model.cost(x, self._sigma(x, sigma))

    def grad(self, x, sigma, *args, **kwargs):
        return self.model.grad(x, self._sigma(x, sigma))

    def get_mu(self):
        """Lipschitz constant of grad R."""
        return self._mu


def _img_grad(x):
    """Forward differences along x and y (zero at the last row / column)."""
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def _img_grad_adjoint(dx, dy):
    """Adjoint of _img_grad (minus the divergence)."""
    dx, dy = dx[..., :, :-1], dy[..., :-1, :]
    return (F.pad(dx, (1, 0)) - F.pad(dx, (0, 1))
            + F.pad(dy, (0, 0, 1, 0)) - F.pad(dy, (0, 0, 0, 1)))


class SmoothTVPrior(dinv.optim.Prior):
    """
    Smoothed total variation: R(x) = sum_pixels sqrt(|grad x|^2 + eps^2).
    Differentiable, with Lip(grad R) = 8 / eps. Use with 'PROJ_GD'.
    """

    def __init__(self, eps=1e-2):
        super().__init__()
        self.explicit_prior = True
        self.smooth_prior = True
        self.eps = eps

    def fn(self, x, *args, **kwargs):
        dx, dy = _img_grad(x)
        return torch.sqrt(dx ** 2 + dy ** 2 + self.eps ** 2).flatten(1).sum(1)

    def grad(self, x, *args, **kwargs):
        dx, dy = _img_grad(x)
        norm = torch.sqrt(dx ** 2 + dy ** 2 + self.eps ** 2)
        return _img_grad_adjoint(dx / norm, dy / norm)

    def get_mu(self):
        return torch.tensor(8.0 / self.eps)


def tv_prox_fgp(z, gamma, n_iter=200, tol=1e-6, check_every=5, nonneg=True, warm_start=None):
    """
    prox of gamma * (TV + indicator{x >= 0}) at z (isotropic TV, same discretization as
    SmoothTVPrior with eps = 0):
        argmin_{x >= 0}  0.5 ||x - z||^2 + gamma * sum_pixels |grad x|
    by FGP (Beck & Teboulle 2009, fast gradient projection on the dual):
        x(q) = P_{x>=0}(z - gamma * grad^T q),
        q   <- P_{|q_i| <= 1}(r + grad x(r) / (8 gamma)),   FISTA extrapolation on r.
    The positivity is inside the prox (P_{x>=0}(prox_TV(z)) would not be the prox of
    TV + positivity in 2D). Stopping (checked every check_every iterations, or after
    n_iter iterations): duality gap
        gap = gamma * (TV(x(q)) - <grad x(q), q>) >= 0.5 ||x(q) - x*||^2
    (the primal is 1-strongly convex), stop when sqrt(2 gap) <= tol * ||x||, i.e. the
    relative error ||x - x*|| / ||x|| is certified <= tol.
    warm_start: dict; its dual "q" is used as initial point (if the shape matches) and
    replaced by the final dual, so consecutive calls (one per outer iteration) are warm.
    """
    if gamma <= 0:
        return z.clamp_min(0) if nonneg else z.clone()
    proj = (lambda v: v.clamp_min(0)) if nonneg else (lambda v: v)
    q = warm_start.get("q") if warm_start is not None else None
    if q is None or q[0].shape != z.shape:
        q = (torch.zeros_like(z), torch.zeros_like(z))
    qx, qy = q
    rx, ry = qx, qy
    t = 1.0
    for k in range(n_iter):
        x = proj(z - gamma * _img_grad_adjoint(rx, ry))
        dx, dy = _img_grad(x)
        nx, ny = rx + dx / (8 * gamma), ry + dy / (8 * gamma)
        norm = torch.sqrt(nx ** 2 + ny ** 2).clamp_min(1.0)
        nx, ny = nx / norm, ny / norm
        t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
        beta = (t - 1) / t_next
        rx, ry = nx + beta * (nx - qx), ny + beta * (ny - qy)
        qx, qy, t = nx, ny, t_next
        if tol is not None and (k + 1) % check_every == 0:
            x_cur = proj(z - gamma * _img_grad_adjoint(qx, qy))
            gx, gy = _img_grad(x_cur)
            gap = gamma * (torch.sqrt(gx ** 2 + gy ** 2) - (gx * qx + gy * qy)).sum()
            if float((2 * gap.clamp_min(0)).sqrt()) <= tol * float(x_cur.norm()):
                break
    if warm_start is not None:
        warm_start["q"] = (qx, qy)
    return proj(z - gamma * _img_grad_adjoint(qx, qy))


class TVPrior(dinv.optim.Prior):
    """
    Isotropic total variation without smoothing: R(x) = sum_pixels |grad x|.
    Not differentiable: used through its prox (tv_prox_fgp, which includes the
    projection on x >= 0). Use with 'PROX_GD'.
    """

    def __init__(self, n_iter=200, tol=1e-6):
        super().__init__()
        self.explicit_prior = True
        self.smooth_prior = False
        self.n_iter, self.tol = n_iter, tol

    def fn(self, x, *args, **kwargs):
        dx, dy = _img_grad(x)
        return torch.sqrt(dx ** 2 + dy ** 2).flatten(1).sum(1)

    def prox(self, x, *args, gamma=1.0, warm_start=None, **kwargs):
        return tv_prox_fgp(x, gamma, n_iter=self.n_iter, tol=self.tol, warm_start=warm_start)

    def get_mu(self):
        return torch.tensor(0.0)     # no gradient step on R


class L1Prior(dinv.optim.Prior):
    """
    R(x) = ||x||_1. Not differentiable: used through its prox (soft thresholding).
    Use with 'PROX_GD'.
    """

    def __init__(self):
        super().__init__()
        self.explicit_prior = True
        self.smooth_prior = False

    def fn(self, x, *args, **kwargs):
        return x.abs().flatten(1).sum(1)

    def prox(self, x, *args, gamma=1.0, **kwargs):
        return torch.sign(x) * (x.abs() - gamma).clamp_min(0)

    def get_mu(self):
        return torch.tensor(0.0)     # no gradient step on R


# >>>>>>>>>> LEARNED-HS ANALYSIS PRIORS >>>>>>>>>>
# Non-learned analysis priors R(x) = h(Kx) of learned-hs (src/priors/analysis.py, imported,
# not copied): h a norm, K a linear operator, each exposing K, K_adjoint, prox_conj (projection
# onto the dual-norm ball), cost and norm_K2. Their prox (+ x >= 0) is computed here by the same
# FGP as tv_prox_fgp, written for any K.
#   TV_lhs  IsotropicTVPrior      h = ||.||_{2,1}, K = forward differences, Neumann (deepinv's
#                                 nabla: the same K as TV_nonsmooth above, a cross-check)
#   HS      HessianSchattenPrior  h = sum_pixels ||.||_S1 (nuclear norm of the 2x2 Hessian),
#                                 K = centred 3x3 Hessian, valid region (null space = affine
#                                 images), orthonormal Sym(2) coordinates; ||K||^2 ~ 32
LEARNED_HS_ANALYSIS = {"TV_lhs": "IsotropicTVPrior", "HS": "HessianSchattenPrior"}
SQRT2 = 2 ** 0.5


def proj_sinf_u_safe(u, lam, dim=2):
    """
    Projection onto {||X||_S-inf <= lam} in orthonormal Sym(2) coordinates u = (u0, u1, u2):
    clip the eigenvalues mu+- = (u0 +- q) / sqrt2, q = ||(u1, u2)||, to [-lam, lam].
    Replaces learned-hs proj_sinf_u (sym2.spectral_value) for the HS prox: there the
    deviatoric scale (f(mu+) - f(mu-)) / (mu+ - mu-) takes the gap as the difference
    mu+ - mu-, which rounds to 0 in float32 when |u0| >> q (e.g. u0 ~ 1e7, q ~ 0.5: the FGP
    dual step K x / (||K||^2 gamma) with KL steps gamma ~ 1e-9) -> 0/0 = NaN, and the whole
    prox (and run) became NaN. Here the gap is the exact sqrt2 * q >= sqrt2 * sqrt(eps): never
    0, same accuracy otherwise (output error O(eps * max(|f|, |u0|))).
    """
    u0, u1, u2 = u.unbind(dim)
    q = torch.sqrt(torch.clamp(u1 * u1 + u2 * u2, min=torch.finfo(u.dtype).eps ** 2))
    fp = torch.clamp((u0 + q) / SQRT2, -lam, lam)
    fm = torch.clamp((u0 - q) / SQRT2, -lam, lam)
    kappa = (fp - fm) / (SQRT2 * q)
    return torch.stack([(fp + fm) / SQRT2, kappa * u1, kappa * u2], dim=dim)


def analysis_prox_fgp(prior, z, gamma, n_iter=200, tol=1e-6, check_every=5, nonneg=True,
                      warm_start=None):
    """
    prox of gamma * (h(K .) + indicator{x >= 0}) at z, for a learned-hs analysis prior:
        argmin_{x >= 0}  0.5 ||x - z||^2 + gamma * h(Kx)
    tv_prox_fgp with K, its adjoint, ||K||^2 and the projection onto the dual unit ball taken
    from the prior (FGP, Beck & Teboulle 2009, on the dual q in the unit h*-ball):
        x(q) = P_{x>=0}(z - gamma K^T q),
        q   <- P_{h* <= 1}(r + K x(r) / (||K||^2 gamma)),   FISTA extrapolation on r.
    Stopping (checked every check_every iterations, or after n_iter): duality gap
        gap = gamma * (h(K x(q)) - <K x(q), q>) >= 0.5 ||x(q) - x*||^2,
    stop when sqrt(2 gap) <= tol * ||x||. warm_start: dict, as in tv_prox_fgp.
    """
    if gamma <= 0:
        return z.clamp_min(0) if nonneg else z.clone()
    proj = (lambda v: v.clamp_min(0)) if nonneg else (lambda v: v)
    L = prior.norm_K2(z)
    q = warm_start.get("q_analysis") if warm_start is not None else None
    if q is None or q[0] != tuple(z.shape):
        q = (tuple(z.shape), torch.zeros_like(prior.K(z)))
    q = q[1]
    r, t = q, 1.0
    for k in range(n_iter):
        x = proj(z - gamma * prior.K_adjoint(r))
        q_next = prior.prox_conj(r + prior.K(x) / (L * gamma), 1.0)
        t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
        r = q_next + (t - 1) / t_next * (q_next - q)
        q, t = q_next, t_next
        if tol is not None and (k + 1) % check_every == 0:
            # gap in float64: h(Kx) and <Kx, q> are two large sums that cancel (in float32
            # the gap came out too small and the prox stopped ~10x less accurate)
            x_cur = proj(z - gamma * prior.K_adjoint(q)).double()
            gap = gamma * (prior.cost(x_cur).sum() - (prior.K(x_cur) * q.double()).sum())
            if float((2 * gap.clamp_min(0)).sqrt()) <= tol * float(x_cur.norm()):
                break
    if warm_start is not None:
        warm_start["q_analysis"] = (tuple(z.shape), q)
    return proj(z - gamma * prior.K_adjoint(q))


class LearnedHSAnalysisPrior(dinv.optim.Prior):
    """
    A non-learned analysis prior of learned-hs (LEARNED_HS_ANALYSIS: "TV_lhs", "HS"), used
    through its prox like TVPrior above (analysis_prox_fgp, includes x >= 0). Use with 'PROX_GD'.
    """

    def __init__(self, name, n_iter=200, tol=1e-6):
        super().__init__()
        self.explicit_prior = True
        self.smooth_prior = False
        self.n_iter, self.tol = n_iter, tol
        if _LEARNED_HS_ROOT not in sys.path:
            sys.path.append(_LEARNED_HS_ROOT)
        from src.priors import analysis   # noqa: E402
        self.analysis = getattr(analysis, LEARNED_HS_ANALYSIS[name])()
        # HessianSchattenPrior._w copies its CPU float64 stencil to the GPU on EVERY K / K^T
        # call: a blocking host->device copy, ~2 per FGP iteration, which on a shared GPU made
        # HS ~5x slower per iteration than TV. Memoize it per (dtype, device): same values,
        # bit-identical results.
        if hasattr(self.analysis, "_w"):
            w, cache = self.analysis._w, {}

            def w_cached(like):
                key = (like.dtype, like.device)
                if key not in cache:
                    cache[key] = w(like)
                return cache[key]
            self.analysis._w = w_cached
        if name == "HS":                 # learned-hs proj_sinf_u gives NaN at large |u0| / q
            self.analysis.prox_conj = lambda z, gamma: proj_sinf_u_safe(z, gamma, dim=2)

    def fn(self, x, *args, **kwargs):
        return self.analysis.cost(x)

    def prox(self, x, *args, gamma=1.0, warm_start=None, **kwargs):
        return analysis_prox_fgp(self.analysis, x, gamma, n_iter=self.n_iter, tol=self.tol,
                                 warm_start=warm_start)

    def get_mu(self):
        return torch.tensor(0.0)     # no gradient step on R
# <<<<<<<<<< LEARNED-HS ANALYSIS PRIORS <<<<<<<<<<


class PositiveDenoiser(torch.nn.Module):
    """Denoiser with output clipped to [0, 1]."""

    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x, sigma, **kwargs):
        return self.denoiser(x, sigma, **kwargs).clamp(0, 1)


def get_denoiser(loss, cfg):
    """The bare denoiser / regularizer (not yet wrapped as PnP or RED)."""
    name = cfg.prior_name
    if name == "TV":
        return dinv.models.TVDenoiser(n_it_max=20)
    if name == "WCRR":
        return WCRRPrior(device=cfg.device, mode='pretrained')
    if name == "WCRR_ISM":
        return WCRRPrior(device=cfg.device, mode='ISMtrain')
    if name == "WCRR_softplus_ISM":
        return WCRRPrior(device=cfg.device, mode='ISMtrain_softplus')
    if name == "WCRR_allpass":                       # learned activations, all-pass filters
        return WCRRPrior(device=cfg.device, mode='pretrained', allpass=True)
    if name == "WCRR_ISM_allpass":
        return WCRRPrior(device=cfg.device, mode='ISMtrain', allpass=True)
    if name == "WCRR_softplus_ISM_allpass":
        return WCRRPrior(device=cfg.device, mode='ISMtrain_softplus', allpass=True)
    if name in WCHS_MODELS:                          # learned-hs: WCHS_free, WCHS_free_smlm
        return WCHSPrior(WCHS_MODELS[name], device=cfg.device)
    if name == "TV_smooth":
        return SmoothTVPrior(eps=cfg.tv_eps)
    if name == "TV_nonsmooth":
        return TVPrior(n_iter=cfg.tv_prox_iter, tol=cfg.tv_prox_tol)
    if name == "L1":
        return L1Prior()
    if name in LEARNED_HS_ANALYSIS:                  # learned-hs: TV_lhs, HS (prox by FGP)
        return LearnedHSAnalysisPrior(name, n_iter=cfg.tv_prox_iter, tol=cfg.tv_prox_tol)
    if name == "Drunet":
        return PositiveDenoiser(dinv.models.DRUNet(in_channels=1, out_channels=1,
                                                   pretrained="download", device=cfg.device))
    if name == "Drunet_finetune":
        drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                    pretrained=None, device=cfg.device)
        ckpt = torch.load(os.path.join(cfg.ckpt_dir, f"best_model_checkpoint_drunet_{loss}.pth"),
                          map_location=cfg.device)
        drunet.load_state_dict(ckpt["model_state_dict"])
        return PositiveDenoiser(drunet)
    
    if name == "Drunet_fromscratch":
        drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                    pretrained=None, device=cfg.device)
        ckpt = torch.load(os.path.join(cfg.ckpt_dir, f"scratch_{loss}/best_model.pth"),
                          map_location=cfg.device)
        drunet.load_state_dict(ckpt["model_state_dict"])
        return PositiveDenoiser(drunet)
    raise ValueError(f"Unknown prior: {name}")


def wrap_prior(denoiser, algo):
    if algo.upper() in GRADIENT_ALGOS:
        return dinv.optim.RED(denoiser=denoiser)
    return dinv.optim.PnP(denoiser=denoiser)


# 3D data (2 axial planes as channels): plane 0 is OUT of focus, plane 1 is IN focus.
# The regularizer acts on the in-focus plane only, and every metric (pnp_utils.metric_view,
# compare_results) is computed on it.
REG_PLANE = 1


class PlanePrior(dinv.optim.Prior):
    """
    3D data, x of shape (B, 2, Nx, Nx): regularize only plane p (default REG_PLANE, the
    in-focus one); the other plane is left free.
        R(x)       = R(x[:, p])
        grad R(x)  = grad R(x[:, p]) on plane p, 0 on the other
        prox R(x)  = prox R(x[:, p]) on plane p, the other unchanged  (PnP: denoise plane p)
    Works for explicit priors (fn, grad, prox), RED (grad) and PnP (prox).
    """

    def __init__(self, prior, plane=REG_PLANE):
        super().__init__()
        self.prior = prior
        self.plane = plane
        self.explicit_prior = getattr(prior, "explicit_prior", False)
        self.smooth_prior = getattr(prior, "smooth_prior", True)

    def _sel(self, x):
        return x[:, self.plane:self.plane + 1]

    def _put(self, x, xp):
        """x with plane p replaced by xp."""
        return torch.cat([xp if k == self.plane else x[:, k:k + 1]
                          for k in range(x.shape[1])], dim=1)

    def fn(self, x, *args, **kwargs):
        return self.prior.fn(self._sel(x), *args, **kwargs)

    def grad(self, x, *args, **kwargs):
        return self._put(torch.zeros_like(x), self.prior.grad(self._sel(x), *args, **kwargs))

    def prox(self, x, *args, **kwargs):
        return self._put(x, self.prior.prox(self._sel(x), *args, **kwargs))

    def get_mu(self):
        # restricting R to one plane does not change the Lipschitz constant of grad R
        return self.prior.get_mu()


FirstPlanePrior = PlanePrior          # old name


def make_prior(denoiser, algo, is_3d):
    """Final prior used by the solver: explicit (WCRR), RED or PnP,
    applied only to plane REG_PLANE (the in-focus one) for 3D data."""
    explicit = getattr(denoiser, "explicit_prior", False)
    prior = denoiser if explicit else wrap_prior(denoiser, algo)
    return PlanePrior(prior, REG_PLANE) if is_3d else prior
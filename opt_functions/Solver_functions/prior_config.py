"""
Priors.

get_denoiser(loss, cfg)    -> a denoiser (TV, DRUNet, fine-tuned DRUNet) or an explicit
                              regularizer (WCRR, smoothed TV, L1)
make_prior(denoiser, algo, is_3d) -> the prior used by the solver

Explicit regularizers have explicit_prior = True and get_mu() (Lipschitz constant
of grad R). smooth_prior tells the PROJ_GD / PROX_GD iteration how to use R:
    smooth_prior = True   R enters the gradient step      (WCRR, smoothed TV)
    smooth_prior = False  R enters the proximal step      (L1)
"""
import os
import sys

import torch
import torch.nn.functional as F
import deepinv as dinv

# The WCRR code lives in a sibling repository
_WCRR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                          "weakly_convex_ridge_regularizer"))
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
    if name == "TV_smooth":
        return SmoothTVPrior(eps=cfg.tv_eps)
    if name == "L1":
        return L1Prior()
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


class FirstPlanePrior(dinv.optim.Prior):
    """
    3D data, x of shape (B, 2, Nx, Nx): regularize only the first plane.
        R(x)       = R(x[:, 0])
        grad R(x)  = [grad R(x[:, 0]), 0]
        prox R(x)  = [prox R(x[:, 0]), x[:, 1]]      (PnP: denoise plane 0 only)
    Works for explicit priors (fn, grad, prox), RED (grad) and PnP (prox).
    """

    def __init__(self, prior):
        super().__init__()
        self.prior = prior
        self.explicit_prior = getattr(prior, "explicit_prior", False)
        self.smooth_prior = getattr(prior, "smooth_prior", True)

    def fn(self, x, *args, **kwargs):
        return self.prior.fn(x[:, 0:1], *args, **kwargs)

    def grad(self, x, *args, **kwargs):
        g0 = self.prior.grad(x[:, 0:1], *args, **kwargs)
        return torch.cat([g0, torch.zeros_like(x[:, 1:])], dim=1)

    def prox(self, x, *args, **kwargs):
        x0 = self.prior.prox(x[:, 0:1], *args, **kwargs)
        return torch.cat([x0, x[:, 1:]], dim=1)

    def get_mu(self):
        # restricting R to one plane does not change the Lipschitz constant of grad R
        return self.prior.get_mu()


def make_prior(denoiser, algo, is_3d):
    """Final prior used by the solver: explicit (WCRR), RED or PnP,
    applied only to the first plane for 3D data."""
    explicit = getattr(denoiser, "explicit_prior", False)
    prior = denoiser if explicit else wrap_prior(denoiser, algo)
    return FirstPlanePrior(prior) if is_3d else prior
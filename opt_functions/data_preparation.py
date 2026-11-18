"""
Data loading and simulated ISM measurements (2D and 3D).

Main entry points:
    load_data(name, cfg)            -> ground truth, normalized to [0, 1]
    make_measurements(x, flux, cfg) -> dict with y, physics, data fidelities, Lipschitz constants

3D data has two axial planes stored as channels: x has shape (B, 2, Nx, Nx).
"""
import os

import torch
import deepinv as dinv
from deepinv.optim.data_fidelity import PoissonLikelihood, L2
from deepinv.optim.distance import PoissonLikelihoodDistance      # MD ADDED
from deepinv.optim.distance import L2Distance                     # L2 BKG ADDED
from deepinv.physics import PoissonNoise

import ISM.simulation.PSF_sim as ism

N_DET = 25          # 5 x 5 SPAD detector elements
CENTER = 12         # index of the central detector
BKG_LEVEL = 1e-4    # background level, relative to the central detector


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(name, cfg):
    """Load cfg.data_dir/<name>.pth and normalize it to [0, 1].
    3D: each plane is normalized separately, then stacked as channels."""
    x = torch.load(os.path.join(cfg.data_dir, f"{name}.pth"), map_location=cfg.device)
    if cfg.is_3d:
        return torch.cat([x[0] / x[0].max() + 1e-5, x[1] / x[1].max() + 1e-5], dim=1)
    
    else:
        return x / x.max() + 1e-5


def make_measurements(x, flux, cfg):
    """Same seed -> same noise realization for every run at this (data, flux).
    cfg.batched_physics: y, physics and the data fidelities in batched form (one
    25-channel tensor instead of a list of 25, see batch_measurements); the simulated
    y, the noise and the Lipschitz constants are the same."""
    torch.manual_seed(cfg.seed)
    generate = generate_meas_ism_3D if cfg.is_3d else generate_meas_ism
    meas = generate(x, cfg.Nx, cfg.Nz, cfg.pxsizex, flux, cfg.device)
    if getattr(cfg, "batched_physics", False):
        meas = batch_measurements(meas)
    return meas


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

class PoissonWithBackground(PoissonNoise):
    """Poisson noise on x + bkg."""

    def __init__(self, gain, bkg):
        super().__init__(gain=gain)
        self.bkg = bkg

    def forward(self, x, **kwargs):
        return super().forward(x + self.bkg, **kwargs)


# >>>>>>>>>> MD ADDED >>>>>>>>>>
class PoissonNoBkgDistance(PoissonLikelihoodDistance):
    """KL without background: sum z - y log z (z = A x in counts, y in counts).
    A x is clamped to 1e-12 so that tiny FFT round-off values never reach the log."""

    def fn(self, x, y, *args, **kwargs):
        if self.denormalize:
            y = y / self.gain
        z = (x / self.gain).clamp_min(1e-12)
        return (-y * torch.log(z)).flatten().sum() + (z - y).reshape(x.shape[0], -1).sum(dim=1)

    def grad(self, x, y, *args, **kwargs):
        if self.denormalize:
            y = y / self.gain
        z = (x / self.gain).clamp_min(1e-12)
        return (1 / self.gain) * (1 - y / z)


class PoissonLikelihoodNoBkg(PoissonLikelihood):
    """PoissonLikelihood with background 0 (fidelity 'KL0', used by mirror descent)."""

    def __init__(self, gain=1.0):
        super().__init__(gain=gain, bkg=0.0, denormalize=True)
        self.d = PoissonNoBkgDistance(gain=gain, bkg=0.0, denormalize=True)
# <<<<<<<<<< MD ADDED <<<<<<<<<<


# >>>>>>>>>> L2 BKG ADDED >>>>>>>>>>
class L2BkgDistance(L2Distance):
    """d(u, y) = 1 / (2 sigma^2) ||u + bkg - y||^2, u = A x (bkg in the units of y)."""

    def __init__(self, sigma=1.0, bkg=0.0):
        super().__init__(sigma=sigma)
        self.bkg = bkg

    def fn(self, x, y, *args, **kwargs):
        return super().fn(x + self.bkg, y)

    def grad(self, x, y, *args, **kwargs):
        return super().grad(x + self.bkg, y)

    def prox(self, x, y, *args, gamma=1.0, **kwargs):
        # argmin_u gamma/(2 sigma^2) ||u + bkg - y||^2 + 1/2 ||u - x||^2
        return (x + self.norm * gamma * (y - self.bkg)) / (1 + gamma * self.norm)


class L2WithBackground(L2):
    """f(x) = 1 / (2 sigma^2) ||A x + bkg - y||^2: L2 fidelity with the same additive
    background as the measurement model (PoissonWithBackground), bkg in intensity units."""

    def __init__(self, sigma=1.0, bkg=0.0):
        super().__init__(sigma=sigma)
        self.d = L2BkgDistance(sigma=sigma, bkg=bkg)
        self.bkg = bkg

    def prox(self, x, y, physics, *args, gamma=1.0, **kwargs):
        # argmin_u gamma/(2 sigma^2) ||A u + bkg - y||^2 + 1/2 ||u - x||^2
        return physics.prox_l2(x, y - self.bkg, self.norm * gamma)
# <<<<<<<<<< L2 BKG ADDED <<<<<<<<<<


class SummedBlurPhysics(dinv.physics.LinearPhysics):
    """One detector in 3D: y = A1 x[:, 0] + A2 x[:, 1] (one blur per axial plane)."""

    def __init__(self, Nx, kernel1, kernel2, noise_model, device):
        super().__init__(noise_model=noise_model)
        self.A1 = dinv.physics.BlurFFT(img_size=(1, Nx, Nx), filter=kernel1, device=device)
        self.A2 = dinv.physics.BlurFFT(img_size=(1, Nx, Nx), filter=kernel2, device=device)

    def A(self, x, **kwargs):
        return self.A1.A(x[:, 0:1]) + self.A2.A(x[:, 1:2])

    def A_adjoint(self, y, **kwargs):
        return torch.cat([self.A1.A_adjoint(y), self.A2.A_adjoint(y)], dim=1)


def _optical_setup(pxsizex, Nz):
    grid = ism.GridParameters()
    grid.N = 5                # detector elements per side
    grid.pxsizex = pxsizex    # lateral pixel size of the simulation (nm)
    grid.pxdim = 50e3         # detector element size (nm)
    grid.pxpitch = 75e3       # detector element pitch (nm)
    grid.M = 500              # total magnification
    grid.Nz = Nz
    grid.pxsizez = 700        # axial pixel size (nm)

    exPar = ism.simSettings()
    exPar.wl = 640            # excitation wavelength (nm)
    exPar.mask_sampl = 31
    emPar = exPar.copy()
    emPar.wl = 660            # emission wavelength (nm)
    return grid, exPar, emPar


def generate_meas_ism(x, Nx, Nz, pxsizex, flux, device):
    """2D ISM: 25 blurred, noisy images of x. Nz is not used."""
    grid, exPar, emPar = _optical_setup(pxsizex, Nz=1)
    PSF, _, _ = ism.SPAD_PSF_2D(grid, exPar, emPar)
    PSF = PSF.to(device)

    finger_print = PSF.reshape(N_DET, -1).sum(1)
    eta_vec = BKG_LEVEL * finger_print / finger_print[CENTER]

    physics_list = [
        dinv.physics.BlurFFT(img_size=(1, Nx, Nx), filter=PSF[i].unsqueeze(0), device=device,
                             noise_model=PoissonWithBackground(gain=1 / flux, bkg=eta_vec[i]))
        for i in range(N_DET)
    ]
    return _finish(x, PSF, physics_list, finger_print, eta_vec, flux, device)


def generate_meas_ism_3D(x, Nx, Nz, pxsizex, flux, device):
    """3D ISM with 2 axial planes: each detector sees the sum of both blurred planes."""
    grid, exPar, emPar = _optical_setup(pxsizex, Nz)
    PSF, _, _ = ism.SPAD_PSF_3D(grid, exPar, emPar)
    PSF = PSF.permute(3, 0, 1, 2).contiguous().to(device)      # (25, Nz, k, k)
    assert PSF.shape[1] == 2, f"expected 2 axial planes, got PSF shape {tuple(PSF.shape)}"
    PSF = PSF / PSF.sum(dim=(0, 2, 3), keepdim=True)           # each plane sums to 1

    finger_print = PSF.reshape(N_DET, -1).sum(1)
    eta_vec = BKG_LEVEL * finger_print / PSF[CENTER, 0].sum()

    physics_list = [
        SummedBlurPhysics(Nx, PSF[i:i + 1, 0:1], PSF[i:i + 1, 1:2], noise_model =  PoissonWithBackground(gain=1 / flux, bkg=eta_vec[i]), device = device)
        for i in range(N_DET)
    ]
    return _finish(x, PSF, physics_list, finger_print, eta_vec, flux, device)


# >>>>>>>>>> REAL DATA ADDED >>>>>>>>>>
def real_measurements_3D(counts, PSF, flux=1.0, device="cuda"):
    """
    Real 3D ISM data (Data/Real_data/<name>_data.pth, PSF estimated from the data by
    s2ism.psf_estimator_from_data in real_data_load.py) with the forward model, background
    convention, fidelities and Lipschitz bounds of generate_meas_ism_3D, without simulating.

    counts: (25, 1, N, N) photon counts (the file's 'measurment'), PSF: (25, 2, k, k),
            plane 0 out of focus, plane 1 in focus (as the simulation).
    flux:   intensity scale alpha, y_j = counts_j / flux, i.e. counts ~ Poisson(flux (A_j x + eta_j)).
            flux = 1: x in photon counts (use with cfg.x_max = None);
            flux = alpha ~ peak counts: x in ~[0, 1] (learned priors).
    Background eta_j = BKG_LEVEL * fingerprint_j / PSF[CENTER, 0].sum() in units of x
    (flux = 1: counts), as generate_meas_ism_3D and the old back_vec of ism_regularisation.
    """
    counts = counts.float().to(device)
    N = counts.shape[-1]
    assert counts.shape[-2] == N and counts.shape[0] == N_DET, \
        f"expected (25, 1, N, N) counts, got {tuple(counts.shape)}"
    PSF = PSF.float().to(device)
    assert PSF.shape[:2] == (N_DET, 2), f"expected (25, 2, k, k) PSF, got {tuple(PSF.shape)}"
    PSF = PSF / PSF.sum(dim=(0, 2, 3), keepdim=True)            # each plane sums to 1

    finger_print = PSF.reshape(N_DET, -1).sum(1)
    eta_vec = BKG_LEVEL * finger_print / PSF[CENTER, 0].sum()
    physics_list = [
        SummedBlurPhysics(N, PSF[i:i + 1, 0:1], PSF[i:i + 1, 1:2],
                          noise_model=PoissonWithBackground(gain=1 / flux, bkg=eta_vec[i]),
                          device=device)
        for i in range(N_DET)
    ]
    y = dinv.utils.TensorList([counts[i:i + 1] / flux for i in range(N_DET)])   # (1, 1, N, N) each
    return _finish(None, PSF, physics_list, finger_print, eta_vec, flux, device, y=y)
# <<<<<<<<<< REAL DATA ADDED <<<<<<<<<<


# >>>>>>>>>> KL BKG FIXED >>>>>>>>>>
def _norm_A_sq(A_j, y_j):
    """Upper bound of ||A_j||^2: max(A_j 1) * max(A_j^T 1)."""
    ones_x = torch.ones_like(A_j.A_adjoint(y_j))
    return A_j.A(ones_x).max() * A_j.A_adjoint(torch.ones_like(y_j)).max()


def kl_with_background(physics_list, y, eta_vec, flux):
    """
    KL fidelity of the measurement model y_j = Poisson(flux (A_j x + eta_j)) / flux,
    and an upper bound of the Lipschitz constant of its gradient on x >= 0.

    y_j, eta_j in intensity units (the units of A x). deepinv's PoissonLikelihood works in
    counts: its mean is A_j x / gain + bkg with gain = 1 / flux, so bkg = eta_j * flux
    (the background in counts; before this fix bkg = eta_j, flux times too small).

        f(x) = sum_j sum_i [ flux (A_j x + eta_j)_i - flux y_ji log(flux (A_j x + eta_j)_i) ] + const
        grad f = flux sum_j A_j^T (1 - y_j / (A_j x + eta_j))
        Hess f = flux sum_j A_j^T diag(y_j / (A_j x + eta_j)^2) A_j
              <= flux sum_j max(y_j) / eta_j^2 ||A_j||^2      (A_j x >= 0)
    """
    df_kl = dinv.optim.StackedPhysicsDataFidelity(
        [PoissonLikelihood(gain=1 / flux, bkg=eta * flux) for eta in eta_vec])
    L_kl = sum(flux * y_j.abs().max() / eta_j ** 2 * _norm_A_sq(A_j, y_j)
               for A_j, y_j, eta_j in zip(physics_list, y, eta_vec))
    return df_kl, L_kl
# <<<<<<<<<< KL BKG FIXED <<<<<<<<<<


def _finish(x, PSF, physics_list, finger_print, eta_vec, flux, device, y=None):
    """Simulate y (or use the given measured y), build the data fidelities and the
    Lipschitz constants."""
    physics = dinv.physics.StackedLinearPhysics(physics_list, device=device)
    if y is None:
        y = physics(x)

    df_kl, L_kl = kl_with_background(physics_list, y, eta_vec, flux)    # KL BKG FIXED
    # L2 with the background of the measurement model: 0.5 ||A_j x + eta_j - y_j||^2
    df_l2 = dinv.optim.StackedPhysicsDataFidelity(
        [L2WithBackground(sigma=1.0, bkg=eta) for eta in eta_vec])     # L2 BKG ADDED

    # Upper bound of Lip(grad f_L2) = sum_j ||A_j||^2 (sigma = 1)
    L_l2 = sum(_norm_A_sq(A_j, y_j) for A_j, y_j in zip(physics_list, y))

    # >>>>>>>>>> MD ADDED >>>>>>>>>>
    # KL without background: not Lipschitz smooth, but L-smooth relative to the Burg
    # entropy with L = ||y||_1 in counts (as in ism_regularisation, _effective_L_max 'md')
    df_kl0 = dinv.optim.StackedPhysicsDataFidelity(
        [PoissonLikelihoodNoBkg(gain=1 / flux) for _ in eta_vec])
    L_kl0 = sum(float(y_j.abs().sum()) for y_j in y) * flux      # y_j / gain = counts
    # <<<<<<<<<< MD ADDED <<<<<<<<<<

    sum_y = torch.stack(list(y)).sum(0)           # sum over detectors
    return {"PSF": PSF, "y": y, "avg_y": sum_y / N_DET, "sum_y": sum_y,
            "finger_print": finger_print, "physics": physics,
            "df_kl": df_kl, "df_l2": df_l2, "L_kl": L_kl, "L_l2": L_l2,
            "df_kl0": df_kl0, "L_kl0": L_kl0,                    # MD ADDED: df_kl0, L_kl0
            "eta_vec": eta_vec, "flux": flux}                    # BATCHED ADDED


# >>>>>>>>>> BATCHED ADDED >>>>>>>>>>
# The 25 detectors as ONE operator. StackedLinearPhysics applies 25 BlurFFT one after the
# other (25 small FFTs + their elementwise ops, each a separate GPU kernel): on 256 x 256
# images the GPU is then idle most of the time, waiting for Python to launch kernels.
# Here every detector is a circular convolution, so it is fully described by its
# transfer function H_j = rfft2(A_j delta) (delta = impulse at pixel (0, 0)):
#     A x   = irfft2( sum_p H_jp rfft2(x_p) )               for all j at once
#     A^T y = irfft2( sum_j conj(H_jp) rfft2(y_j) )         for all planes p at once
# H is computed FROM the stacked operators, so this is the same operator (up to
# float rounding); 3D (SummedBlurPhysics, 2 planes) is the case Nz = 2.

class BatchedISMPhysics(dinv.physics.LinearPhysics):
    """All detectors at once: x (B, Nz, N, N) -> y (B, N_DET, N, N)."""

    def __init__(self, physics_list, img_shape, device):
        super().__init__()
        nz, n, m = img_shape
        H = []
        for p_j in physics_list:
            row = []
            for p in range(nz):
                delta = torch.zeros((1, nz, n, m), device=device)
                delta[0, p, 0, 0] = 1.0
                row.append(torch.fft.rfft2(p_j.A(delta))[0, 0])     # (N, N//2 + 1)
            H.append(torch.stack(row))
        self.register_buffer("H", torch.stack(H))                    # (N_DET, Nz, N, N//2+1)
        self.img_shape = (n, m)

    def A(self, x, **kwargs):
        X = torch.fft.rfft2(x)                                       # (B, Nz, N, M)
        if self.H.shape[1] == 1:
            Y = X * self.H[:, 0]                                     # (B, N_DET, N, M)
        else:
            Y = (X.unsqueeze(1) * self.H).sum(2)
        return torch.fft.irfft2(Y, s=self.img_shape)

    def A_adjoint(self, y, **kwargs):
        Yf = torch.fft.rfft2(y)                                      # (B, N_DET, N, M)
        X = (Yf.unsqueeze(2) * self.H.conj()).sum(1)                 # (B, Nz, N, M)
        return torch.fft.irfft2(X, s=self.img_shape)


def batch_measurements(meas):
    """meas of _finish in batched form: y (1, N_DET, N, N), BatchedISMPhysics and the KL /
    L2 / KL0 fidelities with the per-detector background as a (1, N_DET, 1, 1) tensor.
    Kept from the stacked version: L_kl / L_l2 / L_kl0 (same values), sum_y, avg_y, PSF;
    the stacked y and physics stay available as y_stacked / physics_stacked."""
    y_list, physics = meas["y"], meas["physics"]
    flux = meas["flux"]
    x_shape = physics.physics_list[0].A_adjoint(y_list[0]).shape[1:]          # (Nz, N, N)
    eta = torch.as_tensor(meas["eta_vec"]).reshape(1, -1, 1, 1).to(y_list[0])
    out = dict(meas)
    out.update(
        y=torch.cat(list(y_list), dim=1),
        physics=BatchedISMPhysics(physics.physics_list, x_shape, y_list[0].device),
        df_kl=PoissonLikelihood(gain=1 / flux, bkg=eta * flux),               # as kl_with_background
        df_l2=L2WithBackground(sigma=1.0, bkg=eta),
        df_kl0=PoissonLikelihoodNoBkg(gain=1 / flux),
        y_stacked=y_list, physics_stacked=physics, batched=True)
    return out
# <<<<<<<<<< BATCHED ADDED <<<<<<<<<<
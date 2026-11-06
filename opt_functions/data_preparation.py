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
    """Same seed -> same noise realization for every run at this (data, flux)."""
    torch.manual_seed(cfg.seed)
    generate = generate_meas_ism_3D if cfg.is_3d else generate_meas_ism
    return generate(x, cfg.Nx, cfg.Nz, cfg.pxsizex, flux, cfg.device)


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


def _finish(x, PSF, physics_list, finger_print, eta_vec, flux, device):
    """Simulate y, build the data fidelities and the Lipschitz constants."""
    physics = dinv.physics.StackedLinearPhysics(physics_list, device=device)
    y = physics(x)

    df_kl = dinv.optim.StackedPhysicsDataFidelity(
        [PoissonLikelihood(gain=1 / flux, bkg=eta) for eta in eta_vec])
    df_l2 = dinv.optim.StackedPhysicsDataFidelity([L2(sigma=1.0) for _ in physics_list])

    # Upper bounds of Lip(grad f), summed over detectors: ||A||^2 <= max(A 1) * max(A^T 1)
    L_kl, L_l2 = 0.0, 0.0
    ones = torch.ones_like(x)
    for A_j, y_j, eta_j in zip(physics_list, y, eta_vec):
        norm_A = A_j.A(ones).max() * A_j.A_adjoint(torch.ones_like(y_j)).max()
        L_kl = L_kl + y_j.abs().max() / eta_j ** 2 * norm_A
        L_l2 = L_l2 + norm_A                      # L2 with sigma = 1

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
            "df_kl0": df_kl0, "L_kl0": L_kl0}                    # MD ADDED: df_kl0, L_kl0
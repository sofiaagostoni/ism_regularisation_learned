# %%
"""
Generatore di phantom con parametri casuali per ogni struttura.

Rispetto alla versione precedente, generate_phantom accetta un dizionario `params`
(se manca, usa esattamente i valori di prima), e make_one_phantom ne estrae uno
nuovo per ogni immagine con sample_params(): numero, dimensione, spessore e
intensita' di ogni struttura cambiano da un phantom all'altro.

Le funzioni di phantom_simulator vengono chiamate con gli stessi argomenti di prima,
cambiano solo i valori.
"""
import os
import ISM.simulation.phantom_simulator as ph
import numpy as np
import torch
import ISM.simulation.Tubulin_sim as st
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

_STRUCTURES = ["tubulin", "balls", "sparse", "nucleus", "membrane", "mitochondria"]
_OTHERS = [s for s in _STRUCTURES if s != "tubulin"]   # strutture di "contorno"

# Valori fissi della versione precedente (usati se params non viene passato)
DEFAULT_PARAMS = {
    "tub_intensity": [0.2, 1],
    "balls_num": 10, "balls_radius": 20, "balls_range": (0.5, 1.0),
    "sparse_num": 20, "sparse_range": (0.5, 1.0),
    "nucleus_gauss": 300, "nucleus_range": (0.5, 0.8), "nucleus_std": (2, 6),
    "nucleus_axes": (60, 30),
    "membrane_radius": (65, 35), "membrane_thickness": 5.0,
    "membrane_range": (0.5, 0.8), "membrane_blur": 1.5,
    "mito_num": 20, "mito_radius": (8, 6), "mito_thickness": 1.0,
    "mito_range": (0.8, 1.0), "mito_blur": 1.0,
}


def _vrange(rng, lo_min, lo_max, width_min=0.2, width_max=0.5):
    """Intervallo di intensita' casuale (lo, hi) dentro [0, 1]."""
    lo = float(rng.uniform(lo_min, lo_max))
    hi = min(1.0, lo + float(rng.uniform(width_min, width_max)))
    return (lo, hi)


def sample_params(rng, Nx=256):
    """Parametri casuali per un phantom. Gli intervalli sono pensati per Nx=256:
    se cambi Nx, le dimensioni in pixel scalano di conseguenza."""
    s = Nx / 256
    a_nuc = rng.uniform(35, 80) * s
    a_mem = rng.uniform(35, 90) * s
    r_mito = int(rng.integers(5, 13))
    return {
        # tubulina: da filamenti quasi invisibili a tutti brillanti
        "tub_intensity": [float(rng.uniform(0.05, 0.5)), 1],
        # palle: da poche e grandi a tante e piccole
        "balls_num": int(rng.integers(2, 26)),
        "balls_radius": int(rng.integers(6, 26) * s),
        "balls_range": _vrange(rng, 0.2, 0.7),
        # punti sparsi
        "sparse_num": int(rng.integers(5, 61) * s ** 2),
        "sparse_range": _vrange(rng, 0.2, 0.7),
        # nucleo: dimensione, forma e texture
        "nucleus_gauss": int(rng.integers(100, 501)),
        "nucleus_range": _vrange(rng, 0.3, 0.6, 0.1, 0.4),
        "nucleus_std": (2, int(rng.integers(4, 9))),
        "nucleus_axes": (int(a_nuc), int(a_nuc * rng.uniform(0.35, 0.9))),
        # membrana: dimensione, forma e spessore
        "membrane_radius": (int(a_mem), int(a_mem * rng.uniform(0.35, 0.9))),
        "membrane_thickness": float(rng.uniform(2.0, 7.0)),
        "membrane_range": _vrange(rng, 0.3, 0.7, 0.1, 0.4),
        "membrane_blur": float(rng.uniform(0.8, 2.0)),
        # mitocondri: numero, dimensione, spessore
        "mito_num": int(rng.integers(5, 41)),
        "mito_radius": (r_mito, int(rng.integers(3, r_mito + 1))),
        "mito_thickness": float(rng.uniform(0.8, 2.0)),
        "mito_range": _vrange(rng, 0.4, 0.8, 0.1, 0.4),
        "mito_blur": float(rng.uniform(0.5, 1.5)),
    }


def generate_phantom(phantom_type, Nx, Ny, Nz, pxsizex, n_filament=20, radius_scale=1.0, params=None):
    P = dict(DEFAULT_PARAMS)
    if params:
        P.update(params)

    phantom = np.zeros([Nx, Nx], dtype=np.float32)
    shape = (Nx, Nx)

    if 'tubulin' in phantom_type or 'mixture' in phantom_type:
        tubulin_planar = st.tubSettings()
        tubulin_planar.xy_pixel_size = pxsizex
        tubulin_planar.xy_dimension = Nx
        tubulin_planar.xz_dimension = 1
        tubulin_planar.z_pixel = 1
        tubulin_planar.n_filament = n_filament
        tubulin_planar.radius_filament = pxsizex * radius_scale
        tubulin_planar.intensity_filament = P["tub_intensity"]
        phTub = np.zeros([Nz, Nx, Nx])
        for i in range(Nz):
            phTub_planar = st.functionPhTub(tubulin_planar)
            phTub_planar = np.swapaxes(phTub_planar, 2, 0)
            phTub[i, :, :] = phTub_planar * (np.power(3, np.abs(i)))
        phantom += phTub.sum(axis=0)

    if 'balls' in phantom_type or 'mixture' in phantom_type:
        phantom += ph.generate_disk_phantom(shape, P["balls_num"], P["balls_radius"],
                                            value_range=P["balls_range"], smooth=True)

    if 'sparse' in phantom_type or 'mixture' in phantom_type:
        phantom += ph.sparse_random_image(shape, P["sparse_num"],
                                          value_range=P["sparse_range"], margin=2)

    if 'nucleus' in phantom_type or 'mixture' in phantom_type:
        phantom += ph.Nucleus(shape, P["nucleus_gauss"], P["nucleus_range"], P["nucleus_std"],
                              P["nucleus_axes"], background_noise=0.05, seed=None)

    if 'membrane' in phantom_type or 'mixture' in phantom_type:
        phantom += ph.generate_membrane_phantom(shape, P["membrane_radius"], P["membrane_thickness"],
                                                P["membrane_range"], noise_level=0.01,
                                                blur_sigma=P["membrane_blur"])

    if 'mitochondria' in phantom_type or 'mixture' in phantom_type:
        phantom += ph.generate_hollow_membrane_phantom(shape, P["mito_num"], P["mito_radius"],
                                                       P["mito_range"], P["mito_thickness"],
                                                       P["mito_blur"], 0.01)

    phantom = torch.from_numpy(phantom).float()
    phantom = phantom.unsqueeze(0).unsqueeze(0)
    return phantom


def save_phantom(name, phantom_type, Nx, Ny, Nz, pxsizex,
                 n_filament=5, radius_scale=1.0, out_dir="data", metadata=True, params=None):
    phantom = generate_phantom(phantom_type, Nx, Ny, Nz, pxsizex,
                               n_filament=n_filament, radius_scale=radius_scale, params=params)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pth")
    if metadata:
        torch.save({
            "phantom": phantom, "phantom_type": phantom_type,
            "shape": tuple(phantom.shape), "Nx": Nx, "Ny": Ny, "Nz": Nz,
            "pxsizex": pxsizex, "n_filament": n_filament, "radius_scale": radius_scale,
            "params": params,
        }, path)
    else:
        torch.save(phantom, path)
    print(f"Salvato {path}  |  shape {tuple(phantom.shape)}")
    return path


# %%
def sample_phantom_type(rng, p_tubulin=0.80):
    """Tipo di phantom, sbilanciato verso la tubulina (fino a 3 strutture extra)."""
    if rng.random() < 0.10:
        return "mixture"                                   # tutto insieme
    if rng.random() < p_tubulin:
        if rng.random() < 0.4:
            return "tubulin"                               # tubulina pura
        k = int(rng.integers(1, 4))                        # 1-3 strutture extra
        extra = rng.choice(_OTHERS, size=k, replace=False)
        return "_".join(["tubulin", *extra])
    k = int(rng.integers(1, 4))                            # senza tubulina, per varieta'
    return "_".join(rng.choice(_OTHERS, size=k, replace=False))


def make_one_phantom(Nx=256, pxsizex=25.0, rng=None, augment=True, max_tries=10):
    """Un phantom pulito con parametri casuali, normalizzato in [0, 1], shape (1, Nx, Nx)."""
    rng = rng or np.random.default_rng()

    for _ in range(max_tries):
        ph_type = sample_phantom_type(rng)
        x = generate_phantom(ph_type, Nx, Nx, 1, pxsizex,
                             n_filament=int(rng.integers(3, 41)),       # da sparsa a molto densa
                             radius_scale=float(rng.uniform(0.6, 1.8)),
                             params=sample_params(rng, Nx))
        x = x.squeeze(0)                                    # (1, Nx, Nx)
        if x.max() > 0:
            break                                           # scarta i phantom vuoti
    x = x / x.max() if x.max() > 0 else x

    if augment:
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[-1])
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[-2])
        x = torch.rot90(x, k=int(rng.integers(0, 4)), dims=[-2, -1])

    return x.contiguous().float()


class PhantomDataset(Dataset):
    """Genera phantom puliti on-the-fly. Il rumore viene aggiunto nel training loop."""

    def __init__(self, length, Nx=256, pxsizex=25.0, seed=None, augment=True):
        self.length = length
        self.Nx = Nx
        self.pxsizex = pxsizex
        self.seed = seed
        self.augment = augment

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seed = None if self.seed is None else self.seed + idx
        rng = np.random.default_rng(seed)
        # phantom_simulator usa np.random globale: seed anche quello per la riproducibilita'
        if seed is not None:
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
        x = make_one_phantom(self.Nx, self.pxsizex, rng, self.augment)
        return x, x


def pregenerate_dataset(n, Nx=256, pxsizex=25.0, seed=0, path=None):
    """Genera n phantom una volta sola e li restituisce come tensore (n, 1, Nx, Nx)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)            # anche per le funzioni che usano np.random globale
    out = torch.empty(n, 1, Nx, Nx)
    for i in range(n):
        out[i] = make_one_phantom(Nx, pxsizex, rng)
        if (i + 1) % 25 == 0:
            print(f"  generati {i+1}/{n} phantom")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(out, path)
        print(f"Dataset salvato in {path}")
    return out


def inspect_dataset(data, n_show=12, ncols=4, cmap="hot", random_pick=True, seed=0):
    """Griglia di phantom dal dataset + statistiche. data: tensore (N, 1, Nx, Nx)."""
    N = data.shape[0]
    n_show = min(n_show, N)

    print(f"Shape: {tuple(data.shape)} | dtype: {data.dtype}")
    print(f"min: {data.min():.4f} | max: {data.max():.4f} | mean: {data.mean():.4f}")
    print(f"NaN: {torch.isnan(data).any().item()} | Inf: {torch.isinf(data).any().item()}")
    print(f"pixel a zero: {(data == 0).float().mean():.1%}")
    per_img_max = data.amax(dim=(1, 2, 3))
    print(f"max per immagine -> min {per_img_max.min():.4f}, max {per_img_max.max():.4f}")
    n_vuote = (per_img_max == 0).sum().item()
    if n_vuote:
        print(f"  ATTENZIONE: {n_vuote} immagini completamente vuote!")
    # quanto e' "piena" ogni immagine: utile per controllare la varieta' di densita'
    fg = (data > 0.1).float().mean(dim=(1, 2, 3))
    print(f"frazione di struttura (>0.1) per immagine: "
          f"min {fg.min():.1%}, mediana {fg.median():.1%}, max {fg.max():.1%}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n_show, replace=False) if random_pick else np.arange(n_show)
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idx):
        img = data[i, 0].cpu().numpy()
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(f"#{i} | max {img.max():.2f} | mean {img.mean():.3f}", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[n_show:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# %%


# N_TRAIN, N_VAL = 2000, 300

# data = pregenerate_dataset(
#     N_TRAIN + N_VAL,
#     Nx=256,
#     pxsizex=25.0,
#     seed=1,   # different from the old dataset, which used seed=0
#     path="training_drunet/training_dataset/varied_dataset_v2.pth",
# )

# #%%
data = torch.load("training_drunet/training_dataset/varied_dataset_v2.pth")
inspect_dataset(data, n_show=32)
# %%

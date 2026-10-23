"""
Esporta il dataset di phantom (.pth, lo stesso usato per DRUNet) nei file .h5
letti dal training del weakly convex ridge regularizer (WCRR).

Lancia da ism_regularisation_learned/:
    python export_wcrr_h5.py

Output:
    training_wcrr/data/train_v2.h5        patch 40x40 a zoom diversi, chiavi "0","1",...
    training_wcrr/data/validation_v2.h5   immagini intere (zoom 1), chiavi "0","1",...
Ogni elemento: float32, shape (1, H, W), valori in [0, 1].
Ogni patch di training ha gli attributi "scale" (zoom) e "source_image".

Zoom continuo:
  per ogni patch si estrae uno zoom z a caso (log-uniforme tra ZOOM_MIN e ZOOM_MAX),
  si ritaglia una regione di lato 40/z dall'immagine originale e la si ridimensiona a 40x40.
    z = 0.35 -> regione 114x114 rimpicciolita: tante strutture piccole e sottili
    z = 1    -> ritaglio diretto 40x40
    z = 2.5  -> regione 16x16 ingrandita: poche strutture grandi e spesse
  Log-uniforme = tanti zoom-out quanti zoom-in (0.5 e 2 sono "equidistanti" da 1).
"""

import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------
SRC        = "training_drunet/training_dataset/varied_dataset.pth"   # stesso .pth di DRUNet
OUT_DIR    = "training_wcrr/data"
TRAIN_NAME = "train.h5"
VAL_NAME   = "validation.h5"

N_TRAIN  = 1000     # i primi N_TRAIN sono training (stesso split di DRUNet)
N_VAL    = 50       # sottoinsieme della validazione DRUNet (WCRR valida lentamente)

PATCH    = 40

ZOOM_MIN          = 0.25   # sotto ~0.3 i filamenti sottili diventano quasi invisibili (mediati col fondo)
ZOOM_MAX          = 1.5    # sopra ~2.5 si vedono i "gradini" dei pixel dei phantom, molto ingranditi
PATCHES_PER_IMAGE = 40     # totale massimo: N_TRAIN * PATCHES_PER_IMAGE (qui 40.000)
TRIES_PER_PATCH   = 20     # posizioni provate (a zoom fisso) prima di rinunciare a quella patch

# Filtro sul contenuto. Una patch e' "struttura" se:
FG_THR   = 0.10     # ...i pixel sopra questa soglia (sopra la texture di fondo ~0.05)
MIN_FG   = 0.05     # ...sono almeno il 5% della patch (80 pixel su 1600)
MIN_STD  = 0.03     # ...e la patch non e' piatta (esclude zone uniformi, es. interno di una palla)
KEEP_BG  = 0.02     # se a quello zoom non si trova struttura, la patch di fondo si tiene col 2%
SEED     = 0

# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

data = torch.load(SRC).float()
assert data.dim() == 4 and data.shape[1] == 1, f"shape inattesa: {tuple(data.shape)}"
print(f"dataset: {tuple(data.shape)}, range [{data.min():.4f}, {data.max():.4f}]")
if data.min() < 0 or data.max() > 1.0 + 1e-6:
    raise ValueError("I dati devono stare in [0,1] (stessa scala usata per DRUNet).")
if data.shape[0] < N_TRAIN + N_VAL:
    raise ValueError(f"servono almeno {N_TRAIN + N_VAL} immagini, il file ne ha {data.shape[0]}")

train = data[:N_TRAIN]                     # tensore (N, 1, H, W)
val   = data[N_TRAIN:N_TRAIN + N_VAL].numpy()
H, W  = train.shape[-2:]


def sample_zoom():
    """Zoom log-uniforme; restituisce (zoom effettivo, lato della regione da ritagliare)."""
    z = float(np.exp(rng.uniform(np.log(ZOOM_MIN), np.log(ZOOM_MAX))))
    c = int(round(PATCH / z))
    c = max(4, min(c, H, W))
    return PATCH / c, c


def crop_resized(img, c):
    """Regione c x c in posizione casuale, ridimensionata a PATCH x PATCH -> array (1, 40, 40).
    Bicubica con antialias: in riduzione media i pixel (niente aliasing), in ingrandimento
    interpola in modo liscio. Il clamp toglie i piccoli overshoot della bicubica."""
    i = int(rng.integers(0, H - c + 1))
    j = int(rng.integers(0, W - c + 1))
    crop = img[..., i:i + c, j:j + c]                   # (1, 1, c, c)
    if c != PATCH:
        crop = F.interpolate(crop, size=(PATCH, PATCH), mode="bicubic",
                             align_corners=False, antialias=True).clamp(0, 1)
    return crop[0].numpy()


def augment(p):
    """Una delle 8 trasformazioni diedrali, scelta a caso."""
    p = np.rot90(p, k=rng.integers(4), axes=(-2, -1))
    if rng.random() < 0.5:
        p = np.flip(p, axis=-1)
    return np.ascontiguousarray(p)


def is_structure(p):
    return (p > FG_THR).mean() >= MIN_FG and p.std() >= MIN_STD


# ---- training: patch a zoom continuo ----
train_path = os.path.join(OUT_DIR, TRAIN_NAME)
zooms, kinds = [], []
n_skipped = 0
with h5py.File(train_path, "w") as f:
    for k, img in enumerate(train):
        img = img[None]                                     # (1, 1, H, W)
        for _ in range(PATCHES_PER_IMAGE):
            z, c = sample_zoom()
            # lo zoom resta fisso: si cambia solo la posizione finche' non si trova struttura,
            # cosi' gli zoom grandi (che vedono poco campo, spesso vuoto) non vengono scartati
            for _try in range(TRIES_PER_PATCH):
                p = crop_resized(img, c)
                if is_structure(p):
                    kind = "struct"
                    break
            else:
                if rng.random() >= KEEP_BG:
                    n_skipped += 1
                    continue
                kind = "bg"
            n = len(zooms)
            ds = f.create_dataset(str(n), data=augment(p).astype(np.float32))
            ds.attrs["scale"] = z
            ds.attrs["source_image"] = k
            zooms.append(z)
            kinds.append(kind)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(train)} immagini, {len(zooms)} patch")

zooms, kinds = np.array(zooms), np.array(kinds)
print(f"\ntrain patches: {len(zooms)}  (saltate: {n_skipped}, "
      f"solo fondo: {(kinds == 'bg').mean():.1%})  -> {os.path.abspath(train_path)}")

# distribuzione degli zoom: dovrebbe essere circa uniforme sulle fasce
edges = np.geomspace(ZOOM_MIN, ZOOM_MAX, 8)
counts, _ = np.histogram(zooms, bins=edges)
print("patch per fascia di zoom:")
for lo, hi, cnt in zip(edges[:-1], edges[1:], counts):
    print(f"  {lo:5.2f} - {hi:5.2f}: {cnt:6d}  {'#' * int(50 * cnt / max(counts.max(), 1))}")

# ---- validazione: immagini intere, zoom 1 ----
val_path = os.path.join(OUT_DIR, VAL_NAME)
with h5py.File(val_path, "w") as f:
    for k, img in enumerate(val):
        f.create_dataset(str(k), data=img.astype(np.float32))
print(f"\nvalidation images: {len(val)}  -> {os.path.abspath(val_path)}")

# ---- controllo: rileggi un elemento per file ----
for path in (train_path, val_path):
    with h5py.File(path, "r") as f:
        k = list(f.keys())[0]
        x = f[k][()]
        print(f"{os.path.basename(path)}: {len(f.keys())} elementi, "
              f"es. '{k}' shape {x.shape} {x.dtype} range [{x.min():.3f}, {x.max():.3f}]")
"""
256 x 256 piecewise-constant phantom for comparing KL and L2 data fidelities.

A grey disc on a dim (non-zero) background, plus features spanning the intensity
range, so that Poisson noise varies strongly across the image:
  - dim dots of increasing contrast on the background (low counts: where KL and
    L2 should differ most)
  - bright dots of different sizes inside the disc (high counts)
  - dark holes inside the disc (contrast going the other way)
Hard edges keep few distinct values, so error_vs_int.py can use SCHEME = "levels".
Saved as (1, 1, 256, 256) float32 with max = 1 (load_data normalizes by the max anyway).
"""
import os
import torch

N = 256
BACKGROUND = 0.03
DISC = 0.35


def disc(img, cy, cx, r, value):
    yy, xx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = value


def make_phantom():
    x = torch.full((N, N), BACKGROUND)
    disc(x, 128, 128, 80, DISC)                               # main grey disc
    for cx, r in zip((98, 132, 168), (4, 7, 11)):             # bright dots, increasing size
        disc(x, 100, cx, r, 1.0)
    for cx, r in zip((105, 155), (6, 10)):                    # dark holes in the disc
        disc(x, 165, cx, r, 0.10)
    for i, v in enumerate((0.05, 0.08, 0.12, 0.18)):          # dim dots on the background
        disc(x, 18, 40 + 58 * i, 5, v)                        # top row: r = 5
        disc(x, 238, 40 + 58 * i, 3, v)                       # bottom row: r = 3
    return x[None, None]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = make_phantom()
    out = os.path.join("Data", "Simul_data", "circle_phantom.pth")
    torch.save(x, out)
    print(f"saved {out}  shape={tuple(x.shape)}  levels={torch.unique(x).tolist()}")
    plt.imshow(x[0, 0], cmap="hot"); plt.colorbar(); plt.title("circle_phantom")
    plt.show()

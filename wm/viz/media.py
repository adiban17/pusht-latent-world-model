"""GIFs, filmstrips and histograms for data inspection."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from wm.envs.pusht import block_cog, wrap_angles_rad


def upscale(frames: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upscale of [..., H, W, 3] frames."""
    return frames.repeat(scale, axis=-3).repeat(scale, axis=-2)


def save_gif(frames: np.ndarray, path: str | Path, fps: int = 10, scale: int = 2) -> Path:
    """Write uint8 frames [T, H, W, 3] as a looping GIF."""
    imageio.mimsave(path, list(upscale(frames, scale)), duration=1000 / fps, loop=0)
    return Path(path)


def filmstrip(frames: np.ndarray, n: int = 8, scale: int = 2, gap: int = 2) -> np.ndarray:
    """n evenly spaced frames side by side (white gaps), uint8 [H*s, n*W*s + (n-1)*gap, 3]."""
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tiles = upscale(frames[idx], scale)
    h, w = tiles.shape[1:3]
    out = np.full((h, n * w + (n - 1) * gap, 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        out[:, i * (w + gap): i * (w + gap) + w] = tile
    return out


def save_histograms(arrays: dict[str, np.ndarray], path: str | Path) -> Path:
    """Histograms of block centre x/y, block angle, action x/y and per-step block movement."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = arrays["states"]
    cog = block_cog(states)
    last = arrays["episode_starts"] + arrays["episode_lengths"] - 1
    keep = np.ones(len(states), bool)
    keep[last] = False
    move = np.linalg.norm(cog[1:] - cog[:-1], axis=1)[keep[:-1]]
    panels = [
        ("block centre x (px)", cog[:, 0]), ("block centre y (px)", cog[:, 1]),
        ("block angle (deg)", np.degrees(wrap_angles_rad(states[:, 4]))),
        ("action x (px)", arrays["actions"][keep, 0]), ("action y (px)", arrays["actions"][keep, 1]),
        ("block movement per step (px)", move),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (title, values) in zip(axes.flat, panels):
        ax.hist(values, bins=40, color="tab:blue")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return Path(path)

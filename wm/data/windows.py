"""GPU-resident window sampler for the world model (CLAUDE.md §4 "Data loading").

The whole npz goes onto the device once as uint8; batches are gathered there by index.
No DataLoader. A window starting at frame s holds frames s..s+W-1 and actions s..s+W-2
(action j is the one applied at frame j), all inside one episode.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wm.data.collect import seed_list


def frames_to_model(frames_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [..., 64, 64, 3] -> float [..., 3, 64, 64] in [-0.5, 0.5] (same as normalize_image)."""
    return frames_u8.movedim(-1, -3).float() / 255.0 - 0.5


def window_starts(starts: np.ndarray, lengths: np.ndarray, window: int) -> np.ndarray:
    """All global frame indices s where a W-frame window fits inside its episode."""
    out = [np.arange(s, s + n - window + 1) for s, n in zip(starts, lengths) if n >= window]
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)


class WindowData:
    """Train/val windows of a pusht_*.npz dict, split by episode seed."""

    def __init__(self, arrays: dict[str, np.ndarray], device, val_seeds: list[int],
                 window: int = 13):
        self.window, self.device = window, torch.device(device)
        seeds = arrays["episode_seeds"]
        starts, lengths = arrays["episode_starts"], arrays["episode_lengths"]
        is_val = np.isin(seeds, seed_list(val_seeds))
        if is_val.any():
            train_eps, val_eps = ~is_val, is_val
            self.val_source = "val episodes (seeds {}-{})".format(*val_seeds)
        else:  # smoke data (seeds 0-1) has no val episodes: validate on the last train episode
            train_eps = np.ones(len(seeds), bool)
            val_eps = np.zeros(len(seeds), bool)
            val_eps[-1] = True
            self.val_source = f"train episode seed {int(seeds[-1])} (smoke: no val episodes)"
        self.train_seeds = seeds[train_eps].tolist()
        self.val_seeds = seeds[val_eps].tolist()
        self.train_starts = torch.from_numpy(window_starts(starts[train_eps], lengths[train_eps], window))
        self.val_starts = torch.from_numpy(window_starts(starts[val_eps], lengths[val_eps], window))
        self.val_frame_idx = torch.from_numpy(np.concatenate(
            [np.arange(s, s + n) for s, n in zip(starts[val_eps], lengths[val_eps])]))
        self.frames = torch.from_numpy(arrays["frames"]).to(self.device)            # uint8
        acts = arrays["actions"].astype(np.float32) / 256.0 - 1.0                   # normalize_pos
        self.actions = torch.from_numpy(acts).to(self.device)
        self.offsets = torch.arange(window, device=self.device)

    def gather(self, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Window start indices [B] -> frames [B, W, 3, 64, 64] float, actions [B, W-1, 2]."""
        idx = starts.to(self.device)[:, None] + self.offsets
        return frames_to_model(self.frames[idx]), self.actions[idx[:, :-1]]

    def sample(self, batch: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Random train windows; the (CPU) generator makes batches reproducible and resumable."""
        pick = torch.randint(len(self.train_starts), (batch,), generator=generator)
        return self.gather(self.train_starts[pick])

    def fixed_val_starts(self, n: int, seed: int = 0) -> torch.Tensor:
        """A fixed, seeded subset of n val window starts (all of them if fewer), sorted."""
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(self.val_starts), generator=g)[:n]
        return self.val_starts[perm].sort().values

    def train_mean_frame(self) -> torch.Tensor:
        """Mean normalised frame [3, 64, 64] over all frames of the train episodes."""
        idx = torch.unique((self.train_starts[:, None] + torch.arange(self.window)).flatten())
        total = torch.zeros(3, 64, 64, device=self.device)
        for chunk in idx.split(4096):
            total += frames_to_model(self.frames[chunk.to(self.device)]).sum(0)
        return total / len(idx)

    def val_frames(self) -> torch.Tensor:
        """Every frame of the val episodes, [M, 3, 64, 64] float (for the NMSE reference z-bar)."""
        return frames_to_model(self.frames[self.val_frame_idx.to(self.device)])

    def summary(self) -> dict[str, Any]:
        return {"train_windows": len(self.train_starts), "val_windows_total": len(self.val_starts),
                "train_episodes": len(self.train_seeds), "val_episodes": len(self.val_seeds),
                "val_source": self.val_source}

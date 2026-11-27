"""Frames for the fair image judge `probe_mix` (Phase 5b, pre-registered judge decision).

Four frame types, all labelled with the TRUE state of the frame they show:
  real           o_t                                    label s_t
  wm_recon       clip(D_wm(E_wm(o_t)))                  label s_t
  wm1step_recon  clip(D_wm1step(E_wm1step(o_t)))        label s_t
  pixel_1step    clip(pixel(o_{t-2..t}, a_t))           label s_{t+1}   (t must have an action;
                 context frames before the episode start repeat its first frame, as in §8)
Derived frames are made on the fly under no_grad with the frozen models in eval mode.
Every batch has the same share of each type. `shift_with_labels` translates images by up to
±pad px (replicate pad + crop) and moves the position labels with them (8 world px per 64-px
pixel); the angle is unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wm.data.windows import episode_frames, frames_to_model, split_episodes, state_targets
from wm.eval.rollouts import clip_frames

FRAME_TYPES = ("real", "wm_recon", "wm1step_recon", "pixel_1step")
WORLD_PX_PER_CELL = 8.0           # 512 px table drawn on 64 px
NORM_PER_CELL = WORLD_PX_PER_CELL / 256.0   # one image pixel in normalised position units


def select_episodes(arrays: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    """Arrays restricted to the episodes where `keep` is True (frame-level arrays re-packed)."""
    starts, lengths = arrays["episode_starts"][keep], arrays["episode_lengths"][keep]
    idx = episode_frames(starts, lengths)
    out = {k: arrays[k][idx] for k in ("frames", "states", "actions")}
    out["episode_lengths"] = lengths.astype(np.int64)
    out["episode_starts"] = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    out["episode_seeds"] = arrays["episode_seeds"][keep]
    return out


def concat_arrays(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Concatenate pusht_*.npz-style arrays (episode starts re-offset)."""
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("frames", "states", "actions",
                                                               "episode_lengths", "episode_seeds")}
    out["episode_starts"] = np.concatenate([[0], np.cumsum(out["episode_lengths"])[:-1]]).astype(np.int64)
    return out


def train_split(arrays: dict[str, np.ndarray], val_seeds: list[int]) -> dict[str, np.ndarray]:
    """The train-split episodes of a real pusht_train.npz (val seeds removed)."""
    train_eps, _, _ = split_episodes(arrays["episode_seeds"], val_seeds)
    return select_episodes(arrays, train_eps)


def val_split(arrays: dict[str, np.ndarray], val_seeds: list[int]) -> dict[str, np.ndarray]:
    """The val episodes of a real pusht_train.npz (smoke: the last train episode)."""
    _, val_eps, _ = split_episodes(arrays["episode_seeds"], val_seeds)
    return select_episodes(arrays, val_eps)


def shift_with_labels(images: torch.Tensor, targets: torch.Tensor, pad: int,
                      generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random ±pad px translation of images [B, 3, H, W], one per sample, with the position
    targets (pusher xy, block xy: columns 0-3, normalised) moved by the same amount."""
    if pad <= 0:
        return images, targets
    b, c, h, w = images.shape
    dev = images.device
    padded = torch.nn.functional.pad(images, (pad,) * 4, mode="replicate")
    oy, ox = torch.randint(0, 2 * pad + 1, (2, b), generator=generator).to(dev)
    rows = oy[:, None] + torch.arange(h, device=dev)
    cols = ox[:, None] + torch.arange(w, device=dev)
    bi = torch.arange(b, device=dev)[:, None, None]
    out = padded[bi, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)   # [B, C, H, W]
    # content at original (row r, col c) lands at (r + pad - oy, c + pad - ox)
    d = torch.stack([pad - ox, pad - oy], 1).to(targets.dtype) * NORM_PER_CELL    # [B, 2] (x, y)
    t = targets.clone()
    t[:, 0:2] += d
    t[:, 2:4] += d
    return out, t


class MixData:
    """Frame-type sampler over one set of episodes (pusht_*.npz layout), GPU-resident."""

    def __init__(self, arrays: dict[str, np.ndarray], device, models: dict[str, torch.nn.Module],
                 context: int = 3):
        self.device, self.models, self.context = torch.device(device), models, context
        starts, lengths = arrays["episode_starts"], arrays["episode_lengths"]
        ep_first = np.repeat(starts, lengths)
        ep_last = np.repeat(starts + lengths - 1, lengths)
        n = len(ep_first)
        self.frame_idx = torch.arange(n)
        self.action_idx = torch.from_numpy(np.arange(n)[np.arange(n) < ep_last])
        self.ep_first = torch.from_numpy(ep_first).to(self.device)
        self.frames = torch.from_numpy(arrays["frames"]).to(self.device)
        self.states = np.asarray(arrays["states"])
        self.targets = torch.from_numpy(state_targets(self.states)).to(self.device)
        self.actions = torch.from_numpy(arrays["actions"].astype(np.float32) / 256.0 - 1.0).to(self.device)
        self.ctx_offsets = torch.arange(-(context - 1), 1, device=self.device)

    def pool(self, kind: str) -> torch.Tensor:
        """Valid frame indices t for a frame type (pixel_1step needs an action at t)."""
        return self.action_idx if kind == "pixel_1step" else self.frame_idx

    @torch.no_grad()
    def images(self, kind: str, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """(frames [n, 3, 64, 64] float, targets [n, 6], true states [n, 5]) of type `kind` at t."""
        t = t.to(self.device)
        if kind == "pixel_1step":
            ctx_idx = torch.maximum(t[:, None] + self.ctx_offsets, self.ep_first[t][:, None])
            pix = self.models["pixel"].eval()
            x = clip_frames(pix(frames_to_model(self.frames[ctx_idx]), self.actions[t]))
            label = t + 1
        else:
            x = frames_to_model(self.frames[t])
            if kind != "real":
                m = self.models["wm" if kind == "wm_recon" else "wm_1step"].eval()
                x = clip_frames(m.decode(m.encode(x)))
            label = t
        return x.float(), self.targets[label], self.states[label.cpu().numpy()]

    def sample(self, batch: int, gen: torch.Generator, pad: int = 0,
               kinds: tuple[str, ...] = FRAME_TYPES) -> tuple[torch.Tensor, torch.Tensor]:
        """A balanced random batch over `kinds` (shifted with labels if pad > 0)."""
        per = [batch // len(kinds) + (1 if i < batch % len(kinds) else 0) for i in range(len(kinds))]
        xs, ys = [], []
        for kind, m in zip(kinds, per):
            pool = self.pool(kind)
            x, y, _ = self.images(kind, pool[torch.randint(len(pool), (m,), generator=gen)])
            xs.append(x); ys.append(y)
        return shift_with_labels(torch.cat(xs), torch.cat(ys), pad, gen)

    def fixed(self, kind: str, n: int | None, seed: int = 0) -> torch.Tensor:
        """A fixed, seeded subset of n valid indices for `kind` (all if n is None or larger), sorted."""
        pool = self.pool(kind)
        if n is None or n >= len(pool):
            return pool
        perm = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))[:n]
        return pool[perm].sort().values


def summary_counts(data: MixData) -> dict[str, Any]:
    return {"frames": int(len(data.frame_idx)), "with_action": int(len(data.action_idx))}

"""GPU-resident samplers (CLAUDE.md §4 "Data loading", §6, §7).

The whole npz goes onto the device once as uint8; batches are gathered there by index.
No DataLoader. A window starting at frame s holds frames s..s+W-1 and actions s..s+W-2
(action j is the one applied at frame j), all inside one episode.

WindowData: windows for the world models and the pixel baseline.
GoalData:   (3 context frames, goal frame, pusher position, action target) for BC.
FrameData:  single frames + state targets for the probe.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wm.data.collect import seed_list
from wm.envs.pusht import block_cog


def split_episodes(seeds: np.ndarray, val_seeds: list[int]) -> tuple[np.ndarray, np.ndarray, str]:
    """Boolean masks (train, val) over episodes, and a description of the val source.

    Smoke data (seeds 0-1) has no val episodes: then every episode trains and the last one is
    also used for validation.
    """
    is_val = np.isin(seeds, seed_list(val_seeds))
    if is_val.any():
        return ~is_val, is_val, "val episodes (seeds {}-{})".format(*val_seeds)
    val_eps = np.zeros(len(seeds), bool)
    val_eps[-1] = True
    return np.ones(len(seeds), bool), val_eps, f"train episode seed {int(seeds[-1])} (smoke: no val episodes)"


def episode_frames(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """All global frame indices of the given episodes."""
    out = [np.arange(s, s + n) for s, n in zip(starts, lengths)]
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)


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
        train_eps, val_eps, self.val_source = split_episodes(seeds, val_seeds)
        self._val_eps = (starts[val_eps], lengths[val_eps])
        self.train_seeds = seeds[train_eps].tolist()
        self.val_seeds = seeds[val_eps].tolist()
        self.train_starts = torch.from_numpy(window_starts(starts[train_eps], lengths[train_eps], window))
        self.val_starts = torch.from_numpy(window_starts(starts[val_eps], lengths[val_eps], window))
        self.val_frame_idx = torch.from_numpy(episode_frames(starts[val_eps], lengths[val_eps]))
        self.frames = torch.from_numpy(arrays["frames"]).to(self.device)            # uint8
        acts = arrays["actions"].astype(np.float32) / 256.0 - 1.0                   # normalize_pos
        self.actions = torch.from_numpy(acts).to(self.device)
        self.offsets = torch.arange(window, device=self.device)

    def gather(self, starts: torch.Tensor, length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Window start indices [B] -> frames [B, L, 3, 64, 64] float, actions [B, L-1, 2].

        L = the sampler's window unless `length` is given (e.g. 13-frame val rollouts of the
        4-frame pixel sampler); the caller makes sure such windows fit in their episode.
        """
        offsets = self.offsets if length is None else torch.arange(length, device=self.device)
        idx = starts.to(self.device)[:, None] + offsets
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

    def fixed_val_starts_of(self, length: int, n: int, seed: int = 0) -> torch.Tensor:
        """Like fixed_val_starts, for windows of another length (e.g. 13-frame val rollouts)."""
        starts = torch.from_numpy(window_starts(*self._val_eps, length))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(starts), generator=g)[:n]
        return starts[perm].sort().values

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


def random_shift(images: torch.Tensor, pad: int, generator: torch.Generator) -> torch.Tensor:
    """Random ±pad px translation (replicate-pad, random crop), one shift per sample.

    images [B, N, 3, H, W]: all N images of a sample get the same shift (§6: context + goal).
    """
    b, n, c, h, w = images.shape
    padded = torch.nn.functional.pad(images.reshape(b, n * c, h, w), (pad,) * 4, mode="replicate")
    oy, ox = torch.randint(0, 2 * pad + 1, (2, b), generator=generator).to(images.device)
    rows = oy[:, None] + torch.arange(h, device=images.device)           # [B, H]
    cols = ox[:, None] + torch.arange(w, device=images.device)           # [B, W]
    bi = torch.arange(b, device=images.device)[:, None, None]
    out = padded[bi, :, rows[:, :, None], cols[:, None, :]]              # [B, H, W, N*C]
    return out.permute(0, 3, 1, 2).reshape(b, n, c, h, w)


class GoalData:
    """Goal-conditioned BC samples (CLAUDE.md §6), split by episode seed.

    A sample is a frame t that has an action (every frame but an episode's last):
      context = frames t-2, t-1, t (indices before the episode start repeat its first frame, as
      the planner does at episode start, §8); goal = frame min(t + k, episode end), k ~ U{k_min..k_max};
      pusher = normalised pusher position at t; target = (action_t - pusher_t) / delta_scale.
    """

    def __init__(self, arrays: dict[str, np.ndarray], device, val_seeds: list[int], *,
                 context: int = 3, k_min: int = 5, k_max: int = 30, delta_scale: float = 50.0,
                 use_clean_actions: bool = False):
        self.device, self.context, self.k_min, self.k_max = torch.device(device), context, k_min, k_max
        starts, lengths = arrays["episode_starts"], arrays["episode_lengths"]
        train_eps, val_eps, self.val_source = split_episodes(arrays["episode_seeds"], val_seeds)
        ep_first = np.repeat(starts, lengths)                    # per frame: its episode's first frame
        ep_last = np.repeat(starts + lengths - 1, lengths)       # ... and last frame
        has_action = np.arange(len(ep_last)) < ep_last

        def samples(eps: np.ndarray) -> torch.Tensor:
            idx = episode_frames(starts[eps], lengths[eps])
            return torch.from_numpy(idx[has_action[idx]])

        self.train_idx, self.val_idx = samples(train_eps), samples(val_eps)
        self.ep_first = torch.from_numpy(ep_first).to(self.device)
        self.ep_last = torch.from_numpy(ep_last).to(self.device)
        self.frames = torch.from_numpy(arrays["frames"]).to(self.device)
        acts = arrays["clean_actions" if use_clean_actions else "actions"].astype(np.float32)
        pusher = arrays["states"][:, :2].astype(np.float32)
        self.pusher = torch.from_numpy(pusher / 256.0 - 1.0).to(self.device)          # normalize_pos
        self.target = torch.from_numpy((acts - pusher) / delta_scale).to(self.device)
        self.ctx_offsets = torch.arange(-(context - 1), 1, device=self.device)

    def gather(self, t: torch.Tensor, k: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample frame indices t [B] and goal offsets k [B] -> batch dict (images float, unshifted)."""
        t, k = t.to(self.device), k.to(self.device)
        ctx_idx = torch.maximum(t[:, None] + self.ctx_offsets, self.ep_first[t][:, None])
        goal_idx = torch.minimum(t + k, self.ep_last[t])
        images = frames_to_model(self.frames[torch.cat([ctx_idx, goal_idx[:, None]], dim=1)])
        return {"images": images, "pusher": self.pusher[t], "target": self.target[t],
                "t": t, "goal_idx": goal_idx}

    def _draw(self, pool: torch.Tensor, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        t = pool[torch.randint(len(pool), (n,), generator=gen)]
        k = torch.randint(self.k_min, self.k_max + 1, (n,), generator=gen)
        return t, k

    def sample(self, batch: int, gen: torch.Generator, shift: int = 0) -> dict[str, torch.Tensor]:
        """Random train samples; `shift` > 0 applies random_shift to all 4 images of a sample."""
        out = self.gather(*self._draw(self.train_idx, batch, gen))
        if shift > 0:
            out["images"] = random_shift(out["images"], shift, gen)
        return out

    def fixed_val(self, n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """A fixed, seeded set of n val samples (t, k)."""
        return self._draw(self.val_idx, n, torch.Generator().manual_seed(seed))

    def mean_train_target(self) -> torch.Tensor:
        """Mean action target over all train samples (the constant-action baseline)."""
        return self.target[self.train_idx.to(self.device)].mean(0)

    def summary(self) -> dict[str, Any]:
        return {"train_samples": len(self.train_idx), "val_samples_total": len(self.val_idx),
                "val_source": self.val_source}


def state_targets(states: np.ndarray) -> np.ndarray:
    """Probe targets (CLAUDE.md §7) [..., 6]: pusher (x, y), block centre of gravity (x, y), both
    normalised x/256 - 1, then sin θ, cos θ."""
    s = np.asarray(states, np.float64)
    out = np.concatenate([s[..., :2] / 256.0 - 1.0, block_cog(s) / 256.0 - 1.0,
                          np.sin(s[..., 4:5]), np.cos(s[..., 4:5])], axis=-1)
    return out.astype(np.float32)


class FrameData:
    """Single frames with probe targets, split by episode seed (every frame of an episode)."""

    def __init__(self, arrays: dict[str, np.ndarray], device, val_seeds: list[int] | None):
        self.device = torch.device(device)
        starts, lengths = arrays["episode_starts"], arrays["episode_lengths"]
        if val_seeds is None:   # a test file: every frame is evaluation data
            train_eps = val_eps = np.ones(len(starts), bool)
            self.val_source = "all frames"
        else:
            train_eps, val_eps, self.val_source = split_episodes(arrays["episode_seeds"], val_seeds)
        self.train_idx = torch.from_numpy(episode_frames(starts[train_eps], lengths[train_eps]))
        self.val_idx = torch.from_numpy(episode_frames(starts[val_eps], lengths[val_eps]))
        self.frames = torch.from_numpy(arrays["frames"]).to(self.device)
        self.states = arrays["states"]
        self.targets = torch.from_numpy(state_targets(arrays["states"])).to(self.device)

    def gather(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Frame indices [B] -> frames [B, 3, 64, 64] float, targets [B, 6]."""
        idx = idx.to(self.device)
        return frames_to_model(self.frames[idx]), self.targets[idx]

    def sample(self, batch: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gather(self.train_idx[torch.randint(len(self.train_idx), (batch,), generator=gen)])

    def fixed_val(self, n: int, seed: int = 0) -> torch.Tensor:
        """A fixed, seeded subset of n val frame indices (all of them if fewer), sorted."""
        perm = torch.randperm(len(self.val_idx), generator=torch.Generator().manual_seed(seed))[:n]
        return self.val_idx[perm].sort().values

    def mean_train_target(self) -> torch.Tensor:
        return self.targets[self.train_idx.to(self.device)].mean(0)

"""Goal-frame weight mask for the Phase 6b pixel cost (pre-registered in PROGRESS.md).

w = 1 + alpha * (fg_mask AND NOT pusher_mask), computed from the goal image only:
  fg_mask     = the world model's training rule (any channel |frame - train mean| > threshold);
  pusher_mask = pixels whose colour is a blend of a pusher colour and a scene colour, with pusher
                share >= 0.25 and distance from that blend line <= 12 uint8 levels (blur from the
                512 -> 96 -> 64 px resizes). Pusher pixels keep weight 1.
Colours are gym-pusht's render colours (gym_pusht/envs/pusht.py: add_circle "RoyalBlue",
add_tee "LightSlateGray", goal "LightGreen", walls add_segment "LightGray", background white;
pymunk_override.py draw_circle / draw_polygon also paint light_color = min(1.2 * colour, 255)).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

PUSHER_COLOURS = np.array([[65, 105, 225],     # RoyalBlue (circle rim)
                           [78, 126, 255]],    # light_color(RoyalBlue) (circle inside)
                          np.float32)
SCENE_COLOURS = np.array([[255, 255, 255],     # background
                          [211, 211, 211],     # LightGray walls
                          [144, 238, 144],     # LightGreen goal region
                          [119, 136, 153],     # LightSlateGray block
                          [143, 163, 184]],    # light_color(LightSlateGray)
                         np.float32)
MIN_SHARE = 0.25     # pusher share of the blend
MAX_RESIDUAL = 12.0  # uint8 levels from the blend line


def pusher_mask(frame_u8: np.ndarray, min_share: float = MIN_SHARE, max_residual: float = MAX_RESIDUAL) -> np.ndarray:
    """uint8 [H, W, 3] -> bool [H, W]: pixels that look like the pusher (pure or blended)."""
    x = np.asarray(frame_u8, np.float32).reshape(-1, 3)
    out = np.zeros(len(x), bool)
    for p in PUSHER_COLOURS:
        for b in SCENE_COLOURS:
            v = p - b
            a = np.clip(((x - b) @ v) / float(v @ v), 0.0, 1.0)
            res = np.linalg.norm(x - (b + a[:, None] * v), axis=1)
            out |= (a >= min_share) & (res <= max_residual)
    return out.reshape(np.asarray(frame_u8).shape[:2])


def goal_weight(wm: torch.nn.Module, goal_u8: np.ndarray, device=None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Weight [1, 64, 64] = 1 + alpha * (fg AND NOT pusher) for a goal frame (uint8 [64, 64, 3]),
    with the world model's own fg_mean / alpha / threshold. Info: fg and removed pixel counts."""
    if getattr(wm, "fg_mean", None) is None:
        raise RuntimeError("cost_space 'pixel' needs a world model trained with fg_weight (fg_mean buffer)")
    device = device if device is not None else wm.fg_mean.device
    g = torch.as_tensor(np.asarray(goal_u8)).to(device).movedim(-1, -3).float() / 255.0 - 0.5
    fg = ((g - wm.fg_mean.to(device)).abs().amax(dim=0) > wm.fg_threshold)
    pm = torch.as_tensor(pusher_mask(goal_u8), device=device)
    keep = fg & ~pm
    w = 1.0 + wm.fg_alpha * keep.float()
    n_fg = int(fg.sum())
    return w[None], {"fg_pixels": n_fg, "removed": int((fg & pm).sum()),
                     "removed_share": float((fg & pm).sum()) / n_fg if n_fg else 0.0,
                     "fg": fg.cpu().numpy(), "pusher": pm.cpu().numpy()}

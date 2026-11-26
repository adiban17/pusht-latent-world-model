"""Pixel-space baseline (CLAUDE.md §5): a U-Net that predicts the next frame from 3 frames + action.

Input: frames t-2, t-1, t stacked (9 ch) + a_t tiled as 2 constant channels = 11 ch at 64x64.
Down convs (k4, s2, p1) 11->48->96->192->384 (64->4 px); bottleneck Conv 3x3 on
[384 + 2 tiled action channels] -> 384; up ConvTranspose (k4, s2, p1) with skip concatenation
384->192, (192+192)->96, (96+96)->48, (48+48)->3. SiLU everywhere except the last layer.
Output Δ; prediction = frame t + Δ (normalised pixel space). 4,813,731 params.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

DOWN = (48, 96, 192, 384)
EXPECTED_PARAMS = 4_813_731
WM_PARAMS = 4_184_899          # the world model's total (§4)
PARAM_TOLERANCE = 0.20         # Phase 1 decision: within ±20% of WM_PARAMS


def tile(a: torch.Tensor, size: int) -> torch.Tensor:
    """Action [B, A] -> constant planes [B, A, size, size]."""
    return a[:, :, None, None].expand(-1, -1, size, size)


class PixelUNet(nn.Module):
    def __init__(self, context: int = 3, action_dim: int = 2):
        super().__init__()
        self.context = context
        ins = (3 * context + action_dim,) + DOWN[:-1]
        self.downs = nn.ModuleList(nn.Conv2d(i, o, 4, 2, 1) for i, o in zip(ins, DOWN))
        self.mid = nn.Conv2d(DOWN[-1] + action_dim, DOWN[-1], 3, padding=1)
        # up path: 384->192, then each takes [up, skip] concatenated
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(DOWN[3], DOWN[2], 4, 2, 1),
            nn.ConvTranspose2d(2 * DOWN[2], DOWN[1], 4, 2, 1),
            nn.ConvTranspose2d(2 * DOWN[1], DOWN[0], 4, 2, 1),
            nn.ConvTranspose2d(2 * DOWN[0], 3, 4, 2, 1),
        ])
        self.act = nn.SiLU()

    def forward(self, frames: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """frames [B, C, 3, 64, 64] (oldest first), action [B, 2] -> next frame [B, 3, 64, 64]."""
        x = torch.cat([frames.flatten(1, 2), tile(action, frames.shape[-1])], dim=1)
        skips = []
        for down in self.downs:
            x = self.act(down(x))
            skips.append(x)                                   # 48@32, 96@16, 192@8, 384@4
        x = self.act(self.mid(torch.cat([x, tile(action, x.shape[-1])], dim=1)))
        x = self.act(self.ups[0](x))                          # 192@8
        for up, skip in zip(self.ups[1:], reversed(skips[:-1])):
            x = up(torch.cat([x, skip], dim=1))
            if up is not self.ups[-1]:
                x = self.act(x)
        return frames[:, -1] + x

    def rollout(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Autoregressive: frames [B, C, 3, 64, 64], actions [B, H, 2] -> predictions [B, H, 3, 64, 64].

        The C-frame window slides over the model's own predictions.
        """
        window, out = frames, []
        for k in range(actions.shape[1]):
            nxt = self.forward(window, actions[:, k])
            out.append(nxt)
            window = torch.cat([window[:, 1:], nxt[:, None]], dim=1)
        return torch.stack(out, dim=1)


def param_counts(model: PixelUNet) -> dict[str, int]:
    return {"total": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def within_tolerance(total: int) -> bool:
    return abs(total / WM_PARAMS - 1.0) <= PARAM_TOLERANCE


def build_pixel_unet(cfg: dict[str, Any]) -> PixelUNet:
    return PixelUNet(context=cfg["model"]["context"], action_dim=cfg["model"]["action_dim"])

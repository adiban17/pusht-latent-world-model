"""World model (CLAUDE.md §4): encoder + dynamics + decoder, the window loss and the planning API.

Window = 13 frames o[t-2..t+10] and 12 actions a[t-2..t+9] (action j is applied at frame j).
  z_j = E(o_j);  warm-up h over (z_{t-2}, a_{t-2}), (z_{t-1}, a_{t-1});
  rollout from z_t with a_t..a_{t+9} -> z_hat_{t+1..t+10}.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wm.models.decoder import Decoder
from wm.models.dynamics import Dynamics
from wm.models.encoder import Encoder

# Expected parameter counts (CLAUDE.md §4); asserted in tests and checked in the REPORT.
EXPECTED_PARAMS = {"encoder": 1_738_976, "decoder": 1_742_563, "dynamics": 703_360,
                   "total": 4_184_899, "planner": 2_442_336}


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 320, action_dim: int = 2,
                 context: int = 3, horizon: int = 10, teacher_forcing: bool = False,
                 input_mean_frame: bool = False, detach_dynamics_inputs: bool = False,
                 var_target: float = 1.0):
        """The last three options are collapse-fix variants under evaluation (Phase 3); all off
        by default, i.e. exactly the CLAUDE.md §4 model."""
        super().__init__()
        self.context, self.horizon, self.teacher_forcing = context, horizon, teacher_forcing
        self.detach_dynamics_inputs, self.var_target = detach_dynamics_inputs, var_target
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.dynamics = Dynamics(latent_dim, action_dim, hidden_dim)
        if input_mean_frame:  # a buffer, not a parameter: param counts are unchanged
            self.register_buffer("input_mean", torch.zeros(3, 64, 64))
        else:
            self.input_mean = None

    def set_input_mean(self, mean_frame: torch.Tensor) -> None:
        """Dataset mean frame [3, 64, 64] (normalised) subtracted from every encoder input."""
        if self.input_mean is None:
            raise RuntimeError("model was built without input_mean_frame")
        self.input_mean.copy_(mean_frame)

    # ---- planning API ------------------------------------------------------------------
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [..., 3, 64, 64] -> z [..., 256]."""
        if self.input_mean is not None:
            frames = frames - self.input_mean
        return self.encoder(frames)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [..., 256] -> frames [..., 3, 64, 64]."""
        return self.decoder(z)

    def warmup(self, z_ctx: torch.Tensor, a_ctx: torch.Tensor) -> torch.Tensor:
        """Hidden state from the context frames before t: z_ctx [B, C-1, 256], a_ctx [B, C-1, 2]."""
        h = self.dynamics.init_hidden(z_ctx.shape[0], z_ctx.device)
        for j in range(z_ctx.shape[1]):
            h = self.dynamics.observe(z_ctx[:, j], a_ctx[:, j], h)
        return h

    def rollout(self, z0: torch.Tensor, h: torch.Tensor, actions: torch.Tensor,
                teacher: torch.Tensor | None = None) -> torch.Tensor:
        """Imagine H steps from z0 [B, 256] with actions [B, H, 2] -> z_hat [B, H, 256].

        `teacher` [B, H, 256] (real z_t .. z_{t+H-1}) replaces the fed-back prediction at every
        step (teacher forcing, used by wm_1step).
        """
        z, out = z0, []
        for k in range(actions.shape[1]):
            z_in = teacher[:, k] if teacher is not None else z
            z, h = self.dynamics.step(z_in, actions[:, k], h)
            out.append(z)
        return torch.stack(out, dim=1)

    # ---- training ----------------------------------------------------------------------
    def window_loss(self, frames: torch.Tensor, actions: torch.Tensor,
                    weights: dict[str, float]) -> dict[str, torch.Tensor]:
        """frames [B, C+H, 3, 64, 64], actions [B, C+H-1, 2] -> loss terms (element means).

        Returns loss, rec, lat, img, var, z_std (all scalars) plus z [B, C+H, 256] and
        z_hat [B, H, 256] for metrics. `var` (VICReg-style hinge, mean_d relu(target - std_d)) is
        always reported but only added to the loss with a weights["var"] > 0.
        """
        c = self.context
        z = self.encode(frames)
        z_in = z.detach() if self.detach_dynamics_inputs else z  # variant (c)
        h = self.warmup(z_in[:, :c - 1], actions[:, :c - 1])
        teacher = z_in[:, c - 1:-1] if self.teacher_forcing else None
        z_hat = self.rollout(z_in[:, c - 1], h, actions[:, c - 1:], teacher=teacher)

        decoded = self.decode(torch.cat([z, z_hat], dim=1)).float()
        n = z.shape[1]
        rec = F.mse_loss(decoded[:, :n], frames)
        img = F.mse_loss(decoded[:, n:], frames[:, c:])
        lat = F.mse_loss(z_hat.float(), z[:, c:].detach().float())
        flat = z.float().reshape(-1, z.shape[-1])
        var = F.relu(self.var_target - torch.sqrt(flat.var(dim=0) + 1e-4)).mean()
        loss = weights["rec"] * rec + weights["lat"] * lat + weights["img"] * img
        if weights.get("var", 0.0) > 0:
            loss = loss + weights["var"] * var
        with torch.no_grad():
            z_std = flat.std(dim=0).mean()
        return {"loss": loss, "rec": rec, "lat": lat, "img": img, "var": var, "z_std": z_std,
                "z": z, "z_hat": z_hat}


def param_counts(model: WorldModel) -> dict[str, int]:
    """Trainable parameter counts per part, total and planner path (encoder + dynamics)."""
    def n(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    counts = {"encoder": n(model.encoder), "decoder": n(model.decoder),
              "dynamics": n(model.dynamics), "total": n(model)}
    counts["planner"] = counts["encoder"] + counts["dynamics"]
    return counts


def build_world_model(cfg: dict[str, Any]) -> WorldModel:
    """WorldModel from a wm.yaml-style config."""
    m, w = cfg["model"], cfg["window"]
    return WorldModel(latent_dim=m["latent_dim"], hidden_dim=m["hidden_dim"],
                      action_dim=m["action_dim"], context=w["context"], horizon=w["horizon"],
                      teacher_forcing=bool(cfg.get("teacher_forcing", False)),
                      input_mean_frame=bool(cfg.get("input_mean_frame", False)),
                      detach_dynamics_inputs=bool(cfg.get("detach_dynamics_inputs", False)),
                      var_target=float(cfg.get("var_target", 1.0)))


def load_world_model(path: str | Path, device="cpu") -> tuple[WorldModel, dict[str, Any]]:
    """Load model.pt written by train_wm; returns (model in eval mode, saved info)."""
    blob = torch.load(path, map_location=device, weights_only=True)
    model = build_world_model(blob["config"]).to(device)
    model.load_state_dict(blob["model"])
    return model.eval(), blob

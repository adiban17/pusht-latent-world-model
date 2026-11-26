"""World model (CLAUDE.md §4): encoder + dynamics + decoder, the window loss and the planning API.

Window = 13 frames o[t-2..t+10] and 12 actions a[t-2..t+9] (action j is applied at frame j).
  z_j = E(o_j);  warm-up h over (z_{t-2}, a_{t-2}), (z_{t-1}, a_{t-1});
  rollout from z_t with a_t..a_{t+9} -> z_hat_{t+1..t+10}.

Phase 3b options (config flags, all off by default = the §4 model):
  fg_weight: per-pixel weight w = 1 + alpha * mask in the rec and img losses, where mask marks
             pixels whose |frame - train mean frame| > threshold in any channel;
  proprio:   the pusher position p (normalised) is part of the model state; the dynamics see
             [z, p, a] and also predict p (MSE term `proprio`); rollouts feed back p_hat.
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
# proprio (Phase 3b B): phi gets 2 more inputs (+512) and the p head Linear(320 -> 2) (+642).
EXPECTED_PARAMS_PROPRIO = {"encoder": 1_738_976, "decoder": 1_742_563, "dynamics": 704_514,
                           "total": 4_186_053, "planner": 2_443_490}


def expected_params(model: "WorldModel") -> dict[str, int]:
    return EXPECTED_PARAMS_PROPRIO if model.proprio else EXPECTED_PARAMS


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 320, action_dim: int = 2,
                 context: int = 3, horizon: int = 10, teacher_forcing: bool = False,
                 input_mean_frame: bool = False, detach_dynamics_inputs: bool = False,
                 var_target: float = 1.0, fg_weight: bool = False, fg_alpha: float = 20.0,
                 fg_threshold: float = 25.0 / 255.0, proprio: bool = False):
        """input_mean_frame / detach_dynamics_inputs are Phase 3 collapse-fix variants; fg_weight
        and proprio are Phase 3b options. All off by default, i.e. exactly the §4 model."""
        super().__init__()
        self.context, self.horizon, self.teacher_forcing = context, horizon, teacher_forcing
        self.detach_dynamics_inputs, self.var_target = detach_dynamics_inputs, var_target
        self.fg_alpha, self.fg_threshold, self.proprio = fg_alpha, fg_threshold, proprio
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.dynamics = Dynamics(latent_dim, action_dim, hidden_dim, proprio=proprio)
        if input_mean_frame:  # a buffer, not a parameter: param counts are unchanged
            self.register_buffer("input_mean", torch.zeros(3, 64, 64))
        else:
            self.input_mean = None
        if fg_weight:         # train mean frame for the foreground mask (buffer, saved in model.pt)
            self.register_buffer("fg_mean", torch.zeros(3, 64, 64))
        else:
            self.fg_mean = None

    def set_input_mean(self, mean_frame: torch.Tensor) -> None:
        """Dataset mean frame [3, 64, 64] (normalised) subtracted from every encoder input."""
        if self.input_mean is None:
            raise RuntimeError("model was built without input_mean_frame")
        self.input_mean.copy_(mean_frame)

    def set_fg_mean(self, mean_frame: torch.Tensor) -> None:
        """Train mean frame [3, 64, 64] (normalised) that defines the foreground mask."""
        if self.fg_mean is None:
            raise RuntimeError("model was built without fg_weight")
        self.fg_mean.copy_(mean_frame)

    # ---- planning API ------------------------------------------------------------------
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [..., 3, 64, 64] -> z [..., 256]."""
        if self.input_mean is not None:
            frames = frames - self.input_mean
        return self.encoder(frames)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [..., 256] -> frames [..., 3, 64, 64]."""
        return self.decoder(z)

    def warmup(self, z_ctx: torch.Tensor, a_ctx: torch.Tensor,
               p_ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Hidden state from the context frames before t: z_ctx [B, C-1, 256], a_ctx [B, C-1, 2]
        (+ the real pusher positions p_ctx [B, C-1, 2] for a proprio model)."""
        self._need_p(p_ctx)
        h = self.dynamics.init_hidden(z_ctx.shape[0], z_ctx.device)
        for j in range(z_ctx.shape[1]):
            p = p_ctx[:, j] if p_ctx is not None else None
            h = self.dynamics.observe(z_ctx[:, j], a_ctx[:, j], h, p)
        return h

    def rollout(self, z0: torch.Tensor, h: torch.Tensor, actions: torch.Tensor,
                teacher: torch.Tensor | None = None, p0: torch.Tensor | None = None,
                teacher_p: torch.Tensor | None = None):
        """Imagine H steps from z0 [B, 256] with actions [B, H, 2] -> z_hat [B, H, 256].

        `teacher` [B, H, 256] (real z_t .. z_{t+H-1}) replaces the fed-back prediction at every
        step (teacher forcing, used by wm_1step).
        A proprio model needs p0 [B, 2] (real pusher at t) and returns (z_hat, p_hat [B, H, 2]),
        feeding back p_hat (or the real teacher_p [B, H, 2] under teacher forcing).
        """
        self._need_p(p0)
        if not self.proprio:
            z, out = z0, []
            for k in range(actions.shape[1]):
                z_in = teacher[:, k] if teacher is not None else z
                z, h = self.dynamics.step(z_in, actions[:, k], h)
                out.append(z)
            return torch.stack(out, dim=1)
        z, p, zs, ps = z0, p0, [], []
        for k in range(actions.shape[1]):
            z_in = teacher[:, k] if teacher is not None else z
            p_in = teacher_p[:, k] if teacher_p is not None else p
            z, p, h = self.dynamics.step_proprio(z_in, p_in, actions[:, k], h)
            zs.append(z)
            ps.append(p)
        return torch.stack(zs, dim=1), torch.stack(ps, dim=1)

    def _need_p(self, p: torch.Tensor | None) -> None:
        if self.proprio and p is None:
            raise ValueError("proprio world model: pass the pusher positions (p_ctx / p0 / pusher)")

    # ---- training ----------------------------------------------------------------------
    def fg_weights(self, frames: torch.Tensor) -> torch.Tensor | None:
        """Per-pixel loss weights 1 + alpha * mask [..., 1, 64, 64] (None if fg_weight is off)."""
        if self.fg_mean is None:
            return None
        mask = (frames - self.fg_mean).abs().amax(dim=-3, keepdim=True) > self.fg_threshold
        return 1.0 + self.fg_alpha * mask.float()

    def window_loss(self, frames: torch.Tensor, actions: torch.Tensor,
                    weights: dict[str, float], pusher: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """frames [B, C+H, 3, 64, 64], actions [B, C+H-1, 2] -> loss terms (element means).

        Returns loss, rec, lat, img, var, z_std (all scalars) plus z [B, C+H, 256] and
        z_hat [B, H, 256] for metrics. `var` (VICReg-style hinge, mean_d relu(target - std_d)) is
        always reported but only added to the loss with a weights["var"] > 0.
        A proprio model needs pusher [B, C+H, 2] (normalised) and adds `proprio` (MSE of p_hat vs
        the real p_{t+1..t+H}, weight weights.get("proprio", 1.0)) and `p_hat` [B, H, 2].
        """
        self._need_p(pusher)
        c = self.context
        z = self.encode(frames)
        z_in = z.detach() if self.detach_dynamics_inputs else z  # variant (c)
        p_ctx = pusher[:, :c - 1] if self.proprio else None
        h = self.warmup(z_in[:, :c - 1], actions[:, :c - 1], p_ctx)
        teacher = z_in[:, c - 1:-1] if self.teacher_forcing else None
        if self.proprio:
            teacher_p = pusher[:, c - 1:-1] if self.teacher_forcing else None
            z_hat, p_hat = self.rollout(z_in[:, c - 1], h, actions[:, c - 1:], teacher=teacher,
                                        p0=pusher[:, c - 1], teacher_p=teacher_p)
        else:
            z_hat = self.rollout(z_in[:, c - 1], h, actions[:, c - 1:], teacher=teacher)

        decoded = self.decode(torch.cat([z, z_hat], dim=1)).float()
        n = z.shape[1]
        w = self.fg_weights(frames)
        if w is None:
            rec = F.mse_loss(decoded[:, :n], frames)
            img = F.mse_loss(decoded[:, n:], frames[:, c:])
        else:
            rec = (w * (decoded[:, :n] - frames) ** 2).mean()
            img = (w[:, c:] * (decoded[:, n:] - frames[:, c:]) ** 2).mean()
        lat = F.mse_loss(z_hat.float(), z[:, c:].detach().float())
        flat = z.float().reshape(-1, z.shape[-1])
        var = F.relu(self.var_target - torch.sqrt(flat.var(dim=0) + 1e-4)).mean()
        loss = weights["rec"] * rec + weights["lat"] * lat + weights["img"] * img
        if weights.get("var", 0.0) > 0:
            loss = loss + weights["var"] * var
        with torch.no_grad():
            z_std = flat.std(dim=0).mean()
        out = {"loss": loss, "rec": rec, "lat": lat, "img": img, "var": var, "z_std": z_std,
               "z": z, "z_hat": z_hat}
        if self.proprio:
            prop = F.mse_loss(p_hat.float(), pusher[:, c:].float())
            out["loss"] = loss + weights.get("proprio", 1.0) * prop
            out["proprio"], out["p_hat"] = prop, p_hat
        return out


def param_counts(model: WorldModel) -> dict[str, int]:
    """Trainable parameter counts per part, total and planner path (encoder + dynamics)."""
    def n(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    counts = {"encoder": n(model.encoder), "decoder": n(model.decoder),
              "dynamics": n(model.dynamics), "total": n(model)}
    counts["planner"] = counts["encoder"] + counts["dynamics"]
    return counts


def build_world_model(cfg: dict[str, Any]) -> WorldModel:
    """WorldModel from a wm.yaml-style config (Phase 3b keys are optional, default off)."""
    m, w = cfg["model"], cfg["window"]
    return WorldModel(latent_dim=m["latent_dim"], hidden_dim=m["hidden_dim"],
                      action_dim=m["action_dim"], context=w["context"], horizon=w["horizon"],
                      teacher_forcing=bool(cfg.get("teacher_forcing", False)),
                      input_mean_frame=bool(cfg.get("input_mean_frame", False)),
                      detach_dynamics_inputs=bool(cfg.get("detach_dynamics_inputs", False)),
                      var_target=float(cfg.get("var_target", 1.0)),
                      fg_weight=bool(cfg.get("fg_weight", False)),
                      fg_alpha=float(cfg.get("fg_alpha", 20.0)),
                      fg_threshold=float(cfg.get("fg_threshold", 25.0 / 255.0)),
                      proprio=bool(cfg.get("proprio", False)))


def load_world_model(path: str | Path, device="cpu") -> tuple[WorldModel, dict[str, Any]]:
    """Load model.pt written by train_wm; returns (model in eval mode, saved info)."""
    blob = torch.load(path, map_location=device, weights_only=True)
    model = build_world_model(blob["config"]).to(device)
    model.load_state_dict(blob["model"])
    return model.eval(), blob

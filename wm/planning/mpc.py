"""Closed-loop MPC runner and policies (CLAUDE.md §8).

Every policy has `reset(goal_frame, gen)` and `act(history) -> (target px [2], info)`.
Context each step = the last 3 real frames and the 2 actions between them (plus, for a proprio
world model, the real pusher positions at those frames). At episode start the first frame is
repeated and the pusher's current position stands in for the missing actions and positions.

  CEMPolicy(wm)          CEM over waypoints with the world model; state (z, p) for a proprio model;
                         cost on the image latent only (the goal is an image).
  PixelCEMPolicy(u, wm)  the same CEM with the pixel U-Net as dynamics; predicted frames are clipped
                         and encoded by E_wm, then scored with the same cost (Phase 6, flag).
  BCPolicy(net)          goal-conditioned BC; target = clip(p + 50 * output, 0, 512).
  RandomPolicy           δ ~ N(0, 30²) through the waypoint transform (H = 1).
  NoopPolicy             target = current pusher position.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from wm.data.windows import frames_to_model
from wm.envs.pusht import SUCCESS_POS_PX, get_state, pose_error, render64, reset_to_state
from wm.models.bc import delta_to_target
from wm.planning.cem import (CEMConfig, cem, deltas_to_waypoints, latent_cost, shift_warm_start,
                             waypoints_to_actions)

CONTEXT = 3


class History:
    """Real frames (uint8 [64, 64, 3]), pusher positions and executed actions (px) of an episode."""

    def __init__(self, frame0: np.ndarray, pusher0: np.ndarray):
        self.frames = [frame0]
        self.pushers = [np.asarray(pusher0, np.float32)]
        self.actions: list[np.ndarray] = []

    def push(self, action: np.ndarray, frame: np.ndarray, pusher: np.ndarray) -> None:
        self.actions.append(np.asarray(action, np.float32))
        self.frames.append(frame)
        self.pushers.append(np.asarray(pusher, np.float32))

    @property
    def pusher(self) -> np.ndarray:
        return self.pushers[-1]

    def context(self, c: int = CONTEXT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(frames [c, 64, 64, 3] uint8, actions [c-1, 2] px, pushers [c-1, 2] px) before and at
        the current step t: frames t-c+1..t, actions and pushers t-c+1..t-1. Indices before the
        start repeat the first frame and use the *current* pusher position (§8)."""
        t = len(self.frames) - 1
        frames = np.stack([self.frames[max(j, 0)] for j in range(t - c + 1, t + 1)])
        acts = np.stack([self.actions[j] if j >= 0 else self.pusher for j in range(t - c + 1, t)])
        push = np.stack([self.pushers[j] if j >= 0 else self.pusher for j in range(t - c + 1, t)])
        return frames, acts, push


def _norm(x: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, np.float32) / 256.0 - 1.0, device=device)


class Policy:
    name = "policy"

    def reset(self, goal_frame: np.ndarray, gen: torch.Generator) -> None:
        self.gen = gen

    def act(self, hist: History) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


class NoopPolicy(Policy):
    name = "noop"

    def act(self, hist: History) -> tuple[np.ndarray, dict[str, Any]]:
        return hist.pusher.copy(), {}


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, sigma: float = 30.0, delta_clip: float = 50.0):
        self.sigma, self.delta_clip = sigma, delta_clip

    def act(self, hist: History) -> tuple[np.ndarray, dict[str, Any]]:
        d = (self.sigma * torch.randn((1, 1, 2), generator=self.gen)).clamp(-self.delta_clip, self.delta_clip)
        w = deltas_to_waypoints(torch.as_tensor(hist.pusher), d)
        return w[0, 0].numpy(), {}


class BCPolicy(Policy):
    def __init__(self, net: torch.nn.Module, device, name: str = "bc", delta_scale: float = 50.0):
        self.net, self.device, self.name, self.scale = net.eval(), device, name, delta_scale

    def reset(self, goal_frame: np.ndarray, gen: torch.Generator) -> None:
        super().reset(goal_frame, gen)
        self.goal = torch.as_tensor(goal_frame)

    @torch.no_grad()
    def act(self, hist: History) -> tuple[np.ndarray, dict[str, Any]]:
        frames, _, _ = hist.context(self.net.context)
        imgs = frames_to_model(torch.cat([torch.as_tensor(frames), self.goal[None]]))[None].to(self.device)
        out = self.net(imgs, _norm(hist.pusher, self.device)[None])[0].float().cpu().numpy()
        return delta_to_target(hist.pusher, out, self.scale).astype(np.float32), {}


class _CEMBase(Policy):
    """Shared CEM plumbing: warm start, waypoint transform, best plan bookkeeping."""

    def __init__(self, wm: torch.nn.Module, cfg: CEMConfig, device, name: str):
        self.wm, self.cfg, self.device, self.name = wm.eval(), cfg, device, name

    def reset(self, goal_frame: np.ndarray, gen: torch.Generator) -> None:
        super().reset(goal_frame, gen)
        with torch.no_grad():
            g = frames_to_model(torch.as_tensor(goal_frame)[None]).to(self.device)
            self.z_goal = self.wm.encode(g)[0].float()
        self.prev: torch.Tensor | None = None

    def prepare(self, hist: History) -> None:
        """Encode the current context once per step (before the CEM rounds)."""
        raise NotImplementedError

    def imagine(self, actions: torch.Tensor) -> torch.Tensor:
        """Model actions [N, H, 2] -> imagined latents [N, H, D] (in E_wm's space)."""
        raise NotImplementedError

    @torch.no_grad()
    def act(self, hist: History) -> tuple[np.ndarray, dict[str, Any]]:
        p = torch.as_tensor(hist.pusher)
        self.prepare(hist)

        def score(deltas: torch.Tensor) -> torch.Tensor:
            acts = waypoints_to_actions(deltas_to_waypoints(p, deltas)).to(self.device)
            return latent_cost(self.imagine(acts), self.z_goal, self.cfg.beta)

        mu0 = shift_warm_start(self.prev, self.cfg.horizon) if self.cfg.warm_start else torch.zeros(self.cfg.horizon, 2)
        res = cem(score, mu0, self.cfg, self.gen)
        self.prev = res["best"]
        # the best plan's imagined latents (one extra rollout of a single candidate)
        acts = waypoints_to_actions(deltas_to_waypoints(p, res["best"][None])).to(self.device)
        z_best = self.imagine(acts)[0].float()
        terminal = float(((z_best[-1] - self.z_goal) ** 2).sum())
        target = deltas_to_waypoints(p, res["best"][None])[0, 0].numpy()
        return target.astype(np.float32), {"cost": res["best_cost"], "terminal_cost": terminal,
                                           "plan_z": z_best.cpu()}


class CEMPolicy(_CEMBase):
    """CEM with a world model; a proprio model's planner state is (z, p)."""

    def __init__(self, wm: torch.nn.Module, cfg: CEMConfig, device, name: str = "cem_wm"):
        super().__init__(wm, cfg, device, name)

    def prepare(self, hist: History) -> None:
        frames, acts, push = hist.context(self.wm.context)
        z = self.wm.encode(frames_to_model(torch.as_tensor(frames)).to(self.device))   # [C, D]
        a_ctx = _norm(acts, self.device)[None]
        p_ctx = _norm(push, self.device)[None] if self.wm.proprio else None
        self.h = self.wm.warmup(z[None, :-1], a_ctx, p_ctx)
        self.z0 = z[-1]
        self.p0 = _norm(hist.pusher, self.device)

    def imagine(self, actions: torch.Tensor) -> torch.Tensor:
        n = actions.shape[0]
        z0, h = self.z0[None].expand(n, -1), self.h.expand(n, -1).contiguous()
        if self.wm.proprio:
            z_hat, _ = self.wm.rollout(z0, h, actions, p0=self.p0[None].expand(n, -1))
            return z_hat
        return self.wm.rollout(z0, h, actions)


class PixelCEMPolicy(_CEMBase):
    """CEM with the pixel U-Net as dynamics; frames clipped to [-0.5, 0.5], scored in E_wm's space."""

    def __init__(self, unet: torch.nn.Module, wm: torch.nn.Module, cfg: CEMConfig, device,
                 amp: bool = True, name: str = "cem_pixel"):
        super().__init__(wm, cfg, device, name)
        self.unet = unet.eval()
        self.amp = amp and torch.device(device).type == "cuda"

    def prepare(self, hist: History) -> None:
        frames, _, _ = hist.context(self.unet.context)
        self.ctx_frames = frames_to_model(torch.as_tensor(frames)).to(self.device)     # [C, 3, 64, 64]

    def imagine(self, actions: torch.Tensor) -> torch.Tensor:
        n, h = actions.shape[:2]
        amp = torch.autocast("cuda", dtype=torch.float16) if self.amp else nullcontext()
        with amp:
            pred = self.unet.rollout(self.ctx_frames[None].expand(n, -1, -1, -1, -1), actions)
            z = self.wm.encode(pred.float().clamp(-0.5, 0.5).flatten(0, 1))
        return z.float().view(n, h, -1)


def run_episode(env, start_state: np.ndarray, goal_frame: np.ndarray, goal_state: np.ndarray,
                policy: Policy, budget: int = 40, gen: torch.Generator | None = None,
                keep_frames: bool = False, keep_plans: bool = False) -> dict[str, Any]:
    """reset_to_state(start); loop act -> step -> check is_success; stop at the first success.

    Returns success, pos_success (block CoG < 20 px after any step), steps, start / final / min
    errors, per-step best imagined cost and terminal cost (CEM), policy and env seconds, and
    optionally the real frames and the imagined plan latents.
    """
    from wm.envs.pusht import is_success

    gen = gen if gen is not None else torch.Generator().manual_seed(0)
    goal_state = np.asarray(goal_state, np.float32)
    obs = reset_to_state(env, np.asarray(start_state, np.float32))
    s = get_state(env)
    hist = History(render64(obs["pixels"]), s[:2])
    start_pos, start_ang = pose_error(s, goal_state)
    policy.reset(goal_frame, gen)
    costs, terminal, plans = [], [], []
    success, pos_success, min_pos, steps = False, False, start_pos, 0
    pos, ang = start_pos, start_ang
    t_policy = t_env = 0.0
    for step in range(1, budget + 1):
        t0 = time.perf_counter()
        target, info = policy.act(hist)
        t1 = time.perf_counter()
        obs, *_ = env.step(np.asarray(target, np.float32))
        s = get_state(env)
        hist.push(target, render64(obs["pixels"]), s[:2])
        t_env += time.perf_counter() - t1
        t_policy += t1 - t0
        steps = step
        if "cost" in info:
            costs.append(info["cost"])
            terminal.append(info["terminal_cost"])
            if keep_plans:
                plans.append(info["plan_z"])
        pos, ang = pose_error(s, goal_state)
        min_pos = min(min_pos, pos)
        pos_success = pos_success or pos < SUCCESS_POS_PX
        if is_success(s, goal_state):
            success = True
            break
    out = {"success": success, "pos_success": bool(pos_success), "steps": steps,
           "start_pos_px": float(start_pos), "start_ang_deg": float(start_ang),
           "final_pos_px": float(pos), "final_ang_deg": float(ang), "min_pos_px": float(min_pos),
           "costs": costs, "terminal_costs": terminal,
           "policy_s": t_policy, "env_s": t_env, "seconds": t_policy + t_env}
    if keep_frames:
        out["frames"] = np.stack(hist.frames)
    if keep_plans:
        out["plans"] = plans
    return out

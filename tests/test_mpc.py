"""MPC runner and policies (CLAUDE.md §8): context padding, proprio planner state, episodes end to end."""
import numpy as np
import pytest
import torch

from wm.envs.pusht import get_state, make_env, render64
from wm.models.bc import BCPolicyNet
from wm.models.pixel_unet import PixelUNet
from wm.models.world_model import WorldModel
from wm.planning.cem import CEMConfig
from wm.planning.mpc import (BCPolicy, CEMPolicy, History, NoopPolicy, PixelCEMPolicy, RandomPolicy,
                             run_episode)

SMALL = CEMConfig(candidates=8, horizon=3, rounds=2, elites=2)


@pytest.fixture(scope="module")
def env_state():
    env = make_env()
    obs, _ = env.reset(seed=0)
    return env, get_state(env), render64(obs["pixels"])


def frame(v):
    return np.full((64, 64, 3), v, np.uint8)


def test_history_padding_and_context():
    h = History(frame(1), np.array([100.0, 200.0]))
    f, a, p = h.context(3)
    assert [int(x[0, 0, 0]) for x in f] == [1, 1, 1]
    assert np.allclose(a, [[100, 200], [100, 200]]) and np.allclose(p, [[100, 200], [100, 200]])
    h.push(np.array([110.0, 210.0]), frame(2), np.array([105.0, 205.0]))
    f, a, p = h.context(3)
    assert [int(x[0, 0, 0]) for x in f] == [1, 1, 2]
    assert np.allclose(a, [[105, 205], [110, 210]])          # padding uses the *current* pusher
    assert np.allclose(p, [[105, 205], [100, 200]])
    h.push(np.array([120.0, 220.0]), frame(3), np.array([115.0, 215.0]))
    f, a, p = h.context(3)
    assert [int(x[0, 0, 0]) for x in f] == [1, 2, 3]
    assert np.allclose(a, [[110, 210], [120, 220]]) and np.allclose(p, [[100, 200], [105, 205]])


def test_noop_reaches_its_own_start(env_state):
    env, s, f = env_state
    ep = run_episode(env, s, f, s, NoopPolicy(), budget=5)
    assert ep["success"] and ep["steps"] == 1 and ep["pos_success"]


def test_cem_proprio_episode_runs(env_state):
    env, s, f = env_state
    torch.manual_seed(0)
    wm = WorldModel(proprio=True).eval()
    goal = s.copy()
    goal[2] += 60.0
    pol = CEMPolicy(wm, SMALL, "cpu")
    ep = run_episode(env, s, f, goal, pol, budget=3, keep_frames=True, keep_plans=True)
    assert ep["steps"] == 3 and len(ep["costs"]) == 3 and len(ep["terminal_costs"]) == 3
    assert ep["frames"].shape == (4, 64, 64, 3)
    assert ep["plans"][0].shape == (3, 256)
    assert all(np.isfinite(ep["costs"]))


def test_cem_non_proprio_and_pixel_policies(env_state):
    env, s, f = env_state
    torch.manual_seed(0)
    wm = WorldModel().eval()
    h = History(f, s[:2])
    for pol in (CEMPolicy(wm, SMALL, "cpu"), PixelCEMPolicy(PixelUNet().eval(), wm, SMALL, "cpu")):
        pol.reset(f, torch.Generator().manual_seed(0))
        target, info = pol.act(h)
        assert target.shape == (2,) and (0 <= target).all() and (target <= 512).all()
        assert np.isfinite(info["cost"])


def test_bc_and_random_targets_in_range(env_state):
    env, s, f = env_state
    h = History(f, s[:2])
    gen = torch.Generator().manual_seed(0)
    bc = BCPolicy(BCPolicyNet(), "cpu")
    rnd = RandomPolicy(30.0)
    for pol in (bc, rnd):
        pol.reset(f, gen)
        t, _ = pol.act(h)
        assert t.shape == (2,) and (0 <= t).all() and (t <= 512).all()
        assert np.linalg.norm(t - s[:2]) <= 50 * np.sqrt(2) + 1e-3 or pol is bc

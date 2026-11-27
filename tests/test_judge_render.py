"""Phase 5b rendered judge data: valid, non-overlapping, deterministic micro-episodes."""
import numpy as np
import pytest

from wm.data.judge_render import CLEARANCE, micro_episode, pusher_clearance, render_split, sample_valid_state
from wm.envs.pusht import block_cog, make_env, reset_to_state
from wm.eval.angle_audit import pose_state

CFG = {"steps": 4, "cog_low": 85.0, "cog_high": 427.0, "action_sigma": 30.0}


@pytest.fixture(scope="module")
def env():
    e = make_env()
    e.reset(seed=0)
    return e


def test_clearance_detects_overlap(env):
    reset_to_state(env, pose_state((256.0, 256.0), 0.0, (256.0, 256.0)))
    assert pusher_clearance(env) < 0
    reset_to_state(env, pose_state((256.0, 256.0), 0.0, (40.0, 40.0)))
    assert pusher_clearance(env) > 100


def test_valid_states_are_clear_and_in_range(env):
    rng = np.random.default_rng(0)
    for _ in range(30):
        s, _, _ = sample_valid_state(env, rng, 85.0, 427.0)
        reset_to_state(env, s)
        assert pusher_clearance(env) >= CLEARANCE - 1e-6
        assert np.all((block_cog(s) >= 85.0 - 1e-3) & (block_cog(s) <= 427.0 + 1e-3))


def test_micro_episode_layout_and_determinism(env):
    a, b = micro_episode(env, 5000, CFG), micro_episode(env, 5000, CFG)
    assert a["frames"].shape == (4, 64, 64, 3) and a["states"].shape == (4, 5)
    assert np.array_equal(a["frames"], b["frames"]) and np.array_equal(a["states"], b["states"])
    assert np.all(a["actions"][-1] == 0) and np.all(a["actions"][:3] >= 0) and np.all(a["actions"][:3] <= 512)


def test_render_split_npz_layout(env):
    arrays, stats = render_split(env, [5000, 5001, 5002], CFG)
    assert stats["frames"] == 12 and arrays["frames"].shape[0] == 12
    assert arrays["episode_starts"].tolist() == [0, 4, 8] and arrays["episode_lengths"].tolist() == [4, 4, 4]
    assert arrays["episode_seeds"].tolist() == [5000, 5001, 5002]
    assert np.array_equal(arrays["clean_actions"], arrays["actions"])

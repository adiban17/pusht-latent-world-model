"""Phase 7 search oracle: template_match recovers angle and shift; template_angle unchanged."""
import numpy as np
import pytest

from wm.envs.pusht import make_env
from wm.eval.angle_audit import (angle_err_deg, pose_state, render_state, template_angle, template_candidates,
                                 template_match)


@pytest.fixture(scope="module")
def env():
    e = make_env()
    e.reset(seed=0)
    return e


def _old_template_angle(env, frame, state, step_deg=5.0):
    """The Phase 5b implementation, kept here to show the refactor changes nothing."""
    from wm.envs.pusht import block_cog
    cog = block_cog(state)
    target = np.asarray(frame, np.float32)
    best, best_err = 0.0, np.inf
    for a in np.radians(np.arange(-180.0, 180.0, step_deg)):
        cand = render_state(env, pose_state(tuple(cog), a, (state[0], state[1]))).astype(np.float32)
        err = float(((cand - target) ** 2).mean())
        if err < best_err:
            best, best_err = float(a), err
    return best


@pytest.mark.parametrize("deg", [-150.0, 12.0, 95.0])
def test_template_angle_unchanged(env, deg):
    s = pose_state((250.0, 270.0), np.radians(deg), (120.0, 110.0))
    f = render_state(env, s)
    noisy = np.clip(f.astype(np.float32) + np.random.default_rng(0).normal(0, 20, f.shape), 0, 255)
    for frame in (f, noisy):
        assert template_angle(env, frame, s) == _old_template_angle(env, frame, s)


def test_search_recovers_angle_and_shift(env):
    s = pose_state((256.0, 240.0), np.radians(40.0), (110.0, 120.0))
    angles, cands = template_candidates(env, s)
    frame = render_state(env, s)
    shifted = np.roll(frame, (2, -3), axis=(0, 1))
    m = template_match(cands, np.stack([frame, shifted]).astype(np.float32), search_px=4)
    assert m["dy"].tolist() == [0, 2] and m["dx"].tolist() == [0, -3]
    assert (angle_err_deg(angles[m["index"]], s[4]) < 3.0).all()
    assert not m["edge"].any()


def test_shift_zero_equals_true_position_oracle(env):
    s = pose_state((300.0, 220.0), np.radians(-70.0), (100.0, 400.0))
    angles, cands = template_candidates(env, s)
    frame = np.clip(render_state(env, s).astype(np.float32)
                    + np.random.default_rng(1).normal(0, 30, (64, 64, 3)), 0, 255)
    m = template_match(cands, frame[None], search_px=0)
    assert m["dy"][0] == 0 and m["dx"][0] == 0 and not m["edge"][0]
    assert angles[m["index"][0]] == pytest.approx(template_angle(env, frame, s))


def test_edge_flag(env):
    s = pose_state((256.0, 256.0), 0.5, (100.0, 100.0))
    _, cands = template_candidates(env, s, step_deg=30.0)
    shifted = np.roll(render_state(env, s), (0, 4), axis=(0, 1))
    m = template_match(cands, shifted[None].astype(np.float32), search_px=4)
    assert m["dx"][0] == 4 and m["edge"][0]

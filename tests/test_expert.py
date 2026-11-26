import numpy as np
import pytest

from wm.data import expert as X
from wm.envs.pusht import BLOCK_COG_LOCAL
from wm.repro import REPO_ROOT, load_yaml

CFG = load_yaml(REPO_ROOT / "configs/data.yaml")["expert"]


def state_with_cog(p, b):
    """State whose block centre of gravity is b (angle 0)."""
    return np.array([*p, b[0] - BLOCK_COG_LOCAL[0], b[1] - BLOCK_COG_LOCAL[1], 0.0], np.float32)


def make(cfg=CFG, seed=0, subgoal=(400.0, 256.0), u=0.0):
    e = X.ScriptedPusher(cfg, np.random.default_rng(seed))
    e._new_segment()
    e.subgoal, e.u = np.array(subgoal), u
    return e


def test_helpers():
    np.testing.assert_allclose(X.unit(np.array([3.0, 4.0])), [0.6, 0.8])
    np.testing.assert_allclose(X.rot90(np.array([1.0, 0.0])), [0.0, 1.0])
    np.testing.assert_allclose(X.clip_norm(np.array([30.0, 40.0]), 25), [15.0, 20.0])
    np.testing.assert_allclose(X.clip_norm(np.array([3.0, 4.0]), 25), [3.0, 4.0])


def test_contact_point_behind_block_with_offset():
    b, d = np.array([256.0, 256.0]), np.array([1.0, 0.0])
    np.testing.assert_allclose(X.contact_point(b, d, 0.0, 55), [201.0, 256.0])
    np.testing.assert_allclose(X.contact_point(b, d, 10.0, 55), [201.0, 266.0])


def test_approach_step_is_clipped():
    e = make()
    p, b = np.array([100.0, 256.0]), np.array([256.0, 256.0])   # c = (201, 256)
    np.testing.assert_allclose(e.clean_target(p, b), [125.0, 256.0])


def test_push_when_at_contact():
    e = make()
    p, b = np.array([195.0, 260.0]), np.array([256.0, 256.0])   # within 15 px of c
    np.testing.assert_allclose(e.clean_target(p, b), p + [20.0, 0.0])


def test_far_side_detours_on_nearer_side_until_reached():
    e = make()
    b = np.array([256.0, 256.0])
    p = np.array([330.0, 240.0])   # beyond the block (d = +x), on the -n side (n = +y)
    t = e.clean_target(p, b)
    assert e.detour_side == -1.0
    way = X.detour_point(b, np.array([1.0, 0.0]), -1.0, 90)   # (256, 166)
    np.testing.assert_allclose(t, p + X.clip_norm(way - p, 25))
    # still on the far side, not yet reached -> keeps detouring
    p2 = np.array([300.0, 200.0])
    e.clean_target(p2, b)
    assert e.detour_side == -1.0
    # reached the detour point -> heads to the contact point, even though dot(p - b, d) > 0
    p3 = way + np.array([5.0, 0.0])
    t3 = e.clean_target(p3, b)
    assert e.detour_side is None and e.after_detour
    c = X.contact_point(b, np.array([1.0, 0.0]), 0.0, 55)
    np.testing.assert_allclose(t3, p3 + X.clip_norm(c - p3, 25))
    # next step on the far side: no new detour
    e.clean_target(p3 + np.array([-3.0, 0.0]), b)
    assert e.detour_side is None


def test_noise_free_actions_equal_clean():
    cfg = {**CFG, "noise_sigmas": [0.0], "random_eps": [0.0]}
    e = make(cfg)
    s = state_with_cog((100.0, 100.0), (256.0, 256.0))
    noisy, clean = e.act(s)
    np.testing.assert_array_equal(noisy, clean)


def test_actions_in_bounds_and_deterministic():
    rng = np.random.default_rng(1)
    states = np.concatenate([rng.uniform(0, 512, (200, 4)), rng.uniform(-np.pi, np.pi, (200, 1))], 1)
    outs = []
    for _ in range(2):
        e = X.ScriptedPusher(CFG, np.random.default_rng(7))
        outs.append(np.array([e.act(s) for s in states]))
    np.testing.assert_array_equal(outs[0], outs[1])
    assert outs[0].min() >= 0.0 and outs[0].max() <= 512.0


@pytest.mark.parametrize("seed", range(20))
def test_episode_params_from_sets(seed):
    e = X.ScriptedPusher(CFG, np.random.default_rng(seed))
    assert e.sigma in CFG["noise_sigmas"] and e.eps in CFG["random_eps"]


def test_subgoal_and_offset_redrawn_each_segment():
    e = X.ScriptedPusher(CFG, np.random.default_rng(3))
    s = state_with_cog((100.0, 100.0), (256.0, 256.0))
    seen = []
    for t in range(60):
        e.act(s)
        seen.append((tuple(e.subgoal), e.u))
    assert len(set(seen[:25])) == 1 and len(set(seen[25:50])) == 1 and seen[24] != seen[25]
    lo, hi = CFG["subgoal_low"], CFG["subgoal_high"]
    assert all(lo <= v <= hi for sg, _ in seen for v in sg)
    assert all(abs(u) <= CFG["lateral_max"] for _, u in seen)

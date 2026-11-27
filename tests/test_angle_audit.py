"""Phase 5b angle audit: the stored angle convention matches the drawn T (CPU, no checkpoints)."""
import numpy as np
import pytest

from wm.data.collect import collect_split
from wm.envs.pusht import block_cog, make_env
from wm.eval.angle_audit import (angle_err_deg, grid_image, mask_pose, orientation_check, origin_from_cog,
                                 pose_state, render_pose, rerender_consistency, scoring_grid, target_roundtrip,
                                 template_angle)
from wm.repro import REPO_ROOT, load_yaml


@pytest.fixture(scope="module")
def env():
    e = make_env()
    e.reset(seed=0)
    return e


def test_pose_state_puts_cog_where_asked():
    s = pose_state((300.0, 200.0), 1.1)
    assert np.allclose(block_cog(s), [300.0, 200.0]) and np.allclose(origin_from_cog(np.array([300.0, 200.0]), 1.1), s[2:4])


def test_rerender_reproduces_stored_frames(env):
    cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    arrays, _ = collect_split([0], cfg["expert"], 20, env=env)
    r = rerender_consistency(env, arrays["frames"], arrays["states"])
    assert r["mean_abs_diff"] < 0.5 and r["frames"] == 21


@pytest.mark.parametrize("deg", [0.0, 30.0, 90.0, 180.0, 270.0])
def test_mask_angle_follows_state_angle(env, deg):
    frame, state = render_pose(env, (256.0, 256.0), np.radians(deg))
    mp = mask_pose(frame)
    assert mp is not None and angle_err_deg(mp["angle"], state[4]) < 6.0


@pytest.mark.parametrize("deg,stem_down", [(0.0, True), (180.0, False)])
def test_zero_angle_is_stem_down(env, deg, stem_down):
    """At angle 0 the stem (+y body axis) points down the image (row grows with y): the block
    reaches ~75 px below its CoG (stem end) but only ~45 px above (bar top); 180 deg flips it."""
    from wm.eval.angle_audit import block_mask
    frame, _ = render_pose(env, (256.0, 256.0), np.radians(deg))
    rows, _ = np.nonzero(block_mask(frame))
    below, above = rows.max() + 1 - 32.0, 32.0 - rows.min()
    assert (below > above + 2) == stem_down and (above > below + 2) == (not stem_down)


def test_orientation_and_cog_convention(env):
    o = orientation_check(env, [0.0, 45.0, 135.0, 225.0, 315.0], [(200.0, 220.0), (330.0, 300.0)])
    assert o["angle_err_mean_deg"] < 5.0 and o["cog_closer_share"] == 1.0
    assert o["centroid_to_cog_mean_px"] < 10.0 < o["centroid_to_origin_mean_px"]


def test_target_roundtrip_and_wraps():
    rng = np.random.default_rng(0)
    s = np.zeros((50, 5)); s[:, :4] = rng.uniform(50, 460, (50, 4)); s[:, 4] = rng.uniform(-np.pi, np.pi, 50)
    r = target_roundtrip(s)
    assert r["max_block_deg"] < 1e-3 and r["max_block_px"] < 1e-3
    assert np.allclose(r["wrap_cases_deg"], [2.0, 2.0, 0.0], atol=1e-6)


def test_template_oracle_recovers_angle(env):
    for deg in (12.0, 147.0, -100.0):
        frame, state = render_pose(env, (240.0, 280.0), np.radians(deg), (60.0, 450.0))
        assert angle_err_deg(template_angle(env, frame.astype(np.float32), state), state[4]) <= 5.0   # 5 deg steps


def test_scoring_grid_and_image(env):
    f, s = scoring_grid(env, 90.0, [[200, 200], [300, 300]])
    assert f.shape == (8, 64, 64, 3) and s.shape == (8, 5)
    img = grid_image(env, [0.0, 90.0])
    assert img.ndim == 3 and img.shape[0] > 64 * 4 and img.dtype == np.uint8

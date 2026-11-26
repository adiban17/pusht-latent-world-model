import numpy as np
import pytest

from wm.envs import pusht as E
from wm.jobs.env_check import ROUNDTRIP_STATES


@pytest.fixture(scope="module")
def env():
    e = E.make_env()
    e.reset(seed=0)
    yield e
    e.close()


def test_render64(env):
    obs, _ = env.reset(seed=1)
    assert obs["pixels"].shape == (96, 96, 3)
    frame = E.render64(obs["pixels"])
    assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
    assert frame.std() > 0  # not a blank image


def test_get_state(env):
    env.reset(seed=2)
    s = E.get_state(env)
    assert s.shape == (5,) and s.dtype == np.float32
    assert -np.pi <= s[4] < np.pi


@pytest.mark.parametrize("state", list(ROUNDTRIP_STATES))
def test_reset_roundtrip(env, state):
    obs = E.reset_to_state(env, state)
    got = E.get_state(env)
    assert np.linalg.norm(got[:2] - state[:2]) < 1.0      # pusher px
    pos_px, ang_deg = E.pose_error(got, state)
    assert pos_px < 1.0 and abs(ang_deg) < 0.5
    np.testing.assert_allclose(obs["agent_pos"], state[:2], atol=1.0)


def test_pose_error_wraps():
    a = np.array([0, 0, 100, 100, np.radians(179)], np.float32)
    b = np.array([0, 0, 100, 100, np.radians(-179)], np.float32)
    _, ang = E.pose_error(a, b)
    assert ang == pytest.approx(-2.0, abs=1e-3)
    assert E.wrap_angle_deg(190) == pytest.approx(-170)
    pos, _ = E.pose_error(np.array([0, 0, 100, 100, 0.3]), np.array([0, 0, 103, 104, 0.3]))
    assert pos == pytest.approx(5.0, abs=1e-4)  # same angle: plain translation


def test_block_cog_matches_pymunk(env):
    for seed in range(5):
        env.reset(seed=seed)
        u = env.unwrapped
        np.testing.assert_allclose(np.array(u.block.center_of_gravity), E.BLOCK_COG_LOCAL)
        want = np.array(u.block.local_to_world(u.block.center_of_gravity))
        np.testing.assert_allclose(E.block_cog(E.get_state(env)), want, atol=1e-3)


def test_rotation_in_place_has_no_position_error(env):
    """Rotating the T about its centre (as pymunk does) gives ~0 px position error."""
    reset_state = np.array([100.0, 100.0, 300.0, 300.0, 0.0], np.float32)
    E.reset_to_state(env, reset_state)
    u = env.unwrapped
    for deg in (30, 90, 179):
        u.block.angle = np.radians(deg)  # pymunk keeps the centre of gravity fixed
        pos, ang = E.pose_error(E.get_state(env), reset_state)
        assert pos < 1e-3
        assert abs(ang) == pytest.approx(deg, abs=1e-3)


def test_is_success_thresholds():
    goal = np.array([0, 0, 200, 200, 0.0], np.float32)

    def at(dx, deg):
        return np.array([0, 0, 200 + dx, 200, np.radians(deg)], np.float32)

    assert E.is_success(at(19.9, 0), goal)
    assert not E.is_success(at(20.0, 0), goal)
    assert E.is_success(at(0, 19.9), goal)
    assert E.is_success(at(0, -19.9), goal)
    assert not E.is_success(at(0, 20.1), goal)
    assert not E.is_success(at(0, -20.1), goal)


def test_normalisation_roundtrips():
    img = np.random.default_rng(0).integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
    x = E.normalize_image(img)
    assert x.shape == (2, 3, 64, 64) and x.dtype == np.float32
    assert x.min() >= -0.5 and x.max() <= 0.5
    np.testing.assert_array_equal(E.denormalize_image(x), img)

    xy = np.array([[0.0, 512.0], [256.0, 100.0]])
    np.testing.assert_allclose(E.normalize_pos(xy), [[-1, 1], [0, 100 / 256 - 1]], atol=1e-6)
    np.testing.assert_allclose(E.denormalize_pos(E.normalize_pos(xy)), xy, atol=1e-4)

    sc = E.angle_features(np.array([0.0, np.pi / 2]))
    np.testing.assert_allclose(sc, [[0, 1], [1, 0]], atol=1e-6)

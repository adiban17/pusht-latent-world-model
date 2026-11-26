import gymnasium as gym
import numpy as np
import pytest

from wm.data import collect as C
from wm.envs.pusht import BLOCK_COG_LOCAL, make_env
from wm.repro import REPO_ROOT, load_yaml

CFG = load_yaml(REPO_ROOT / "configs/data.yaml")
STEPS = 12


@pytest.fixture(scope="module")
def data():
    return C.collect_split([0, 1, 5], CFG["expert"], STEPS)


def test_shapes_and_dtypes(data):
    arrays, _ = data
    F = 3 * (STEPS + 1)
    assert arrays["frames"].shape == (F, 64, 64, 3) and arrays["frames"].dtype == np.uint8
    for k in ("states", "actions", "clean_actions"):
        assert arrays[k].dtype == np.float32 and len(arrays[k]) == F
    assert arrays["states"].shape[1] == 5 and arrays["actions"].shape[1] == 2
    for k in ("episode_starts", "episode_lengths", "episode_seeds"):
        assert arrays[k].dtype == np.int64
    np.testing.assert_array_equal(arrays["episode_lengths"], [STEPS + 1] * 3)
    np.testing.assert_array_equal(arrays["episode_starts"], [0, STEPS + 1, 2 * (STEPS + 1)])
    np.testing.assert_array_equal(arrays["episode_seeds"], [0, 1, 5])
    assert C.transitions(arrays) == 3 * STEPS


def test_last_frame_has_zero_actions(data):
    arrays, _ = data
    last = arrays["episode_starts"] + arrays["episode_lengths"] - 1
    assert not arrays["actions"][last].any() and not arrays["clean_actions"][last].any()
    mask = C.step_mask(arrays)
    assert mask.sum() == 3 * STEPS and not mask[last].any()


def test_first_state_matches_env_reset(data):
    arrays, _ = data
    env = make_env()
    obs, _ = env.reset(seed=5)
    np.testing.assert_allclose(arrays["states"][arrays["episode_starts"][2], :2], obs["agent_pos"],
                               atol=1e-3)
    env.close()


def test_deterministic(data):
    again, _ = C.collect_split([0, 1, 5], CFG["expert"], STEPS)
    for k, v in data[0].items():
        np.testing.assert_array_equal(v, again[k], err_msg=k)


class EarlyTerminate(gym.Wrapper):
    """Reports terminated=True from the 3rd step on."""

    def reset(self, **kw):
        self.n = 0
        return self.env.reset(**kw)

    def step(self, action):
        obs, r, _, trunc, info = self.env.step(action)
        self.n += 1
        return obs, r, self.n >= 3, trunc, info


class RefuseAfterTerminate(EarlyTerminate):
    def step(self, action):
        if self.n >= 3:
            raise RuntimeError("episode is over")
        return super().step(action)


def test_never_ends_early():
    env = EarlyTerminate(make_env())
    arrays, info = C.collect_split([0], CFG["expert"], STEPS, env=env)
    assert arrays["episode_lengths"][0] == STEPS + 1 and info["terminated_episodes"] == 1
    assert arrays["actions"][: STEPS].any(axis=1).all()


def test_refused_step_replaces_episode_with_seed_from_2000(monkeypatch):
    calls = []
    real = C.collect_episode

    def fake(env, seed, cfg, steps):
        calls.append(seed)
        if seed < 2000:
            return real(RefuseAfterTerminate(env), seed, cfg, steps)
        return real(env, seed, cfg, steps)

    monkeypatch.setattr(C, "collect_episode", fake)
    arrays, info = C.collect_split([7], CFG["expert"], STEPS, env=make_env())
    assert calls == [7, 2000]
    assert arrays["episode_seeds"].tolist() == [2000]
    assert info["replaced"] == [{"seed": 7, "replacement": 2000}]


def hand_arrays(cog_xy, angles_deg):
    """One episode whose block centre follows cog_xy (angles in degrees)."""
    th = np.radians(np.asarray(angles_deg, float))
    cog = np.asarray(cog_xy, float)
    ox, oy = BLOCK_COG_LOCAL
    bx = cog[:, 0] - (np.cos(th) * ox - np.sin(th) * oy)
    by = cog[:, 1] - (np.sin(th) * ox + np.cos(th) * oy)
    n = len(cog)
    states = np.stack([np.zeros(n), np.zeros(n), bx, by, th], 1).astype(np.float32)
    acts = np.full((n, 2), 256.0, np.float32)
    acts[-1] = 0
    return {"states": states, "actions": acts, "clean_actions": acts.copy(),
            "episode_starts": np.array([0]), "episode_lengths": np.array([n])}


def test_health_contact_and_coverage():
    # 4 steps: move 10 px, stay, rotate in place 5 deg, stay -> contact 2/4
    cog = [[32, 32], [42, 32], [42, 32], [42, 32], [42, 32]]
    ang = [0, 0, 0, 5, 5]
    h = C.health(hand_arrays(cog, ang))
    assert h["contact_ratio"] == pytest.approx(0.5)
    assert h["coverage_8x8"] == pytest.approx(1 / 64)
    assert h["transitions"] == 4
    far = hand_arrays([[32, 32], [480, 480]], [0, 0])
    assert C.health(far)["coverage_8x8"] == pytest.approx(2 / 64)

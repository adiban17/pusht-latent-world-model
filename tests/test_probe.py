import numpy as np
import torch

from wm.data.collect import collect_split
from wm.data.windows import FrameData, state_targets
from wm.envs.pusht import block_cog, pose_error
from wm.models.probe import EXPECTED_PARAMS, build_probe, decode_targets, param_counts, probe_errors
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.train.common import loss_window_means, resolve_config
from wm.train.train_probe import run_training

CFG = load_yaml(REPO_ROOT / "configs/probe.yaml")


def random_states(n, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros((n, 5), np.float32)
    s[:, :4] = rng.uniform(50, 460, (n, 4))
    s[:, 4] = rng.uniform(-np.pi, np.pi, n)
    return s


def test_param_counts():
    counts = param_counts(build_probe(CFG))
    assert counts == EXPECTED_PARAMS and counts["total"] == 1_740_518


def test_targets_round_trip_to_cog_and_angle():
    s = random_states(50)
    y = state_targets(s)
    assert y.shape == (50, 6) and np.abs(y).max() <= 1.0
    errs = probe_errors(decode_targets(y), s)
    assert errs["block_px"].max() < 1e-3 and errs["pusher_px"].max() < 1e-3
    assert errs["block_deg"].max() < 1e-3


def test_errors_match_pose_error():
    s, g = random_states(20, 1), random_states(20, 2)
    errs = probe_errors(decode_targets(state_targets(g)), s)
    for i in range(20):
        pos, ang = pose_error(g[i], s[i])
        assert abs(errs["block_px"][i] - pos) < 1e-3
        assert abs(errs["block_deg"][i] - abs(ang)) < 1e-3


def test_rotation_in_place_is_angle_error_only():
    s = random_states(1)
    g = s.copy()
    g[0, 4] += 0.5                                   # rotate the T ...
    g[0, 2:4] += block_cog(s)[0] - block_cog(g)[0]   # ... in place about its centre of gravity
    errs = probe_errors(decode_targets(state_targets(g)), s)
    assert errs["block_px"][0] < 1e-3 and abs(errs["block_deg"][0] - np.degrees(0.5)) < 1e-3


def test_loss_falls():
    arrays, _ = collect_split([0], load_yaml(REPO_ROOT / "configs/data.yaml")["expert"], steps=20)
    data = FrameData(arrays, "cpu", CFG["val"]["seeds"])
    t = resolve_config(CFG, "smoke", "local")
    t.update(steps=40, warmup_steps=4, val_every=40, ckpt_every=1000, log_every=1000, val_windows=8)
    seed_all(0)
    res = run_training(build_probe(CFG), data, t, torch.device("cpu"))
    first, last = loss_window_means(res["log"], 8, ("loss",))
    assert last["loss"] < first["loss"]
    v = res["val_log"][-1]
    assert {"block_px_mean", "block_deg_median", "pusher_px_mean", "baseline_block_px_mean"} <= set(v)

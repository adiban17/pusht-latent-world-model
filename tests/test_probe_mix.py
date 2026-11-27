"""Phase 5b fair image judge: label-consistent shifts, typed batches, coord_gap variant, acceptance."""
import numpy as np
import torch

from wm.data.judge_mix import (FRAME_TYPES, NORM_PER_CELL, MixData, concat_arrays, select_episodes,
                               shift_with_labels, train_split)
from wm.data.windows import state_targets
from wm.models.pixel_unet import PixelUNet
from wm.models.probe import EXPECTED_PARAMS, EXPECTED_PARAMS_COORD_GAP, StateProbe, build_probe, param_counts
from wm.models.world_model import WorldModel
from wm.repro import REPO_ROOT, load_yaml
from wm.train.train_probe_mix import acceptance, typed_errors


def arrays(n_eps=3, length=6, seed=0, seeds=None):
    rng = np.random.default_rng(seed)
    n = n_eps * length
    s = np.zeros((n, 5), np.float32)
    s[:, :4] = rng.uniform(60, 450, (n, 4)); s[:, 4] = rng.uniform(-np.pi, np.pi, n)
    return {"frames": rng.integers(0, 256, (n, 64, 64, 3), dtype=np.uint8), "states": s,
            "actions": rng.uniform(0, 512, (n, 2)).astype(np.float32),
            "episode_starts": np.arange(0, n, length, dtype=np.int64),
            "episode_lengths": np.full(n_eps, length, np.int64),
            "episode_seeds": np.array(seeds or list(range(n_eps)), np.int64)}


def models():
    torch.manual_seed(0)
    return {"wm": WorldModel(proprio=True).eval(), "wm_1step": WorldModel(proprio=True).eval(), "pixel": PixelUNet().eval()}


def test_shift_moves_marker_and_labels_together():
    img = torch.zeros(1, 3, 64, 64)
    img[0, :, 20, 30] = 1.0                                     # marker at row 20 (y), col 30 (x)
    y = torch.from_numpy(state_targets(np.array([[30 * 8.0, 20 * 8.0, 30 * 8.0, 20 * 8.0, 0.3]], np.float32)))
    for seed in range(5):
        out, t = shift_with_labels(img, y, 4, torch.Generator().manual_seed(seed))
        r, c = np.unravel_index(int(out[0, 0].argmax()), (64, 64))
        dx, dy = (t[0, 0] - y[0, 0]) / NORM_PER_CELL, (t[0, 1] - y[0, 1]) / NORM_PER_CELL
        assert (c - 30, r - 20) == (round(float(dx)), round(float(dy)))
        d = t - y
        assert torch.allclose(d[0, 0:2], d[0, 2:4]) and torch.equal(t[0, 4:], y[0, 4:])   # angle unchanged


def test_select_concat_and_train_split():
    a = arrays(3, 5, seeds=[0, 190, 2])
    tr = train_split(a, [190, 199])
    assert tr["episode_seeds"].tolist() == [0, 2] and len(tr["frames"]) == 10
    assert np.array_equal(tr["frames"][5:], a["frames"][10:])
    c = concat_arrays(tr, select_episodes(a, np.array([False, True, False])))
    assert c["episode_starts"].tolist() == [0, 5, 10] and len(c["frames"]) == 15


def test_typed_batch_balanced_and_labels_aligned():
    a = arrays()
    d = MixData(a, "cpu", models())
    x, y = d.sample(8, torch.Generator().manual_seed(0), pad=0)
    assert x.shape == (8, 3, 64, 64) and y.shape == (8, 6)
    t = torch.tensor([0, 7])
    _, yt, st = d.images("pixel_1step", t)                  # label = the next frame's state
    assert np.array_equal(st, a["states"][[1, 8]]) and torch.allclose(yt, d.targets[[1, 8]])
    _, yr, sr = d.images("wm_recon", t)
    assert np.array_equal(sr, a["states"][[0, 7]])
    assert 5 not in d.pool("pixel_1step").tolist() and 5 in d.pool("real").tolist()   # episode end
    xr, _, _ = d.images("real", torch.tensor([3]))
    assert xr.min() >= -0.5 and xr.max() <= 0.5


def test_coord_gap_variant_and_flatten_unchanged():
    cfg = load_yaml(REPO_ROOT / "configs/probe_mix.yaml")
    assert param_counts(build_probe(cfg)) == EXPECTED_PARAMS
    m = StateProbe(256, "coord_gap")
    assert param_counts(m) == EXPECTED_PARAMS_COORD_GAP
    assert m(torch.zeros(3, 3, 64, 64)).shape == (3, 6)
    assert param_counts(build_probe(load_yaml(REPO_ROOT / "configs/probe.yaml"))) == EXPECTED_PARAMS


def test_typed_errors_and_acceptance():
    d = MixData(arrays(), "cpu", models())
    out = typed_errors(StateProbe(), d)
    assert set(out) == set(FRAME_TYPES) and out["pixel_1step"]["n"] == 15 and out["real"]["n"] == 18
    acc = {"real_angle_deg": 15.0, "real_block_px": 13.4, "derived_ratio": 1.5}
    good = {k: {"block_px_mean": 10.0, "block_deg_mean": 5.0, "pusher_px_mean": 10.0} for k in FRAME_TYPES}
    assert acceptance(good, acc)["accepted"]
    bad = {**good, "wm_recon": {"block_px_mean": 16.0, "block_deg_mean": 90.0, "pusher_px_mean": 10.0}}
    r = acceptance(bad, acc)
    assert not r["accepted"] and not r["checks"]["wm_recon_block_px"] and r["checks"]["wm_recon_pusher_px"]
    angle_only = {**good, "wm_recon": {"block_px_mean": 10.0, "block_deg_mean": 90.0, "pusher_px_mean": 10.0}}
    assert acceptance(angle_only, acc)["accepted"]           # derived-type angle is reported, not gated

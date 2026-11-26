"""Phase 5 rollout evaluation: metrics, clipping, bootstrap, filmstrip, skip rule, smoke job."""
import importlib.util
import json

import numpy as np
import pytest
import torch

from wm.data.windows import frames_to_model
from wm.eval.rollouts import (PIXEL_HI, PIXEL_LO, bootstrap_indices, clip_frames, diff_ci, latent_nmse,
                              latent_sq_err, mean_ci, pixel_mse, predict_copy_last, predict_pixel, predict_wm,
                              probe_scores, pusher_px, recon_wm, reduction_ci, summarize)
from wm.models.pixel_unet import PixelUNet
from wm.models.probe import StateProbe
from wm.models.world_model import WorldModel
from wm.repro import REPO_ROOT, load_yaml
from wm.viz.media import multirow_filmstrip

spec = importlib.util.spec_from_file_location("run", REPO_ROOT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

C, H = 3, 10


def windows(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    frames = frames_to_model(torch.randint(0, 256, (b, C + H, 64, 64, 3), generator=g, dtype=torch.uint8))
    actions = torch.rand(b, C + H - 1, 2, generator=g) * 2 - 1
    pusher = torch.rand(b, C + H, 2, generator=g) * 2 - 1
    return frames, actions, pusher


def random_states(shape, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros((*shape, 5), np.float32)
    s[..., :4] = rng.uniform(50, 460, (*shape, 4))
    s[..., 4] = rng.uniform(-np.pi, np.pi, shape)
    return s


# ---- clipping ----------------------------------------------------------------------------------
def test_clip_frames_to_real_range():
    x = torch.linspace(-3, 3, 100)
    y = clip_frames(x)
    assert y.min() == PIXEL_LO and y.max() == PIXEL_HI
    real = frames_to_model(torch.tensor([0, 255], dtype=torch.uint8).reshape(1, 1, 2, 1).expand(1, 1, 2, 3))
    assert real.min() == PIXEL_LO and real.max() == PIXEL_HI          # same range as real frames


def test_predictions_are_clipped_before_scoring():
    torch.manual_seed(0)
    frames, actions, pusher = windows()
    unet = PixelUNet()
    with torch.no_grad():
        unet.ups[-1].bias.fill_(10.0)                                   # Δ far outside the pixel range
    assert unet.rollout(frames[:, :C], actions[:, C - 1:]).max() > 1.0
    pred = predict_pixel(unet, frames, actions)
    assert pred.min() >= PIXEL_LO and pred.max() <= PIXEL_HI
    wm = WorldModel(proprio=True)
    with torch.no_grad():
        wm.decoder.deconvs[-1].bias.fill_(-10.0)                       # decoded frames far below -0.5
    assert wm.decode(torch.zeros(1, 256)).min() < -1.0
    pred_wm, _, _ = predict_wm(wm, frames, actions, pusher)
    assert pred_wm.min() >= PIXEL_LO and pred_wm.max() <= PIXEL_HI
    rec = recon_wm(wm, frames[:, C:])
    assert rec.min() >= PIXEL_LO and rec.max() <= PIXEL_HI


# ---- predictions ---------------------------------------------------------------------------------
@pytest.mark.parametrize("proprio", [True, False])
def test_predict_wm_shapes(proprio):
    torch.manual_seed(0)
    frames, actions, pusher = windows()
    pred, z_hat, p_hat = predict_wm(WorldModel(proprio=proprio), frames, actions, pusher)
    assert pred.shape == (4, H, 3, 64, 64) and z_hat.shape == (4, H, 256)
    assert (p_hat is not None and p_hat.shape == (4, H, 2)) if proprio else p_hat is None
    assert torch.isfinite(pred).all()


def test_teacher_forced_model_is_rolled_open_loop():
    """wm_1step is evaluated open-loop: predict_wm gives the same result with teacher_forcing on."""
    torch.manual_seed(0)
    frames, actions, pusher = windows()
    a = WorldModel(proprio=True)
    b = WorldModel(proprio=True, teacher_forcing=True)
    b.load_state_dict(a.state_dict())
    assert torch.equal(predict_wm(a, frames, actions, pusher)[1], predict_wm(b, frames, actions, pusher)[1])


def test_zero_delta_pixel_model_equals_copy_last():
    frames, actions, _ = windows()
    unet = PixelUNet()
    with torch.no_grad():
        unet.ups[-1].weight.zero_()
        unet.ups[-1].bias.zero_()
    copy = predict_copy_last(frames, C)
    assert copy.shape == (4, H, 3, 64, 64) and torch.equal(copy[:, 4], frames[:, C - 1])
    assert torch.allclose(predict_pixel(unet, frames, actions), copy)


# ---- scores ----------------------------------------------------------------------------------------
def test_perfect_predictor_matches_real_reference():
    torch.manual_seed(0)
    frames, _, _ = windows()
    real = frames[:, C:]
    states = random_states((4, H))
    probe = StateProbe()
    a, b = probe_scores(probe, real.clone(), states), probe_scores(probe, real, states)
    for k in a:
        assert a[k].shape == (4, H) and np.allclose(a[k], b[k])
    wm = WorldModel()
    z = wm.encode(real)
    assert np.all(latent_sq_err(z, z) == 0) and np.all(pixel_mse(real, real) == 0)


def test_pusher_px_exact():
    s = random_states((2, 3))
    p = torch.from_numpy(s[..., :2] / 256.0 - 1.0)
    assert np.allclose(pusher_px(p, s), 0.0, atol=1e-4)          # float32 round trip
    assert np.allclose(pusher_px(p + 3.0 / 256.0 * torch.tensor([1.0, 0.0]), s), 3.0, atol=1e-4)


def test_nmse_from_summarize_matches_latent_nmse():
    g = torch.Generator().manual_seed(0)
    z_ref = torch.randn(50, 8, generator=g)
    z_true, z_hat = torch.randn(20, H, 8, generator=g), torch.randn(20, H, 8, generator=g)
    from wm.eval.rollouts import latent_spread
    out = summarize({"latent_sq": {"m": latent_sq_err(z_hat, z_true)}}, bootstrap_indices(20, 50, 0), 0.95,
                    latent_spread(z_ref))
    assert np.allclose(out["nmse"]["m"]["mean"], latent_nmse(z_hat, z_true, z_ref).numpy(), rtol=1e-5)
    # copy-last by hand: ||z_t - z_{t+k}||^2 / spread
    copy = z_true[:, :1].expand_as(z_true)
    hand = ((copy - z_true) ** 2).sum(-1).mean(0) / latent_spread(z_ref)
    assert torch.allclose(latent_nmse(copy, z_true, z_ref), hand)


# ---- bootstrap ------------------------------------------------------------------------------------
def test_bootstrap_constant_deterministic_and_paired():
    idx = bootstrap_indices(30, 200, seed=0)
    assert np.array_equal(idx, bootstrap_indices(30, 200, seed=0)) and idx.shape == (200, 30)
    const = mean_ci(np.full((30, H), 2.5), idx)
    assert np.allclose(const["lo"], 2.5) and np.allclose(const["hi"], 2.5) and np.allclose(const["mean"], 2.5)
    rng = np.random.default_rng(1)
    b = rng.normal(10, 5, (30, H))
    a = b + 1.0                                          # noisy, but the paired difference is exactly 1
    d = diff_ci(a, b, idx)
    assert np.allclose(d["lo"], 1.0) and np.allclose(d["hi"], 1.0)
    assert np.allclose(reduction_ci(b, b, idx)["hi"], 0.0)
    r = reduction_ci(0.5 * b, b, idx)
    assert np.allclose(r["mean"], 0.5) and np.allclose(r["lo"], 0.5)
    ci = mean_ci(b, idx)
    assert all(lo <= m <= hi for lo, m, hi in zip(ci["lo"], ci["mean"], ci["hi"]))


# ---- filmstrip -------------------------------------------------------------------------------------
def test_multirow_filmstrip_size():
    rows = {k: np.zeros((11, 64, 64, 3), np.uint8) for k in ("real", "wm", "wm_1step", "pixel")}
    out = multirow_filmstrip(rows, scale=2, gap=2, label_width=76)
    assert out.shape == (4 * 128 + 3 * 4, 76 + 11 * 128 + 10 * 2, 3) and out.dtype == np.uint8


# ---- skip rule ---------------------------------------------------------------------------------------
def test_is_finished_stale_when_a_checkpoint_hash_changes(monkeypatch):
    reg = run.load_registry()
    entry = reg["jobs"]["eval_rollouts"]
    chash = run.job_config_hash("eval_rollouts", entry, run.load_job_config(entry))
    inputs = run.input_hashes(entry, reg)
    assert set(inputs) == {"ckpt-wm", "ckpt-wm-1step", "ckpt-pixel", "ckpt-probe"}

    class FakeStore:
        def __init__(self, meta):
            self.meta = meta

        def own_metadata(self, name):
            return self.meta

    assert run.is_finished(FakeStore({"complete": True, "config_hash": chash, "inputs": inputs}), entry, chash, reg)
    stale = {**inputs, "ckpt-wm": "000000000000"}
    assert not run.is_finished(FakeStore({"complete": True, "config_hash": chash, "inputs": stale}), entry, chash, reg)
    assert not run.is_finished(FakeStore({"complete": True, "config_hash": chash}), entry, chash, reg)
    # training jobs need no checkpoints: unaffected
    tw = reg["jobs"]["train_wm"]
    th = run.job_config_hash("train_wm", tw, run.load_job_config(tw))
    assert run.input_hashes(tw, reg) == {}
    assert run.is_finished(FakeStore({"complete": True, "config_hash": th}), tw, th, reg)


# ---- the job, end to end on synthetic data with untrained checkpoints --------------------------------
def synthetic_npz(seeds, length, seed):
    rng = np.random.default_rng(seed)
    n = len(seeds) * length
    return {"frames": rng.integers(0, 256, (n, 64, 64, 3), dtype=np.uint8),
            "states": random_states((n,), seed), "actions": rng.uniform(0, 512, (n, 2)).astype(np.float32),
            "clean_actions": rng.uniform(0, 512, (n, 2)).astype(np.float32),
            "episode_starts": np.arange(0, n, length, dtype=np.int64),
            "episode_lengths": np.full(len(seeds), length, np.int64),
            "episode_seeds": np.array(seeds, np.int64)}


def test_eval_rollouts_smoke_job(isolated_dirs, capsys):
    from wm.models.pixel_unet import build_pixel_unet
    from wm.models.probe import build_probe
    from wm.models.world_model import build_world_model
    from wm.registry import producer
    from wm.store import LocalStore

    store = LocalStore(isolated_dirs / "store")
    data_dir = isolated_dirs / "data"
    data_dir.mkdir()
    np.savez_compressed(data_dir / "pusht_train.npz", **synthetic_npz([0, 1], 20, 0))
    np.savez_compressed(data_dir / "pusht_test.npz", **synthetic_npz([1000], 40, 1))   # 28 windows
    store.push("pusht-data-smoke", data_dir, {"complete": True, "config_hash": "x"})
    torch.manual_seed(0)
    for art, cfg_file, build in (("ckpt-wm", "wm.yaml", build_world_model),
                                 ("ckpt-wm-1step", "wm_1step.yaml", build_world_model),
                                 ("ckpt-pixel", "pixel.yaml", build_pixel_unet),
                                 ("ckpt-probe", "probe.yaml", build_probe)):
        cfg = load_yaml(REPO_ROOT / "configs" / cfg_file)
        h = producer(art)[1]
        d = isolated_dirs / art
        d.mkdir()
        torch.save({"model": build(cfg).state_dict(), "config": cfg, "step": 1, "config_hash": h, "git": "test"},
                   d / "model.pt")
        store.push(f"{art}-smoke", d, {"complete": True, "config_hash": h, "step": 1})

    assert run.main(["--job", "eval_rollouts"]) == 0
    out = capsys.readouterr().out
    assert "[FAIL]" not in out and "windows: 20 of 28 test windows" in out
    assert "probe_angle_floor" in out and "block_px_wm_minus_pixel" in out

    art = isolated_dirs / "outputs" / "eval_rollouts" / "artifact"
    res = json.loads((art / "rollouts.json").read_text())
    assert res["windows"]["n"] == 20 and res["headline"]["nmse_row"] == "wm_via_frames"
    assert set(res["metrics"]["block_px"]) == {"wm", "wm_1step", "pixel", "copy_last", "real", "wm_recon"}
    assert set(res["metrics"]["nmse"]) == {"wm_via_frames", "wm_1step", "pixel", "copy_last", "wm", "wm_recon"}
    assert set(res["metrics"]["p_hat_pusher_px"]) == {"wm", "wm_1step", "copy_last"}
    for metric in ("block_px", "block_deg", "pusher_px", "p_hat_pusher_px", "nmse", "pixel_mse"):
        for s in res["metrics"][metric].values():
            assert set(s) == {"mean", "lo", "hi"} and len(s["mean"]) == H
    for key in ("block_px_diff_wm_minus_pixel", "nmse_reduction_wm_via_frames_vs_pixel"):
        assert len(res["paired"][key]["lo"]) == H
    assert res["metrics"]["nmse"]["copy_last"]["mean"][0] > 0
    assert len(list((art / "plots").glob("*.png"))) == 5 and len(list((art / "filmstrips").glob("*.png"))) == 6
    meta = store.metadata("results-rollouts-smoke")
    assert meta["complete"] and set(meta["inputs"]) == {"ckpt-wm", "ckpt-wm-1step", "ckpt-pixel", "ckpt-probe"}

    assert run.main(["--job", "eval_rollouts"]) == 0
    assert "already finished, skipping" in capsys.readouterr().out

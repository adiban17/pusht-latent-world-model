import torch

from wm.data.collect import collect_split
from wm.data.windows import WindowData
from wm.models.pixel_unet import (EXPECTED_PARAMS, WM_PARAMS, PixelUNet, build_pixel_unet,
                                  param_counts, within_tolerance)
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.train.common import loss_window_means, resolve_config
from wm.train.train_pixel import run_training

CFG = load_yaml(REPO_ROOT / "configs/pixel.yaml")


def test_param_count_exact_and_within_20_percent():
    total = param_counts(build_pixel_unet(CFG))["total"]
    assert total == EXPECTED_PARAMS == 4_813_731
    assert within_tolerance(total) and abs(total / WM_PARAMS - 1) > 0.15   # why ±15% was widened
    assert not within_tolerance(int(WM_PARAMS * 1.21))


def test_shapes_and_residual_output():
    torch.manual_seed(0)
    m = PixelUNet()
    frames, a = torch.rand(2, 3, 3, 64, 64) - 0.5, torch.rand(2, 2) * 2 - 1
    assert m(frames, a).shape == (2, 3, 64, 64)
    with torch.no_grad():
        m.ups[-1].weight.zero_()
        m.ups[-1].bias.zero_()
    assert torch.equal(m(frames, a), frames[:, -1])         # zero delta -> prediction = last frame


def test_action_changes_prediction():
    torch.manual_seed(0)
    m = PixelUNet()
    frames = torch.rand(1, 3, 3, 64, 64) - 0.5
    assert not torch.allclose(m(frames, torch.zeros(1, 2)), m(frames, torch.ones(1, 2)))


def test_rollout_feeds_back_its_own_predictions():
    torch.manual_seed(0)
    m = PixelUNet()
    frames, actions = torch.rand(2, 3, 3, 64, 64) - 0.5, torch.rand(2, 4, 2) * 2 - 1
    with torch.no_grad():
        out = m.rollout(frames, actions)
        assert out.shape == (2, 4, 3, 64, 64)
        assert torch.allclose(out[:, 0], m(frames, actions[:, 0]))
        window = torch.cat([frames[:, 1:], out[:, :1]], dim=1)          # slid over prediction 1
        assert torch.allclose(out[:, 1], m(window, actions[:, 1]), atol=1e-6)


def test_loss_falls():
    arrays, _ = collect_split([0], load_yaml(REPO_ROOT / "configs/data.yaml")["expert"], steps=20)
    data = WindowData(arrays, "cpu", CFG["val"]["seeds"], window=4)
    t = resolve_config(CFG, "smoke", "local")
    t.update(steps=30, warmup_steps=3, val_every=30, ckpt_every=1000, log_every=1000, val_windows=2)
    seed_all(0)
    res = run_training(build_pixel_unet(CFG), data, t, torch.device("cpu"), horizon=10)
    first, last = loss_window_means(res["log"], 6, ("loss",))
    assert last["loss"] < first["loss"]
    v = res["val_log"][-1]
    assert len(v["mse"]) == 10 and len(v["mse_copy_last"]) == 10 and v["loss"] == v["mse"][0]

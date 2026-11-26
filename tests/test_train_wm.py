import copy

import pytest
import torch

from wm.data.collect import collect_split
from wm.data.windows import WindowData
from wm.models.world_model import build_world_model
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.train.train_wm import loss_window_means, prepare_model, resolve_config, run_training

CFG = load_yaml(REPO_ROOT / "configs/wm.yaml")
DATA_CFG = load_yaml(REPO_ROOT / "configs/data.yaml")
W = dict(CFG["loss"])


@pytest.fixture(scope="module")
def data():
    arrays, _ = collect_split([0], DATA_CFG["expert"], steps=20)
    return WindowData(arrays, "cpu", CFG["val"]["seeds"])


def tcfg(**over):
    t = resolve_config(CFG, "smoke", "local")
    t.update({"batch_size": 2, "val_windows": 2, "log_every": 1000}, **over)
    return t


def fresh_model():
    seed_all(0)
    return build_world_model(CFG)


def test_resolve_config():
    full = resolve_config(CFG, "full", "kaggle")
    assert full["steps"] == 30000 and full["batch_size"] == 64 and full["warmup_steps"] == 500
    assert resolve_config(CFG, "smoke", "local")["steps"] == 50
    assert resolve_config(CFG, "smoke", "kaggle")["steps"] == 300


def test_loss_falls(data):
    t = tcfg(steps=40, warmup_steps=5, val_every=40, ckpt_every=1000, batch_size=4)
    res = run_training(fresh_model(), data, t, torch.device("cpu"), W)
    first, last = loss_window_means(res["log"], 8)
    assert last["loss"] < first["loss"]
    assert res["complete"] and len(res["val_log"]) == 1


def test_resume_is_bit_identical(data):
    t = tcfg(steps=16, warmup_steps=4, val_every=8, ckpt_every=8)
    dev = torch.device("cpu")
    full = run_training(fresh_model(), data, t, dev, W)

    saved = []
    part = run_training(fresh_model(), data, t, dev, W, stop_at=8,
                        on_checkpoint=lambda s: saved.append(copy.deepcopy(s)))
    assert part["step"] == 8 and not part["complete"] and saved[-1]["step"] == 8
    resumed = run_training(fresh_model(), data, t, dev, W, start=saved[-1])

    assert resumed["step"] == 16 and resumed["complete"]
    assert [r["loss"] for r in resumed["log"]] == [r["loss"] for r in full["log"]]
    for k, v in full["state"]["model"].items():
        assert torch.equal(v, resumed["state"]["model"][k]), k
    assert len(resumed["val_log"]) == 2


def test_keep_best_picks_min_nmse_and_resumes_identically(data):
    t = tcfg(steps=16, warmup_steps=4, val_every=4, ckpt_every=8)
    dev = torch.device("cpu")
    full = run_training(fresh_model(), data, t, dev, W, keep_best=True)
    nmse = {v["step"]: v["nmse"][-1] for v in full["val_log"]}
    best_step = min(nmse, key=nmse.get)
    assert full["best"]["step"] == best_step and full["best"]["value"] == nmse[best_step]
    assert "best" in full["state"]

    saved = []
    run_training(fresh_model(), data, t, dev, W, keep_best=True, stop_at=8,
                 on_checkpoint=lambda s: saved.append(copy.deepcopy(s)))
    assert saved[-1]["best"] is not None
    resumed = run_training(fresh_model(), data, t, dev, W, keep_best=True, start=saved[-1])
    assert resumed["best"]["step"] == full["best"]["step"]
    for k, v in full["best"]["model"].items():
        assert torch.equal(v, resumed["best"]["model"][k]), k


def test_v1_has_no_best_key_and_unchanged_log_keys(data):
    t = tcfg(steps=4, warmup_steps=2, val_every=4, ckpt_every=2)
    saved = []
    v1 = build_world_model(load_yaml(REPO_ROOT / "configs/wm_v1.yaml"))
    res = run_training(v1, data, t, torch.device("cpu"), W,
                       on_checkpoint=lambda s: saved.append(s))
    assert "best" not in res and "best" not in res["state"] and "best" not in saved[0]
    assert set(res["log"][0]) == {"loss", "rec", "lat", "img", "var", "z_std", "lr"}


def test_proprio_fg_training_runs_and_validates(data):
    cfg = {**CFG, "fg_weight": True, "proprio": True, "loss": {**CFG["loss"], "proprio": 1.0}}
    seed_all(0)
    m = prepare_model(build_world_model(cfg), data)
    assert m.fg_mean.abs().sum() > 0
    t = tcfg(steps=6, warmup_steps=2, val_every=6, ckpt_every=1000)
    res = run_training(m, data, t, torch.device("cpu"), dict(cfg["loss"]))
    assert "proprio" in res["log"][0]
    v = res["val_log"][-1]
    assert len(v["pusher_px"]) == 10 and len(v["pusher_px_copy_last"]) == 10 and "proprio" in v

import copy

import pytest
import torch

from wm.data.collect import collect_split
from wm.data.windows import WindowData
from wm.models.world_model import build_world_model
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.train.train_wm import loss_window_means, resolve_config, run_training

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

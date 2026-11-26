import math

import pytest
import torch

from wm.train.common import StepTimer, load_checkpoint, lr_at, save_checkpoint


def test_lr_schedule():
    base = 3e-4
    assert lr_at(0, base, 500, 30000) == pytest.approx(base / 500)
    assert lr_at(499, base, 500, 30000) == pytest.approx(base)
    assert lr_at(500, base, 500, 30000) == pytest.approx(base)
    mid = 500 + (30000 - 500) // 2
    assert lr_at(mid, base, 500, 30000) == pytest.approx(base / 2, rel=1e-3)
    assert lr_at(30000, base, 500, 30000) == pytest.approx(0.0, abs=1e-12)
    lrs = [lr_at(s, base, 500, 30000) for s in range(500, 30000, 100)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_checkpoint_roundtrip(tmp_path):
    state = {"step": 7, "w": torch.arange(5.0), "log": [{"loss": 1.5}],
             "rng": torch.Generator().manual_seed(1).get_state()}
    path = save_checkpoint(tmp_path / "c" / "ckpt.pt", state)
    got = load_checkpoint(path)
    assert got["step"] == 7 and torch.equal(got["w"], state["w"]) and got["log"] == state["log"]
    assert torch.equal(got["rng"], state["rng"])
    assert not path.with_suffix(".pt.tmp").exists()


def test_step_timer_skips_warmup():
    t = StepTimer(skip=2)
    for _ in range(5):
        t.start(); t.stop()
    assert t.seen == 5 and t.timed == 3
    assert t.rate() is None or math.isfinite(t.rate())

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


def test_train_loop_resume_bit_identical_for_a_non_wm_model():
    """The shared loop (used by all five training jobs) resumes exactly, here with the probe."""
    import copy

    import numpy as np

    from wm.data.windows import FrameData
    from wm.models.probe import StateProbe
    from wm.repro import seed_all
    from wm.train.common import train_loop
    from wm.train.train_probe import probe_loss

    rng = np.random.default_rng(0)
    n = 40
    states = np.zeros((n, 5), np.float32)
    states[:, :4] = rng.uniform(50, 460, (n, 4))
    arrays = {"frames": rng.integers(0, 255, (n, 64, 64, 3), dtype=np.uint8), "states": states,
              "episode_starts": np.array([0, 20]), "episode_lengths": np.array([20, 20]),
              "episode_seeds": np.array([0, 190])}
    data = FrameData(arrays, "cpu", [190, 199])
    t = {"lr": 3e-4, "weight_decay": 1e-4, "warmup_steps": 2, "steps": 10, "grad_clip": 1.0,
         "amp": False, "val_every": 5, "ckpt_every": 5, "log_every": 100, "seed": 0, "batch_size": 4}
    dev = torch.device("cpu")

    def go(**kw):
        seed_all(0)
        m = StateProbe()
        return train_loop(m, lambda g: data.sample(4, g), lambda b: probe_loss(m, *b),
                          lambda: {"x": 1.0}, t, dev, ("loss",), **kw)

    full = go()
    saved = []
    part = go(stop_at=5, on_checkpoint=lambda s: saved.append(copy.deepcopy(s)))
    assert part["step"] == 5 and not part["complete"]
    resumed = go(start=saved[-1])
    assert resumed["complete"] and [r["loss"] for r in resumed["log"]] == [r["loss"] for r in full["log"]]
    for k, v in full["state"]["model"].items():
        assert torch.equal(v, resumed["state"]["model"][k]), k

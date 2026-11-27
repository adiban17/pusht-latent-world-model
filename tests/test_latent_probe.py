"""Phase 5b latent judge: shapes, params, learning on a toy latent set, verdict thresholds."""
import numpy as np
import torch

from wm.jobs._judges import angle_verdict
from wm.models.latent_probe import EXPECTED_PARAMS, LatentProbe, build_latent_probe, param_counts
from wm.repro import REPO_ROOT, load_yaml
from wm.train.common import resolve_config
from wm.train.train_latent_probe import LatentData, run_training


def test_params_and_shapes():
    cfg = load_yaml(REPO_ROOT / "configs/latent_probe.yaml")
    m = build_latent_probe(cfg)
    assert param_counts(m) == EXPECTED_PARAMS and m(torch.zeros(4, 7, 256)).shape == (4, 7, 6)


def test_learns_a_linear_toy_problem():
    rng = np.random.default_rng(0)
    s = np.zeros((600, 5), np.float32)
    s[:, :4] = rng.uniform(60, 450, (600, 4)); s[:, 4] = rng.uniform(-np.pi, np.pi, 600)
    from wm.data.windows import state_targets
    z = torch.from_numpy(state_targets(s) @ rng.normal(size=(6, 256)).astype(np.float32))
    cfg = load_yaml(REPO_ROOT / "configs/latent_probe.yaml")
    tcfg = {**resolve_config(cfg, "smoke", "local"), "steps": 300, "val_every": 100, "lr": 3e-3}
    torch.manual_seed(0)
    m = LatentProbe()
    res = run_training(m, LatentData(z[:500], s[:500], "cpu"), LatentData(z[500:], s[500:], "cpu"), tcfg,
                       torch.device("cpu"))
    assert res["best"] is not None and res["val_log"][-1]["loss"] < res["val_log"][0]["loss"] < 1.0


def test_angle_verdict_thresholds():
    assert angle_verdict(10.0, 20.0, 45.0).startswith("present")
    assert angle_verdict(30.0, 20.0, 45.0) == "weak"
    assert angle_verdict(60.0, 20.0, 45.0).startswith("absent")

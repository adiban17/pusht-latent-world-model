
import numpy as np
import torch

from wm.data.windows import state_targets
from wm.eval.diagnose import fit_ridge, recon_errors, ridge_predict, ridge_probe, subset
from wm.models.world_model import WorldModel


def test_ridge_recovers_a_linear_map():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(500, 16, generator=g)
    w_true = torch.randn(16, 3, generator=g).double()
    y = (z.double() @ w_true + 0.5).numpy()
    w = fit_ridge(z, y, lam=1e-6)
    assert np.allclose(ridge_predict(w, z), y, atol=1e-4)
    assert torch.allclose(w[-1], torch.full((3,), 0.5, dtype=torch.float64), atol=1e-4)   # bias unpenalised


def test_ridge_probe_on_state_features_is_near_exact():
    rng = np.random.default_rng(0)
    s = np.zeros((400, 5), np.float32)
    s[:, :4] = rng.uniform(50, 460, (400, 4))
    s[:, 4] = rng.uniform(-np.pi, np.pi, 400)
    z = torch.from_numpy(state_targets(s)).float()          # a "latent" that is the target itself
    _, out = ridge_probe(z[:300], s[:300], z[300:], s[300:], lam=1e-6)
    assert out["val"]["block_px_mean"] < 0.1 and out["val"]["block_deg_mean"] < 0.1
    assert out["val_predict_mean"]["block_px_mean"] > 50


def test_recon_errors_foreground_mask():
    torch.manual_seed(0)
    frames = torch.full((4, 64, 64, 3), 255, dtype=torch.uint8)
    frames[:, 10:20, 10:20] = 0                              # a 10x10 dark square on white
    mean = torch.full((3, 64, 64), 0.5)                      # mean frame = white background
    r = recon_errors(WorldModel(), frames, mean)
    assert abs(r["foreground_fraction"] - 100 / 4096) < 1e-6
    assert abs(r["mean_frame_mse_foreground"] - 1.0) < 1e-6  # white vs black on the square only
    assert r["frames"] == 4 and r["mse"] > 0


def test_subset_is_fixed_and_sorted():
    a = subset(torch.arange(100), 10, seed=3)
    assert torch.equal(a, subset(torch.arange(100), 10, seed=3)) and torch.equal(a, a.sort().values)
    assert len(subset(torch.arange(5), 10)) == 5

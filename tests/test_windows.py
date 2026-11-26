import numpy as np
import torch

from wm.data.windows import WindowData, frames_to_model, window_starts
from wm.envs.pusht import normalize_image, normalize_pos
from wm.repro import REPO_ROOT, load_yaml


def fake_arrays(seeds, length=20):
    """Episodes whose frame pixels encode (episode, frame) so windows can be checked exactly."""
    e = len(seeds)
    f = e * length
    frames = np.zeros((f, 64, 64, 3), np.uint8)
    actions = np.zeros((f, 2), np.float32)
    for i in range(e):
        for j in range(length):
            g = i * length + j
            frames[g, 0, 0] = [i, j, 7]
            actions[g] = [10 * i + j, 100 + j] if j < length - 1 else [0, 0]
    return {"frames": frames, "actions": actions,
            "episode_starts": np.arange(e, dtype=np.int64) * length,
            "episode_lengths": np.full(e, length, np.int64),
            "episode_seeds": np.array(seeds, np.int64)}


def test_window_starts():
    s = window_starts(np.array([0, 20]), np.array([20, 20]), 13)
    assert s.tolist() == list(range(0, 8)) + list(range(20, 28))


def test_windows_stay_in_episode_and_align_actions():
    d = WindowData(fake_arrays([5, 190, 6, 199]), "cpu", [190, 199])
    assert d.train_seeds == [5, 6] and d.val_seeds == [190, 199]
    frames, actions = d.sample(64, torch.Generator().manual_seed(0))
    assert frames.shape == (64, 13, 3, 64, 64) and actions.shape == (64, 12, 2)
    ep = ((frames[:, :, 0, 0, 0] + 0.5) * 255).round()
    idx = ((frames[:, :, 1, 0, 0] + 0.5) * 255).round()
    assert (ep == ep[:, :1]).all()                                   # one episode per window
    assert (idx[:, 1:] - idx[:, :-1] == 1).all()                     # consecutive frames
    assert set(ep[:, 0].tolist()) <= {0.0, 2.0}                      # train episodes only
    px = (actions + 1) * 256                                         # action j taken at frame j
    assert torch.allclose(px[..., 1], 100 + idx[:, :-1], atol=1e-3)
    assert (px[..., 1] > 0).all()                                    # never the zero last action


def test_val_windows_only_from_val_seeds_and_fixed():
    d = WindowData(fake_arrays([5, 190, 6, 199]), "cpu", [190, 199])
    starts = d.fixed_val_starts(10, seed=0)
    assert torch.equal(starts, d.fixed_val_starts(10, seed=0))
    frames, _ = d.gather(starts)
    ep = ((frames[:, 0, 0, 0, 0] + 0.5) * 255).round()
    assert set(ep.tolist()) <= {1.0, 3.0}
    assert d.val_frames().shape == (40, 3, 64, 64)


def test_smoke_fallback_val_is_last_train_episode():
    d = WindowData(fake_arrays([0, 1]), "cpu", [190, 199])
    assert d.train_seeds == [0, 1] and d.val_seeds == [1]
    assert "smoke" in d.val_source


def test_same_seed_same_batch():
    d = WindowData(fake_arrays([5, 190, 6]), "cpu", [190, 199])
    a = d.sample(8, torch.Generator().manual_seed(3))
    b = d.sample(8, torch.Generator().manual_seed(3))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_normalisation_matches_env_helpers():
    img = np.random.default_rng(0).integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
    np.testing.assert_allclose(frames_to_model(torch.from_numpy(img)).numpy(), normalize_image(img),
                               atol=1e-6)
    arr = fake_arrays([1, 2])
    d = WindowData(arr, "cpu", [190, 199])
    np.testing.assert_allclose(d.actions.numpy(), normalize_pos(arr["actions"]), atol=1e-6)


def test_val_seeds_match_data_config():
    wm = load_yaml(REPO_ROOT / "configs/wm.yaml")
    data = load_yaml(REPO_ROOT / "configs/data.yaml")
    assert wm["val"]["seeds"] == data["val_seeds"]

import numpy as np
import pytest
import torch

from wm.data.collect import collect_split
from wm.data.windows import GoalData, random_shift
from wm.models.bc import EXPECTED_PARAMS, BCPolicyNet, build_bc, delta_to_target, param_counts
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.train.common import loss_window_means, resolve_config
from wm.train.train_bc import run_training

CFG = load_yaml(REPO_ROOT / "configs/bc.yaml")


def fake_arrays(n_eps=2, length=40):
    """Frames whose pixel (0, 0) encodes (episode, frame index in episode)."""
    f = n_eps * length
    frames = np.zeros((f, 64, 64, 3), np.uint8)
    states = np.zeros((f, 5), np.float32)
    actions = np.zeros((f, 2), np.float32)
    for e in range(n_eps):
        for j in range(length):
            g = e * length + j
            frames[g, 0, 0] = [e, j, 0]
            states[g, :2] = [100 + j, 200 + e]
            actions[g] = [110 + j, 190 + e] if j < length - 1 else [0, 0]
    return {"frames": frames, "states": states, "actions": actions, "clean_actions": actions.copy(),
            "episode_starts": np.arange(n_eps, dtype=np.int64) * length,
            "episode_lengths": np.full(n_eps, length, np.int64),
            "episode_seeds": np.array([0, 190][:n_eps], np.int64)}


def code(images):
    """(episode, frame) codes read back from normalised images [..., 3, 64, 64]."""
    px = torch.round((images[..., :2, 0, 0] + 0.5) * 255).long()
    return px


@pytest.fixture
def data():
    return GoalData(fake_arrays(), "cpu", [190, 199])


def test_param_counts():
    counts = param_counts(build_bc(CFG))
    assert counts == EXPECTED_PARAMS and counts["total"] == 2_528_482


def test_split_and_samples(data):
    assert data.train_idx.tolist() == list(range(0, 39))     # episode seed 0, frames with an action
    assert data.val_idx.tolist() == list(range(40, 79))      # seed 190 is validation


def test_goal_offsets_context_and_targets(data):
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        b = data.sample(64, gen)
        c = code(b["images"])                                # [B, 4, 2] (episode, frame)
        t = b["t"] % 40
        assert (c[:, :, 0] == 0).all()                       # everything from the sample's episode
        k = c[:, 3, 1] - t
        clipped = c[:, 3, 1] == 39
        assert ((k >= 5) & (k <= 30) | clipped).all() and (k >= 1).all()
        assert (c[:, 2, 1] == t).all()
        assert (c[:, 1, 1] == (t - 1).clamp(min=0)).all() and (c[:, 0, 1] == (t - 2).clamp(min=0)).all()
        # target round trip: p + 50 * target == logged action
        p = (b["pusher"] + 1) * 256
        assert torch.allclose(p + 50 * b["target"], torch.stack([110 + t.float(), torch.full_like(p[:, 1], 190)], 1),
                              atol=1e-3)


def test_padding_at_episode_start(data):
    b = data.gather(torch.tensor([0, 1, 41]), torch.tensor([5, 5, 30]))
    c = code(b["images"])
    assert c[0, :3, 1].tolist() == [0, 0, 0]
    assert c[1, :3, 1].tolist() == [0, 0, 1]
    assert c[2, :3, 1].tolist() == [0, 0, 1] and c[2, 3, 1].tolist() == 31


def test_goal_clipped_to_episode_end(data):
    b = data.gather(torch.tensor([30]), torch.tensor([30]))
    assert code(b["images"])[0, 3].tolist() == [0, 39]


def test_random_shift_same_within_sample_different_across():
    gen = torch.Generator().manual_seed(0)
    base = torch.zeros(1, 1, 3, 64, 64)
    base[..., 30, 30] = 1.0
    imgs = base.repeat(16, 4, 1, 1, 1)
    out = random_shift(imgs, 4, gen)
    assert out.shape == imgs.shape
    pos = [tuple((out[b, i, 0] == 1).nonzero()[0].tolist()) for b in range(16) for i in range(4)]
    per_sample = [set(pos[b * 4:(b + 1) * 4]) for b in range(16)]
    assert all(len(s) == 1 for s in per_sample)                   # same shift for all 4 images
    shifts = {next(iter(s)) for s in per_sample}
    assert len(shifts) > 1                                         # differs across samples
    assert all(26 <= r <= 34 and 26 <= c <= 34 for r, c in shifts)  # within ±4 px


def test_delta_to_target_clips():
    p = np.array([[500.0, 10.0]])
    assert delta_to_target(p, np.array([[1.0, -1.0]])).tolist() == [[512.0, 0.0]]
    assert torch.equal(delta_to_target(torch.tensor([[100.0, 100.0]]), torch.tensor([[0.2, -0.4]])),
                       torch.tensor([[110.0, 80.0]]))


def test_loss_falls_and_resume_with_dropout_is_identical():
    import copy

    arrays, _ = collect_split([0], load_yaml(REPO_ROOT / "configs/data.yaml")["expert"], steps=40)
    data = GoalData(arrays, "cpu", CFG["val"]["seeds"])
    t = resolve_config(CFG, "smoke", "local")
    t.update(steps=12, warmup_steps=2, val_every=6, ckpt_every=6, log_every=1000, val_windows=8)
    dev = torch.device("cpu")

    def fresh():
        seed_all(0)
        return BCPolicyNet()

    full = run_training(fresh(), data, t, dev, 4, 50.0)
    saved = []
    run_training(fresh(), data, t, dev, 4, 50.0, stop_at=6, on_checkpoint=lambda s: saved.append(copy.deepcopy(s)))
    torch.manual_seed(123)                                   # disturb the global RNG
    resumed = run_training(fresh(), data, t, dev, 4, 50.0, start=saved[-1])
    assert [r["loss"] for r in resumed["log"]] == [r["loss"] for r in full["log"]]

    t.update(steps=40, val_every=40, ckpt_every=1000)
    res = run_training(fresh(), data, t, dev, 4, 50.0)
    first, last = loss_window_means(res["log"], 8, ("loss",))
    assert last["loss"] < first["loss"]

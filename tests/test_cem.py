"""CEM planner (CLAUDE.md §8): toy quadratic, mean candidate, sigma floor, clipping, waypoints."""
import torch

from wm.planning.cem import (CEMConfig, cem, deltas_to_waypoints, latent_cost, shift_warm_start,
                             waypoints_to_actions)


def test_toy_quadratic_converges():
    """Cost = ||w_H - target||^2 for the last waypoint; CEM must get within a few px."""
    p = torch.tensor([200.0, 200.0])
    target = torch.tensor([290.0, 150.0])
    cfg = CEMConfig(candidates=128, horizon=5, rounds=6, elites=12)

    def score(d):
        return ((deltas_to_waypoints(p, d)[:, -1] - target) ** 2).sum(-1)

    res = cem(score, torch.zeros(5, 2), cfg, torch.Generator().manual_seed(0))
    w = deltas_to_waypoints(p, res["best"][None])[0, -1]
    assert torch.linalg.norm(w - target) < 5.0
    assert res["round_min"][-1] <= res["round_min"][0]


def test_mean_is_a_candidate_and_sigma_floor():
    seen = []

    def score(d):
        seen.append(d.clone())
        return (d ** 2).sum((1, 2))

    mu0 = torch.full((3, 2), 7.0)
    cfg = CEMConfig(candidates=16, horizon=3, rounds=3, elites=4, sigma_floor=2.0)
    res = cem(score, mu0, cfg, torch.Generator().manual_seed(0))
    assert torch.equal(seen[0][0], mu0)              # round 1: the initial mean is candidate 0
    assert all(s.shape == (16, 3, 2) for s in seen)
    assert (res["sigma"] >= 2.0).all()


def test_delta_clip_and_waypoint_clip():
    cfg = CEMConfig(candidates=64, horizon=4, rounds=1, elites=4, sigma_init=500.0)
    got = []
    cem(lambda d: got.append(d) or d.sum((1, 2)), torch.zeros(4, 2), cfg, torch.Generator().manual_seed(1))
    assert got[0].abs().max() <= 50.0
    w = deltas_to_waypoints(torch.tensor([500.0, 5.0]), torch.full((1, 3, 2), 50.0) * torch.tensor([1.0, -1.0]))
    assert w.min() >= 0 and w.max() <= 512
    assert torch.allclose(w[0, 0], torch.tensor([512.0, 0.0]))
    assert torch.allclose(waypoints_to_actions(torch.tensor([0.0, 256.0, 512.0])), torch.tensor([-1.0, 0.0, 1.0]))


def test_latent_cost_and_warm_start():
    z_g = torch.zeros(4)
    z_hat = torch.ones(2, 3, 4)                      # each step distance^2 = 4
    assert torch.allclose(latent_cost(z_hat, z_g, 0.1), torch.full((2,), 4.0 + 0.1 * 8.0))
    assert torch.allclose(latent_cost(z_hat, z_g, 0.0), torch.full((2,), 4.0))
    prev = torch.arange(6.0).view(3, 2)
    ws = shift_warm_start(prev, 3)
    assert torch.equal(ws, torch.tensor([[2.0, 3.0], [4.0, 5.0], [0.0, 0.0]]))
    assert torch.equal(shift_warm_start(None, 2), torch.zeros(2, 2))

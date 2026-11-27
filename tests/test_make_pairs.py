"""Evaluation pairs (CLAUDE.md §9 Pairs)."""
import numpy as np
import pytest

from wm.eval.make_pairs import candidate_pairs, make_pairs


def fake_arrays(n_eps=3, length=101):
    """Block moves 2 px per frame in episode 0, stays still in episode 1, turns 1.5 deg/frame in 2."""
    f = n_eps * length
    states = np.zeros((f, 5), np.float32)
    for e in range(n_eps):
        for j in range(length):
            g = e * length + j
            states[g] = [50, 50, 200 + (2 * j if e == 0 else 0), 200, np.radians(1.5 * j) if e == 2 else 0.0]
    return {"states": states, "episode_starts": np.arange(n_eps, dtype=np.int64) * length,
            "episode_lengths": np.full(n_eps, length, np.int64),
            "episode_seeds": np.array([190, 191, 192], np.int64)}


def test_filter_and_indices():
    a = fake_arrays()
    c = candidate_pairs(a, np.ones(3, bool))
    seeds = {p["episode_seed"] for p in c}
    assert seeds == {190, 192}                         # the still episode gives no pairs
    assert len(c) == 2 * 16                            # starts 0, 5, ..., 75
    for p in c:
        assert p["goal_idx"] - p["start_idx"] == 20
        ep = p["episode_seed"] - 190
        assert ep * 101 <= p["start_idx"] and p["goal_idx"] < (ep + 1) * 101
        assert p["move_px"] >= 30 or abs(p["turn_deg"]) >= 20
    turn = [p for p in c if p["episode_seed"] == 192][0]
    assert turn["turn_deg"] == pytest.approx(-30.0, abs=1e-3) or turn["turn_deg"] == pytest.approx(30.0, abs=1e-3)


def test_seeded_subset_deterministic():
    a = fake_arrays()
    p1, n = make_pairs(a, np.ones(3, bool), 10, seed=0)
    p2, _ = make_pairs(a, np.ones(3, bool), 10, seed=0)
    assert p1 == p2 and n == 32 and [p["pair"] for p in p1] == list(range(10))
    p3, _ = make_pairs(a, np.ones(3, bool), 10, seed=1)
    assert p3 != p1
    only = make_pairs(a, np.array([True, False, False]), 5)[0]
    assert {p["episode_seed"] for p in only} == {190}
    with pytest.raises(ValueError):
        make_pairs(a, np.ones(3, bool), 100)

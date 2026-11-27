"""Phase 7 statistics: exact McNemar, Holm, pair-clustered and paired bootstrap."""
import numpy as np
import pytest

from wm.eval.stats import cluster_bootstrap_rate, holm, mcnemar_exact, paired_bootstrap_diff


def test_mcnemar_known_values():
    a = [True] * 6 + [False] * 6 + [True] * 10
    b = [False] * 6 + [True] * 6 + [True] * 10
    r = mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"], r["discordant"], r["n"]) == (6, 6, 12, 22)
    assert r["p"] == pytest.approx(1.0)
    r = mcnemar_exact([True] * 10 + [False] * 5, [False] * 10 + [False] * 5)
    assert r["p"] == pytest.approx(2 / 1024)
    assert mcnemar_exact([True, False], [True, False])["p"] == 1.0     # no discordant pairs


def test_holm_textbook():
    # Holm (1979)-style example: p = (0.01, 0.04, 0.03, 0.005), m = 4
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])               # capped at 1


def test_holm_monotone_in_p_order():
    p = np.random.default_rng(0).uniform(0, 0.2, 6)
    adj = np.array(holm(p))
    order = np.argsort(p)
    assert (np.diff(adj[order]) >= -1e-12).all() and (adj >= p).all()


def test_cluster_bootstrap():
    r = cluster_bootstrap_rate(np.ones((20, 3), bool), 500, 0)
    assert (r["mean"], r["lo"], r["hi"]) == (1.0, 1.0, 1.0) and r["n_seeds"] == 3 and r["cells"] == 60
    x = np.random.default_rng(1).random((50, 3)) < 0.3
    r1, r2 = cluster_bootstrap_rate(x, 500, 0), cluster_bootstrap_rate(x, 500, 0)
    assert r1 == r2 and r1["lo"] <= r1["mean"] <= r1["hi"]
    assert r1["mean"] == pytest.approx(x.mean())
    one = cluster_bootstrap_rate(x[:, :1], 500, 0)                    # a method with seed 0 only
    assert one["n_seeds"] == 1 and one["mean"] == pytest.approx(x[:, 0].mean())
    assert cluster_bootstrap_rate(np.zeros((0, 3), bool), 10, 0)["n_pairs"] == 0


def test_paired_bootstrap_diff():
    x = np.random.default_rng(2).random(40)
    r = paired_bootstrap_diff(x, x, 300, 0)
    assert r["mean"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0
    r = paired_bootstrap_diff(np.ones(30), np.zeros(30), 300, 0)
    assert (r["mean"], r["lo"], r["hi"]) == (1.0, 1.0, 1.0)

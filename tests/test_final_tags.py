"""Phase 7 rules: failure tags, believed success, difficulty bins, gallery picks, primary tests."""
import numpy as np
import pytest

from wm.eval.final import (TAGS, believed_success, bin_index, bin_labels, difficulty_table, failure_tag,
                           pick_gallery, pooled_summary, primary_tests)


def test_failure_tags_partition():
    assert failure_tag(True, 5.0, 3.0) == "success"
    assert failure_tag(False, 10.0, 35.0) == "position_only"
    assert failure_tag(False, 10.0, -35.0) == "position_only"
    assert failure_tag(False, 30.0, -10.0) == "angle_only"
    assert failure_tag(False, 30.0, 25.0) == "both_failed"
    assert failure_tag(False, 20.0, 20.0) == "both_failed"            # thresholds are strict (< 20)
    with pytest.raises(ValueError):
        failure_tag(False, 10.0, 10.0)                                  # would be a success


def test_believed_success():
    assert believed_success([30.0, 12.0, 25.0], 20.0)
    assert not believed_success([30.0, 21.0], 20.0)
    assert not believed_success([], 20.0) and not believed_success([1.0], None)


def test_bins():
    move = [0, 45, 70, 95, float("inf")]
    turn = [0, 20, 40, 60, 180]
    assert [bin_index(v, move) for v in (20.0, 44.9, 45.0, 94.9, 95.0, 300.0)] == [0, 0, 1, 2, 3, 3]
    assert [bin_index(v, turn) for v in (0.0, 19.99, 20.0, 60.0, 180.0)] == [0, 0, 1, 3, 3]
    with pytest.raises(ValueError):
        bin_index(-1.0, turn)
    assert bin_labels(turn) == ["[0, 20)", "[20, 40)", "[40, 60)", "[60, 180]"]
    assert bin_index(500.0, [0, 45, 70, 95, None]) == 3 and bin_labels([0, 45, None])[-1] == ">= 45"


def test_rollout_angle_verdict():
    from wm.eval.final import rollout_angle_verdict
    pos, neg, zero = {"lo": 1.0, "hi": 5.0}, {"lo": -5.0, "hi": -1.0}, {"lo": -1.0, "hi": 1.0}
    assert rollout_angle_verdict(pos, pos, pos)["verdict"] == "confirmed"
    v = rollout_angle_verdict(pos, pos, zero)          # exact copy_last reference does not agree
    assert v["verdict"] == "partial" and v["B_search"] and not v["B_exact"]
    assert rollout_angle_verdict(pos, neg, neg)["verdict"] == "partial"
    assert rollout_angle_verdict(zero, pos, pos)["verdict"] == "rejected"
    assert rollout_angle_verdict(neg, pos, pos)["verdict"] == "rejected"


def test_gallery_deterministic():
    tags = ["success", "both_failed"] * 10
    a = pick_gallery(tags, "both_failed", 4, 0)
    assert a == pick_gallery(tags, "both_failed", 4, 0) and len(a) == 4 and a == sorted(a)
    assert all(tags[i] == "both_failed" for i in a)
    assert pick_gallery(tags, "angle_only", 4, 0) == []
    assert pick_gallery(["success", "x"], "success", 4, 0) == [0]


def _recs(succ, pos):
    out = []
    for s, p in zip(succ, pos):
        out.append({"success": bool(s), "pos_success": bool(p), "final_pos_px": 10.0 if p else 40.0,
                    "final_ang_deg": 5.0 if s else 40.0, "seconds": 1.0,
                    "tag": failure_tag(bool(s), 10.0 if p else 40.0, 5.0 if s else 40.0)})
    return out


def _pairs(n):
    return [{"move_px": 25.0 + 10 * i, "turn_deg": (-1) ** i * 5.0 * i} for i in range(n)]


def test_primary_tests_and_tables():
    n = 12
    pairs = _pairs(n)
    a0 = [True] * 10 + [False] * 2
    b0 = [False] * 10 + [False] * 2
    rec = {"a": [_recs(a0, a0), _recs(a0, a0)], "b": [_recs(b0, a0)]}
    rows = primary_tests(rec, pairs, [("a", "b")], 200, 0)
    assert [r["metric"] for r in rows] == ["success", "pos_success"]
    s, p = rows
    assert s["a_only"] == 10 and s["b_only"] == 0 and s["seeds_a"] == 2 and s["seeds_b"] == 1
    assert s["p_holm"] == pytest.approx(min(1.0, 2 * s["p"])) and s["claim"] == "a better"
    assert p["discordant"] == 0 and p["p"] == 1.0 and p["claim"].startswith("no claim")
    summ = pooled_summary(rec["a"], pairs, 200, 0)
    assert sum(summ["tags"].values()) == 2 * n and set(summ["tags"]) == set(TAGS)
    assert summ["success"]["mean"] == pytest.approx(10 / 12) and summ["seeds"] == 2
    tab = difficulty_table(pairs, rec, "move_px", [0, 45, 70, 95, float("inf")], 100, 0)
    assert sum(tab["pairs"]) == n and set(tab["methods"]) == {"a", "b"}
    assert tab["methods"]["a"][0]["success"]["n_pairs"] == tab["pairs"][0]

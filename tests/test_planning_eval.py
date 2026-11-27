"""Planning evaluation rules (Phase 6, pre-registered): Wilson CI, summaries, BC selection,
sweep freeze rule, imagined-success threshold."""
import numpy as np
import pytest

from wm.eval.planning import calibrate_tau, select_bc, select_setting, success_matrix, summarize, wilson_ci


def test_wilson_known_values():
    lo, hi = wilson_ci(5, 50)
    assert lo == pytest.approx(0.0435, abs=1e-3) and hi == pytest.approx(0.2136, abs=1e-3)
    lo, hi = wilson_ci(0, 3)
    assert lo == 0.0 and hi == pytest.approx(0.5615, abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 1.0)


def ep(success, pos_success, start_pos=50.0, final_pos=30.0, steps=40):
    return {"success": success, "pos_success": pos_success, "start_pos_px": start_pos, "start_ang_deg": 0.0,
            "final_pos_px": final_pos, "final_ang_deg": 10.0, "min_pos_px": final_pos, "steps": steps,
            "seconds": 1.0, "policy_s": 0.5, "env_s": 0.5, "costs": [], "terminal_costs": []}


def test_summarize_groups_and_subset():
    pairs = [{"turn_deg": 30.0}, {"turn_deg": -25.0}, {"turn_deg": 5.0}, {"turn_deg": 0.0}]
    eps = [ep(True, True, steps=10), ep(False, True, start_pos=10.0), ep(False, True), ep(False, False)]
    s = summarize(eps, pairs)
    assert s["success"]["k"] == 1 and s["success"]["n"] == 4
    assert s["pos_success"]["n"] == 3 and s["pos_success"]["k"] == 2       # pair 2 starts < 20 px
    assert s["pos_success_all"]["k"] == 3
    g = s["groups"]
    assert g["rotation"]["pairs"] == 2 and g["translation"]["pairs"] == 2
    assert g["rotation"]["success"]["k"] == 1 and g["translation"]["success"]["k"] == 0
    assert g["rotation"]["pos_success"]["n"] == 1
    assert s["mean_steps_to_success"] == 10.0


def summ(k, pos_k=0, med=50.0):
    return {"success": {"k": k}, "pos_success": {"k": pos_k}, "median_final_pos_px": med}


def test_select_bc_rule():
    assert select_bc(summ(3), summ(4))[0] == "bc_clean"
    assert select_bc(summ(4), summ(3))[0] == "bc"
    assert select_bc(summ(2, 5), summ(2, 6))[0] == "bc_clean"
    assert select_bc(summ(2, 5, 40.0), summ(2, 5, 41.0))[0] == "bc"
    assert select_bc(summ(2, 5, 41.0), summ(2, 5, 40.0))[0] == "bc_clean"
    assert select_bc(summ(2, 5, 40.0), summ(2, 5, 40.0))[0] == "bc"           # full tie -> spec default


def test_select_setting_rule():
    costs = {"default": 5120, "N=64": 2560, "N=256": 10240, "H=5": 2560}
    rows = {"default": summ(10), "N=64": summ(12), "N=256": summ(14), "H=5": summ(13)}
    assert select_setting(rows, costs)["selected"] == "N=256"                 # 14 >= 10 + 3, the most
    rows = {"default": summ(10), "N=64": summ(12), "N=256": summ(13), "H=5": summ(13)}
    assert select_setting(rows, costs)["selected"] == "H=5"                   # tie at 13 -> cheaper
    rows = {"default": summ(10), "N=64": summ(12), "N=256": summ(12), "H=5": summ(11)}
    assert select_setting(rows, costs)["selected"] == "default"               # no margin
    rows = {"default": summ(1, 4), "N=64": summ(2, 9), "N=256": summ(0, 5), "H=5": summ(2, 6)}
    sel = select_setting(rows, costs)                                         # all <= 2 -> position-only
    assert sel["metric"] == "pos_success" and sel["selected"] == "N=64"


def test_success_matrix_and_tau():
    states = np.array([[0, 0, 100, 100, 0.0], [0, 0, 110, 100, 0.1], [0, 0, 300, 300, 0.0],
                       [0, 0, 100, 100, np.pi]], np.float32)
    m = success_matrix(states)
    assert m[0, 1] and not m[0, 2] and not m[0, 3] and m.diagonal().all()
    z = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [5.0, 5.0]])
    tau = calibrate_tau(z, states, np.zeros(4, int), percentile=50)
    assert tau["n_success_pairs"] == 1 and tau["tau"] == pytest.approx(1.0)
    assert tau["median_non_success"] > tau["tau"]

"""Phase 7 registry: no finished job's hash moves; final_eval / eval_imagined_angle wiring; phase7 group."""
import pytest

from wm.registry import expand_group, input_hashes, job_config_hash, load_job_config, load_registry, producer

# config_hash of finished Kaggle runs (results/*.json)
FINISHED = {"plan_val": "1c1cfd1a332a", "plan_sweep": "6453b9964314", "plan_cost_ablation": "26e5d8dc66b3",
            "train_wm": "d4c8e342b0d6", "train_wm_1step": "dd5934291cbe", "train_pixel": "2438d918b486",
            "train_bc": "dd9930d91fd0", "train_probe": "981a0e656148", "collect_data": "4af4e37608e6"}


@pytest.mark.parametrize("job,h", sorted(FINISHED.items()))
def test_finished_hashes_unchanged(job, h):
    reg = load_registry()
    e = reg["jobs"][job]
    assert job_config_hash(job, e, load_job_config(e), reg) == h


def test_phase7_wiring():
    reg = load_registry()
    fe, ia = reg["jobs"]["final_eval"], reg["jobs"]["eval_imagined_angle"]
    assert set(fe["needs"]) == {"pusht-data", "eval-pairs", "ckpt-wm", "ckpt-wm-1step", "ckpt-pixel", "ckpt-bc",
                                "ckpt-bc-clean", "results-plan-val"}
    assert fe["makes"] == ["results-final", "results-bundle"] and fe["config"] == "configs/final_eval.yaml"
    assert set(ia["needs"]) == {"pusht-data", "ckpt-wm", "ckpt-wm-1step", "ckpt-pixel"}
    assert producer("results-imagined-angle", reg)[0] == "eval_imagined_angle"
    # plan.yaml reaches final_eval's hash through results-plan-val
    assert "results-plan-val" in input_hashes(fe, reg)
    assert expand_group("phase7", reg["groups"]) == ["eval_imagined_angle", "final_eval"]


def test_final_eval_config_preregistered():
    cfg = load_job_config(load_registry()["jobs"]["final_eval"])
    assert cfg["seeds"] == [0, 1, 2] and cfg["pixel_seeds"] == [0]
    assert cfg["order"][-1] == "cem_pixel" and cfg["order"].index("sweep") < cfg["order"].index("cem_pixel")
    assert cfg["comparisons"] == [["cem_wm", "bc_selected"], ["cem_wm", "cem_wm_1step"], ["cem_wm", "cem_pixel"]]
    assert cfg["bins"] == {"move_px": [0, 45, 70, 95, None], "turn_deg": [0, 20, 40, 60, 180]}
    ia = load_job_config(load_registry()["jobs"]["eval_imagined_angle"])
    assert ia["search_px"] == 4 and ia["ks"] == [1, 5, 10] and ia["verdict_k"] == 10

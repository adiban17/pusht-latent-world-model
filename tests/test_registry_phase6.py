"""Phase 6 registry: train_bc_clean = train_bc on clean actions; planning jobs' inputs; group order."""
from wm.registry import expand_group, input_hashes, job_config_hash, load_job_config, load_registry
from wm.repro import REPO_ROOT, load_yaml


def test_bc_clean_config_differs_only_in_clean_actions():
    bc = load_yaml(REPO_ROOT / "configs/bc.yaml")
    clean = load_yaml(REPO_ROOT / "configs/bc_clean.yaml")
    assert clean.pop("use_clean_actions") is True and bc.pop("use_clean_actions") is False
    assert bc == clean


def test_train_bc_hash_frozen_and_bc_clean_trains_from_data_only():
    reg = load_registry()
    e = reg["jobs"]["train_bc"]
    assert job_config_hash("train_bc", e, load_job_config(e), reg) == "dd9930d91fd0"
    assert input_hashes(reg["jobs"]["train_bc_clean"], reg) == {}
    assert reg["jobs"]["train_bc_clean"]["makes"] == ["ckpt-bc-clean"]


def test_planning_inputs_and_group():
    reg = load_registry()
    assert set(input_hashes(reg["jobs"]["plan_val"], reg)) == {"eval-pairs", "ckpt-wm", "ckpt-wm-1step",
                                                                "ckpt-bc", "ckpt-bc-clean", "ckpt-pixel"}
    assert set(input_hashes(reg["jobs"]["plan_sweep"], reg)) == {"eval-pairs", "ckpt-wm"}
    assert expand_group("phase6", reg["groups"]) == ["make_pairs", "train_bc_clean", "plan_val", "plan_sweep"]


def test_plan_config_matches_spec():
    cfg = load_yaml(REPO_ROOT / "configs/plan.yaml")
    c = cfg["cem"]
    assert (c["candidates"], c["horizon"], c["rounds"], c["elites"]) == (128, 10, 4, 12)
    assert (c["delta_clip"], c["sigma_init"], c["sigma_floor"], c["beta"]) == (50.0, 30.0, 2.0, 0.1)
    assert cfg["budget"] == 40
    assert set(cfg["methods"]) == {"cem_wm", "cem_wm_1step", "bc", "bc_clean", "random", "noop", "cem_pixel"}
    pairs = load_yaml(REPO_ROOT / "configs/pairs.yaml")
    assert (pairs["n_test"], pairs["n_val"], pairs["offset"], pairs["seed"]) == (200, 50, 20, 0)

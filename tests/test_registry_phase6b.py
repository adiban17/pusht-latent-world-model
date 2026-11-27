"""Phase 6 freeze: the `frozen` flag in plan.yaml never changes a config hash."""
import copy

from wm.registry import job_config_hash, load_job_config, load_registry

# config_hash of the finished Kaggle plan_val / plan_sweep runs (results/plan_val.json, plan_sweep.json)
PLAN_HASHES = {"plan_val": "1c1cfd1a332a", "plan_sweep": "6453b9964314"}
FROZEN_TRAINING = {"train_wm": "d4c8e342b0d6", "train_wm_1step": "dd5934291cbe", "train_pixel": "2438d918b486",
                   "train_bc": "dd9930d91fd0", "train_probe": "981a0e656148", "collect_data": "4af4e37608e6"}


def test_plan_hashes_unchanged_whatever_frozen_says():
    reg = load_registry()
    for job, h in PLAN_HASHES.items():
        e = reg["jobs"][job]
        cfg = load_job_config(e)
        assert cfg["frozen"] is True
        for flag in (True, False):
            c = copy.deepcopy(cfg)
            c["frozen"] = flag
            assert job_config_hash(job, e, c, reg) == h, (job, flag)


def test_other_keys_still_change_the_hash():
    reg = load_registry()
    e = reg["jobs"]["plan_val"]
    cfg = copy.deepcopy(load_job_config(e))
    cfg["budget"] = 41
    assert job_config_hash("plan_val", e, cfg, reg) != PLAN_HASHES["plan_val"]


def test_training_hashes_unchanged():
    reg = load_registry()
    for job, h in FROZEN_TRAINING.items():
        e = reg["jobs"][job]
        assert job_config_hash(job, e, load_job_config(e), reg) == h, job

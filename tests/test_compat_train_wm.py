"""Phase 4 refactor guard: train_wm full is running on Kaggle at commit 5dedc02.

Its finished `ckpt-wm` must stay recognised as complete, and a crashed run must still resume, so
the config_hash, artifact name and checkpoint format of train_wm must not change.
"""
import importlib.util

from wm.repro import REPO_ROOT

spec = importlib.util.spec_from_file_location("run", REPO_ROOT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

TRAIN_WM_HASH = "d2385fb211c8"   # value at commit 5dedc02 (Phase 3), before the Phase 4 refactor


def test_train_wm_config_hash_unchanged():
    reg = run.load_registry()
    entry = reg["jobs"]["train_wm"]
    assert run.job_config_hash("train_wm", entry, run.load_job_config(entry)) == TRAIN_WM_HASH


def test_train_wm_entry_and_artifact_unchanged():
    import wm.jobs.train_wm as job

    entry = run.load_registry()["jobs"]["train_wm"]
    assert entry == {"phase": 3, "module": "wm.jobs.train_wm", "config": "configs/wm.yaml",
                     "needs": ["pusht-data"], "makes": ["ckpt-wm"]}
    assert job.ARTIFACT == "ckpt-wm"

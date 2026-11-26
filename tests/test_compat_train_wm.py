"""Config guard for train_wm: its config_hash decides whether a finished `ckpt-wm` is reused.

Phase 3b (2026-11-27) adopted the A+B recipe (fg_weight + proprio) in configs/wm.yaml, which
deliberately changed the hash from d2385fb211c8 (v1, `ckpt-wm:v5` on Kaggle, commit 5dedc02) to
TRAIN_WM_HASH below, so the next `train_wm` full trains the A+B model from scratch. The v1 recipe
is kept unchanged in configs/wm_v1.yaml.
"""
import importlib.util

import pytest

from wm.repro import REPO_ROOT, load_yaml

spec = importlib.util.spec_from_file_location("run", REPO_ROOT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

TRAIN_WM_HASH = "d4c8e342b0d6"   # A+B recipe (Phase 3b decision); v1 was d2385fb211c8
PHASE3B_FLAGS = {"fg_weight", "fg_alpha", "fg_threshold", "proprio", "keep_best"}


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


def test_wm_v1_yaml_is_the_phase3_recipe():
    """wm_v1.yaml hashes to the v1 train_wm hash and has none of the Phase 3b flags."""
    v1 = load_yaml(REPO_ROOT / "configs" / "wm_v1.yaml")
    assert not PHASE3B_FLAGS & set(v1) and "proprio" not in v1["loss"]
    entry = run.load_registry()["jobs"]["train_wm"]
    assert run.job_config_hash("train_wm", entry, v1) == "d2385fb211c8"


def test_wm_yaml_is_v1_plus_a_and_b():
    """wm.yaml = wm_v1.yaml + fg_weight (alpha 20, 25/255) + proprio (weight 1.0), keep_best off."""
    v1 = load_yaml(REPO_ROOT / "configs" / "wm_v1.yaml")
    wm = load_yaml(REPO_ROOT / "configs" / "wm.yaml")
    assert {k: wm[k] for k in PHASE3B_FLAGS - {"fg_threshold"}} == {
        "fg_weight": True, "fg_alpha": 20.0, "proprio": True, "keep_best": False}
    assert wm["fg_threshold"] == pytest.approx(25 / 255)
    assert wm["loss"] == {**v1["loss"], "proprio": 1.0}
    assert {k: v for k, v in wm.items() if k not in PHASE3B_FLAGS | {"loss"}} == \
           {k: v for k, v in v1.items() if k != "loss"}


def test_wm_yaml_matches_the_ablation_ab_variant():
    """The adopted recipe is exactly the wm_ablate "AB" run, minus keep_best and its schedule."""
    from wm.jobs.wm_ablate import variant_config

    abl = load_yaml(REPO_ROOT / "configs" / "wm_ablate.yaml")
    ab = variant_config(abl, load_yaml(REPO_ROOT / abl["base_config"]), "AB")
    wm = load_yaml(REPO_ROOT / "configs" / "wm.yaml")
    drop = {"train", "smoke", "keep_best"}
    assert {k: v for k, v in ab.items() if k not in drop} == {k: v for k, v in wm.items() if k not in drop}

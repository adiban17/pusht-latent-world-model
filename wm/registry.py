"""The job registry (jobs.yaml): entries, config files, config hashes and groups.

Shared by run.py and by jobs that must find the checkpoint a given config produced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from wm.repro import REPO_ROOT, config_hash, load_yaml


def load_registry(path: Path = REPO_ROOT / "jobs.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_job_config(entry: dict[str, Any]) -> dict[str, Any]:
    """The job's YAML config, or {} if it has none."""
    cfg = entry.get("config")
    if not cfg:
        return {}
    path = REPO_ROOT / cfg
    return load_yaml(path) if path.exists() else {}


# Inputs that never change a consumer's hash: the dataset is frozen since Phase 2, and folding
# it in would change (and retrain) every training job.
FROZEN_INPUTS = ("pusht-data",)


def job_config_hash(name: str, entry: dict[str, Any], cfg: dict[str, Any],
                    registry: dict[str, Any] | None = None) -> str:
    """config_hash = hash of the job name, its jobs.yaml entry and its config file, plus (Phase
    5b) the current config_hashes of the artifacts it needs other than `pusht-data`.

    So a job made from checkpoints (e.g. train_probe_mix from ckpt-wm) gets a new hash when one
    of those checkpoints' configs changes: it retrains or reruns, and `pull_matching` downstream
    never accepts a stale one. Jobs that need only `pusht-data` keep their old hash.
    A top-level `frozen` key (plan.yaml, Phase 6) is hashed as False whatever its value: flipping
    it marks the config as frozen without changing any hash (decision 2026-11-27).
    """
    inputs = input_hashes(entry, registry)
    if "frozen" in cfg:   # status marker, not a setting: hashed as False so freezing never stales results
        cfg = {**cfg, "frozen": False}
    payload: dict[str, Any] = {"job": name, "entry": entry, "config": cfg}
    if inputs:
        payload["inputs"] = inputs
    return config_hash(payload)


def expand_group(name: str, groups: dict[str, Any]) -> list[str]:
    """The jobs of a group in order; a member that is itself a group is expanded in place."""
    out: list[str] = []
    for member in groups[name]["jobs"]:
        out += expand_group(member, groups) if member in groups else [member]
    return out


def input_hashes(entry: dict[str, Any], registry: dict[str, Any] | None = None) -> dict[str, str]:
    """{artifact: producer's current config_hash} for every artifact the job needs except
    `pusht-data` (FROZEN_INPUTS): checkpoints, and judge-data (Phase 5b).

    An eval job stores this in its results metadata (`inputs`); run.is_finished treats the
    results as stale if a needed artifact's producer config changed since (Phase 5). Since
    Phase 5b the same hashes are folded into the job's own config_hash.
    """
    needs = [a for a in (entry.get("needs") or []) if a not in FROZEN_INPUTS]
    if not needs:
        return {}
    reg = registry or load_registry()
    return {a: producer(a, reg)[1] for a in needs}


def producer(artifact: str, registry: dict[str, Any] | None = None) -> tuple[str, str]:
    """(job name, current config_hash) of the job in jobs.yaml that makes `artifact`."""
    reg = registry or load_registry()
    for name, entry in reg["jobs"].items():
        if artifact in (entry.get("makes") or []):
            return name, job_config_hash(name, entry, load_job_config(entry), reg)
    raise KeyError(f"no job in jobs.yaml makes '{artifact}'")

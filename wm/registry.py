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


def job_config_hash(name: str, entry: dict[str, Any], cfg: dict[str, Any]) -> str:
    """config_hash = hash of the job name, its jobs.yaml entry and its config file."""
    return config_hash({"job": name, "entry": entry, "config": cfg})


def expand_group(name: str, groups: dict[str, Any]) -> list[str]:
    """The jobs of a group in order; a member that is itself a group is expanded in place."""
    out: list[str] = []
    for member in groups[name]["jobs"]:
        out += expand_group(member, groups) if member in groups else [member]
    return out


def input_hashes(entry: dict[str, Any], registry: dict[str, Any] | None = None) -> dict[str, str]:
    """{artifact: producer's current config_hash} for every `ckpt-*` the job needs.

    An eval job stores this in its results metadata (`inputs`); run.is_finished treats the
    results as stale if a needed checkpoint's producer config changed since (Phase 5).
    """
    reg = registry or load_registry()
    return {a: producer(a, reg)[1] for a in (entry.get("needs") or []) if a.startswith("ckpt-")}


def producer(artifact: str, registry: dict[str, Any] | None = None) -> tuple[str, str]:
    """(job name, current config_hash) of the job in jobs.yaml that makes `artifact`."""
    reg = registry or load_registry()
    for name, entry in reg["jobs"].items():
        if artifact in (entry.get("makes") or []):
            return name, job_config_hash(name, entry, load_job_config(entry))
    raise KeyError(f"no job in jobs.yaml makes '{artifact}'")

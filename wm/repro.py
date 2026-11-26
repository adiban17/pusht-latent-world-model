"""Reproducibility helpers: seeding, git hash, config loading and hashing."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def seed_all(seed: int) -> None:
    """Seed python, numpy and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_info(repo: Path = REPO_ROOT) -> str:
    """Short git hash, with '-dirty' if there are uncommitted changes; 'nogit' if unavailable."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return f"{sha}-dirty" if dirty else sha


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (empty dict for an empty file)."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def config_hash(obj: Any) -> str:
    """Stable 12-char hash of any JSON-serialisable object (dict keys sorted)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]

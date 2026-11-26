"""Artifact store: a local folder on the Mac, W&B Artifacts on Kaggle/Colab.

Naming rules (CLAUDE.md §10):
- smoke mode writes `<name>-smoke`;
- smoke mode reads `<name>` if it exists, else `<name>-smoke`;
- full mode reads and writes `<name>` only.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from wm.repro import git_info

WANDB_PROJECT = "pusht-world-model"
REQUIRED_META = ("complete", "config_hash")


def write_name(name: str, mode: str) -> str:
    """Artifact name this mode writes to."""
    return f"{name}-smoke" if mode == "smoke" else name


def read_candidates(name: str, mode: str) -> list[str]:
    """Names to try, in order, when reading in this mode."""
    return [name, f"{name}-smoke"] if mode == "smoke" else [name]


class LocalStore:
    """`<root>/<name>/files/...` plus `<root>/<name>/meta.json`."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("WM_STORE_DIR", "./store"))

    def push(self, name: str, path: str | Path, metadata: dict[str, Any]) -> str:
        src = Path(path)
        dest = self.root / name
        if dest.exists():
            shutil.rmtree(dest)
        files = dest / "files"
        if src.is_dir():
            shutil.copytree(src, files)
        else:
            files.mkdir(parents=True)
            shutil.copy2(src, files / src.name)
        (dest / "meta.json").write_text(json.dumps(metadata, indent=2, default=str))
        return str(dest)

    def pull(self, name: str) -> Path | None:
        files = self.root / name / "files"
        return files if files.exists() else None

    def exists(self, name: str) -> bool:
        return (self.root / name / "meta.json").exists()

    def metadata(self, name: str) -> dict[str, Any] | None:
        meta = self.root / name / "meta.json"
        return json.loads(meta.read_text()) if meta.exists() else None


class WandbStore:
    """W&B Artifacts in project `pusht-world-model`. Pushing needs an active wandb run."""

    def __init__(self, run=None, cache_dir: str | Path = "./store"):
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY is not set. On Kaggle: Add-ons -> Secrets -> add WANDB_API_KEY "
                "and attach it to this notebook. On Colab: the key icon -> WANDB_API_KEY."
            )
        import wandb

        self.wandb = wandb
        self.run = run
        self.api = wandb.Api()
        self.cache_dir = Path(cache_dir)
        entity = run.entity if run is not None else self.api.default_entity
        self.prefix = f"{entity}/{WANDB_PROJECT}"

    def _get(self, name: str):
        try:
            return self.api.artifact(f"{self.prefix}/{name}:latest")
        except Exception:  # wandb raises CommError / ValueError for missing artifacts
            return None

    def push(self, name: str, path: str | Path, metadata: dict[str, Any]) -> str:
        if self.run is None:
            raise RuntimeError("WandbStore.push needs an active wandb run")
        src = Path(path)
        art = self.wandb.Artifact(name, type="wm", metadata=metadata)
        if src.is_dir():
            art.add_dir(str(src))
        else:
            art.add_file(str(src))
        logged = self.run.log_artifact(art)
        logged.wait()  # so a pull straight after sees this version
        return f"{name}:{logged.version}"

    def pull(self, name: str) -> Path | None:
        art = self._get(name)
        if art is None:
            return None
        return Path(art.download(root=str(self.cache_dir / name / "files")))

    def exists(self, name: str) -> bool:
        return self._get(name) is not None

    def metadata(self, name: str) -> dict[str, Any] | None:
        art = self._get(name)
        return dict(art.metadata) if art is not None else None


class Store:
    """Mode-aware front end over a backend (LocalStore or WandbStore)."""

    def __init__(self, backend, mode: str):
        self.backend = backend
        self.mode = mode

    def push(self, name: str, path: str | Path, metadata: dict[str, Any]) -> str:
        """Push under this mode's name. Metadata must contain `complete` and `config_hash`."""
        missing = [k for k in REQUIRED_META if k not in metadata]
        if missing:
            raise KeyError(f"store metadata missing {missing}")
        meta = {**metadata, "git": metadata.get("git", git_info())}
        return self.backend.push(write_name(name, self.mode), path, meta)

    def resolve(self, name: str) -> str | None:
        """The artifact name a read in this mode would use, or None."""
        for cand in read_candidates(name, self.mode):
            if self.backend.exists(cand):
                return cand
        return None

    def pull(self, name: str) -> Path | None:
        resolved = self.resolve(name)
        return self.backend.pull(resolved) if resolved else None

    def exists(self, name: str) -> bool:
        return self.resolve(name) is not None

    def metadata(self, name: str) -> dict[str, Any] | None:
        resolved = self.resolve(name)
        return self.backend.metadata(resolved) if resolved else None

    def pull_own(self, name: str) -> Path | None:
        """Pull what *this mode* writes (used to resume a training run from its own checkpoint)."""
        return self.backend.pull(write_name(name, self.mode))

    def own_metadata(self, name: str) -> dict[str, Any] | None:
        """Metadata of what *this mode* would write (used for skip/resume)."""
        return self.backend.metadata(write_name(name, self.mode))


def make_store(platform: str, mode: str, run=None) -> Store:
    """Local folder on the Mac; W&B Artifacts on Kaggle/Colab."""
    backend = LocalStore() if platform == "local" else WandbStore(run=run)
    return Store(backend, mode)

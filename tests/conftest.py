"""Shared test setup: headless pygame, local platform, and temp output/store dirs."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Every test writes outputs and store artifacts under its own tmp dir, as 'local'."""
    monkeypatch.setenv("WM_OUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("WM_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    return tmp_path

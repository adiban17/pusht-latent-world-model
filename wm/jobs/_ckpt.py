"""Loading a trained checkpoint (`ckpt-wm`, `ckpt-wm-1step`, ...) for an eval job.

Rule (fix after the Kaggle overnight smoke, 2026-11-27): an eval job uses only the checkpoint
whose config_hash equals what the producing job's *current* config would give. Otherwise, in
smoke mode, the store's "read `<name>` before `<name>-smoke`" rule silently picks an older
recipe (e.g. `ckpt-wm:v5`, the v1 model, after wm.yaml switched to A+B).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from wm.jobs import JobContext
from wm.registry import producer


def pull_matching(ctx: JobContext, artifact: str) -> tuple[Path, str, dict[str, Any], str]:
    """(local path, artifact name, metadata, expected config_hash) of the complete checkpoint
    made with the current config. Raises RuntimeError naming what was found otherwise."""
    job, expected = producer(artifact)
    name = ctx.store.resolve_matching(artifact, expected)
    if name is None:
        found = "; ".join(
            f"{cand}: " + (f"config_hash {m.get('config_hash')}, complete {m.get('complete')}, "
                           f"step {m.get('step')}" if m else "missing")
            for cand, m in ctx.store.candidates(artifact))
        raise RuntimeError(f"no complete '{artifact}' with config_hash {expected} (the current "
                           f"configs of job {job}). Found: {found}. Fix: run {job} first.")
    path = ctx.store.backend.pull(name)
    if path is None:
        raise RuntimeError(f"'{name}' has matching metadata but could not be pulled")
    return path, name, ctx.store.backend.metadata(name) or {}, expected


def _loader(artifact: str):
    """(load(model_pt_path, device) -> (model, blob), param_counts) for a checkpoint kind."""
    import torch

    def plain(build):
        def load(p: Path, device):
            blob = torch.load(p, map_location=device, weights_only=True)
            model = build(blob["config"]).to(device)
            model.load_state_dict(blob["model"])
            return model.eval(), blob
        return load

    if artifact in ("ckpt-wm", "ckpt-wm-1step"):
        from wm.models.world_model import load_world_model, param_counts
        return load_world_model, param_counts
    if artifact == "ckpt-pixel":
        from wm.models.pixel_unet import build_pixel_unet, param_counts
        return plain(build_pixel_unet), param_counts
    if artifact in ("ckpt-probe", "ckpt-probe-mix"):
        from wm.models.probe import build_probe, param_counts
        return plain(build_probe), param_counts
    if artifact in ("ckpt-bc", "ckpt-bc-clean"):
        from wm.models.bc import build_bc, param_counts
        return plain(build_bc), param_counts
    if artifact == "ckpt-latent-probe":
        from wm.models.latent_probe import build_latent_probe, param_counts
        return plain(build_latent_probe), param_counts
    raise KeyError(f"no loader for '{artifact}'")


def load_matching(ctx: JobContext, artifact: str) -> tuple[Any, dict[str, Any]]:
    """(model in eval mode, info) of the hash-matched checkpoint `artifact` (see pull_matching).

    info = artifact name, step, git, config_hash (in model.pt), expected and store-metadata
    hashes, params, proprio / teacher_forcing (world models) and `summary` (train_summary.json,
    if the artifact has one: judges keep their acceptance results there)."""
    import json

    path, name, meta, expected = pull_matching(ctx, artifact)
    if not (path / "model.pt").exists():
        raise RuntimeError(f"'{name}' has no model.pt")
    load, counts = _loader(artifact)
    model, blob = load(path / "model.pt", ctx.device)
    summary_path = path / "train_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    info = {"artifact": name, "step": blob.get("step"), "git": blob.get("git"),
            "config_hash": blob.get("config_hash"), "expected_config_hash": expected,
            "meta_config_hash": meta.get("config_hash"), "params": counts(model),
            "proprio": getattr(model, "proprio", None),
            "teacher_forcing": getattr(model, "teacher_forcing", None),
            "summary": {k: v for k, v in summary.items() if k not in ("train_log", "val_log")}}
    return model, info

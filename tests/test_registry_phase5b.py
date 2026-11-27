"""Phase 5b registry: no existing checkpoint retrains; judge/eval hashes follow their inputs."""
import copy

from wm import registry
from wm.registry import expand_group, input_hashes, job_config_hash, load_job_config, load_registry, producer

# Training config_hashes of the existing checkpoints (Kaggle full runs). Phase 5b must not change them.
FROZEN = {"train_wm": "d4c8e342b0d6", "train_wm_1step": "dd5934291cbe", "train_pixel": "2438d918b486",
          "train_bc": "dd9930d91fd0", "train_probe": "981a0e656148", "collect_data": "4af4e37608e6",
          "wm_ablate": "76b1611426f5"}


def test_existing_training_hashes_unchanged():
    reg = load_registry()
    for job, h in FROZEN.items():
        e = reg["jobs"][job]
        assert job_config_hash(job, e, load_job_config(e), reg) == h, job
        assert input_hashes(e, reg) == {}


def test_judge_hash_follows_a_needed_checkpoint_config(monkeypatch):
    reg = load_registry()
    before = {a: producer(a, reg)[1] for a in ("ckpt-probe-mix", "ckpt-latent-probe", "ckpt-wm")}
    real_load = registry.load_job_config

    def changed(entry):
        cfg = real_load(entry)
        if entry.get("config") == "configs/wm.yaml":
            cfg = copy.deepcopy(cfg)
            cfg["seed"] = 123
        return cfg

    monkeypatch.setattr(registry, "load_job_config", changed)
    after = {a: producer(a, reg)[1] for a in before}
    assert all(after[a] != before[a] for a in before)
    e = reg["jobs"]["train_probe"]
    assert job_config_hash("train_probe", e, real_load(e), reg) == FROZEN["train_probe"]


def test_judge_inputs_and_group():
    reg = load_registry()
    assert set(input_hashes(reg["jobs"]["train_probe_mix"], reg)) == {"judge-data", "ckpt-wm", "ckpt-wm-1step",
                                                                       "ckpt-pixel"}
    assert set(input_hashes(reg["jobs"]["train_latent_probe"], reg)) == {"judge-data", "ckpt-wm"}
    assert expand_group("phase5b", reg["groups"]) == ["angle_audit", "render_judge_data", "train_probe_mix",
                                                      "train_latent_probe", "eval_rollouts"]

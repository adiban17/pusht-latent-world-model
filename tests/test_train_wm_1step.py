import importlib.util

import torch

from wm.data.collect import collect_split
from wm.data.windows import WindowData
from wm.models.world_model import build_world_model
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.train.train_wm import validate

WM = load_yaml(REPO_ROOT / "configs/wm.yaml")
ONE = load_yaml(REPO_ROOT / "configs/wm_1step.yaml")

spec = importlib.util.spec_from_file_location("run", REPO_ROOT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_config_differs_only_in_teacher_forcing():
    assert WM["teacher_forcing"] is False and ONE["teacher_forcing"] is True
    assert {k: v for k, v in WM.items() if k != "teacher_forcing"} == \
           {k: v for k, v in ONE.items() if k != "teacher_forcing"}
    assert ONE["loss"]["var"] == 1.0                         # keeps the variance hinge


def test_config_hash_differs_from_train_wm():
    reg = run.load_registry()["jobs"]
    h = {j: run.job_config_hash(j, reg[j], run.load_job_config(reg[j])) for j in ("train_wm", "train_wm_1step")}
    assert h["train_wm"] != h["train_wm_1step"]
    assert reg["train_wm_1step"]["makes"] == ["ckpt-wm-1step"]


def test_validation_is_open_loop_even_when_teacher_forced():
    arrays, _ = collect_split([0], load_yaml(REPO_ROOT / "configs/data.yaml")["expert"], steps=20)
    data = WindowData(arrays, "cpu", WM["val"]["seeds"])
    starts = data.fixed_val_starts(3, 0)
    seed_all(0)
    m = build_world_model(WM)
    w = dict(WM["loss"])
    open_loop = validate(m, data, starts, torch.device("cpu"), w, amp=False)
    m.teacher_forcing = True
    tf = validate(m, data, starts, torch.device("cpu"), w, amp=False)
    assert tf["nmse"] == open_loop["nmse"]                   # same weights -> same open-loop NMSE
    assert tf["lat"] != open_loop["lat"]                     # but the training-objective terms differ

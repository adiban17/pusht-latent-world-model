"""Local check (Phase 3b): proprio head init. A vs AB (random p_out init) vs AB (zero init),
200 steps, batch 8, on the local smoke data; prints (step, val z_std, val NMSE@10, val pusher px).
Not a result: it only motivated zero-initialising p_out. Run: .venv/bin/python scripts/check_proprio_init.py
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from wm.data.windows import WindowData
from wm.jobs.wm_ablate import variant_config
from wm.models.world_model import build_world_model
from wm.repro import REPO_ROOT, load_yaml, seed_all
from wm.train.common import resolve_config
from wm.train.train_wm import prepare_model, run_training
abl = load_yaml(REPO_ROOT / "configs/wm_ablate.yaml"); base = load_yaml(REPO_ROOT / "configs/wm_v1.yaml")
with np.load("store/pusht-data-smoke/files/pusht_train.npz") as d: arrays = {k: d[k] for k in d.files}
data = WindowData(arrays, "cpu", base["val"]["seeds"])
for name, zero in [("A", False), ("AB", False), ("AB", True)]:
    cfg = variant_config(abl, base, name); t = resolve_config(cfg, "smoke", "local")
    t.update(steps=200, val_every=50, batch_size=8)
    seed_all(0); m = prepare_model(build_world_model(cfg), data)
    if name == "AB" and not zero:          # the model zero-inits p_out; restore nn.Linear's default
        m.dynamics.p_out.reset_parameters()
    r = run_training(m, data, t, torch.device("cpu"), dict(cfg["loss"]))
    print(name, "zero-init" if zero else ("random-init" if name == "AB" else ""), [(v["step"], round(v["z_std"], 3), round(v["nmse"][-1], 3), round(v.get("pusher_px", [0])[-1], 1)) for v in r["val_log"]], flush=True)

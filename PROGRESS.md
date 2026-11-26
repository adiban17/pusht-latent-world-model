# PROGRESS

**Current phase: 1 — Foundation (done locally; Kaggle env_check pending).**

## Status
| Phase | Status | Local smoke | Kaggle smoke | Kaggle full |
| --- | --- | --- | --- | --- |
| 1 Foundation | done | PASS | pending | n/a |
| 2 Expert + data | not started | – | – | – |
| 3 World model | not started | – | – | – |
| 4 Baselines | not started | – | – | – |
| 5 Rollout eval | not started | – | – | – |
| 6 Planner | not started | – | – | – |
| 7 Final eval | not started | – | – | – |
| 8 Report | not started | – | – | – |

## Decisions
- **Python (Phase 1):** `.venv` made with uv on Python 3.12 (the system `python3` is already 3.12). Every command uses `.venv/bin/python`.
- **`--mode` defaults to `smoke`** (so `run.py --job make_report` works). `--mode full` is refused on local for every job except `make_report`.
- **config_hash** = hash of the job's `jobs.yaml` entry + its config file (`{}` if it has none).
- **`wm/repro.py`** added (not in the original layout): `seed_all`, `git_info`, `config_hash`, `load_yaml`.
- **W&B:** on Kaggle/Colab every job opens one wandb run (project `pusht-world-model`); its URL goes on the REPORT `W&B:` line. Local = `off`.
- **State vector** = `[pusher x, pusher y, block x, block y, block angle]`, px and **radians** (wrapped to [−π, π)), read from `env.unwrapped.agent.position`, `.block.position`, `.block.angle`.
- **reset_to_state** uses gym-pusht's native `reset(options={"reset_to_state": ...})`, but pre-compensates the block position. gym-pusht sets position before angle and pymunk rotates about the T's centre of gravity (local (0, 45)), which otherwise moves the block by up to 90 px. Round trip is exact (0.0000 px / 0.0000°; 200/200 random states with a non-overlapping pusher). Tests use non-overlapping states, because gym-pusht runs one physics step after placing the bodies.
- **OpenCV pinned `opencv-python-headless<5`:** gym-pusht 0.1.8 requires `opencv-python<5`, and both share the `cv2` folder, so they stay on the same major (both 4.14.0.94).
- **§5 pixel U-Net check: within ±20%** of 4,184,899 (was ±15%; the spec'd U-Net has 4,813,731 params, just over +15%).
- **§3 frames:** 101 frames per episode (100 actions + final frame), giving exactly 20,000 train / 3,000 test transitions.
- **§3 termination:** never end an episode early; step all 100 actions even if `terminated`. (gym-pusht 0.1.8's `step()` does keep working after termination.) If it ever refuses, replace that episode with an extra train seed from 2000 upwards. Validation is always seeds 190–199; test is always 1000–1029.

## Installed versions (local)
python 3.12.2, torch 2.15.0, gym-pusht 0.1.8, gymnasium 1.4.0, pymunk 6.11.1, pygame 2.6.1, opencv-python(-headless) 4.14.0.94, numpy 2.5.5, wandb 0.31.2.

## Open issues
- **Mac only:** on import, macOS prints `objc ... Class SDL... is implemented in both cv2/.dylibs/libSDL2 and pygame/.dylibs/libSDL2` to stderr, because both OpenCV wheels bundle SDL2 on macOS. It's harmless with `SDL_VIDEODRIVER=dummy` and all tests pass. It doesn't occur on Kaggle/Colab (Linux).
- Kaggle `env_check` not run yet. It needs: a **public** GitHub repo (pushed), the notebook's Internet on, GPU T4 accelerator, and the `WANDB_API_KEY` secret attached.

## Last REPORT (local smoke)
```
================ PASTE THIS TO CLAUDE ================
job: env_check   mode: smoke   platform: local   device: cpu
git: c305e26-dirty   time: 0.6s   params: n/a
CHECKS
 [PASS] platform: local (local|kaggle|colab)
 [PASS] python_version: 3.12.2 (>= 3.10)
 [PASS] device: cpu (cpu on local; never mps)
 [PASS] render64: (64, 64, 3) uint8 ((64, 64, 3) uint8)
 [PASS] reset_roundtrip: 0.0000 px, 0.0000 deg (< 1.0 px, < 0.5 deg)
 [PASS] store_roundtrip: bytes match (push -> pull)
METRICS
 python = 3.12.2
 torch = 2.15.0
 cuda_available = False
 gpu = none
 gym-pusht = 0.1.8
 gymnasium = 1.4.0
 pymunk = 6.11.1
 pygame = 2.6.1
 opencv-python-headless = 4.14.0.94
 opencv-python = 4.14.0.94
 numpy = 2.5.5
 wandb = 0.31.2
 matmul_2048_ms = 23.0
SAVED
 outputs/env_check/frame.png
 store/env-check-smoke
W&B: off
======================================================
```

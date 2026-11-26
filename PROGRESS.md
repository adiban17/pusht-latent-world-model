# PROGRESS

**Current phase: 2 — Expert + data (done locally; Aditya to confirm filmstrips, then Kaggle smoke + full).**

## Status
| Phase | Status | Local smoke | Kaggle smoke | Kaggle full |
| --- | --- | --- | --- | --- |
| 1 Foundation | done | PASS | PASS | n/a |
| 2 Expert + data | done locally | PASS | pending | pending |
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
- **Kaggle env_check (Phase 1):** smoke PASSED on Kaggle (Tesla T4, commit f85bb8b), artifact `env-check-smoke:v0`.
- **Block position = centre of gravity (Phase 2, pre-registered before any results):** `pose_error`, `is_success`, the expert's b and the data stats (contact ratio, 8×8 coverage) use `block_cog(state) = (block_x, block_y) + R(angle)·(0, 45)`. The stored state's (x, y) is the T's body origin (middle of the bar's top edge, ~45 px from the centre) and is unchanged. Recorded in CLAUDE.md §1. Test: rotating the T in place gives ~0 px error.
- **`episode_seeds` (Phase 2):** int64 [E] array added to both npz files; val episodes are the train episodes with seeds 190–199.
- **Expert detour detail (Phase 2):** the detour side is fixed when the detour starts; once the detour point is reached (≤ 15 px) the pusher heads to c without re-checking the far-side test until it is back on the near side (otherwise it would re-trigger the detour at the detour point and stall).
- **Expert tuning (Phase 2):** none needed. The §2 numbers as written give contact ratio **48.4%** on 20 episodes (seeds 0–19, `scripts/scratch_tune_expert.py`), 8×8 coverage 70%, angle std 114°; filmstrips show the T being pushed. `configs/data.yaml` keeps the spec values.
- **`scripts/scratch_tune_expert.py`** is a tool, not part of the pipeline (no npz, no store).
- **§3 termination:** never end an episode early; step all 100 actions even if `terminated`. (gym-pusht 0.1.8's `step()` does keep working after termination.) If it ever refuses, replace that episode with an extra train seed from 2000 upwards. Validation is always seeds 190–199; test is always 1000–1029.

## Installed versions (local)
python 3.12.2, torch 2.15.0, gym-pusht 0.1.8, gymnasium 1.4.0, pymunk 6.11.1, pygame 2.6.1, opencv-python(-headless) 4.14.0.94, numpy 2.5.5, wandb 0.31.2.

## Open issues
- **Mac only:** on import, macOS prints `objc ... Class SDL... is implemented in both cv2/.dylibs/libSDL2 and pygame/.dylibs/libSDL2` to stderr, because both OpenCV wheels bundle SDL2 on macOS. It's harmless with `SDL_VIDEODRIVER=dummy` and all tests pass. It doesn't occur on Kaggle/Colab (Linux).
- Phase 2 acceptance: Aditya to confirm the pusher pushes the T in ≥ 2 of the 3 smoke filmstrips (`outputs/collect_data/filmstrip_*.png`).
- Kaggle `collect_data` smoke and full not run yet.

## Last REPORT (local smoke)
```
================ PASTE THIS TO CLAUDE ================
job: collect_data   mode: smoke   platform: local   device: cpu
git: f85bb8b-dirty   time: 2.1s   params: n/a
CHECKS
 [PASS] train_transitions: 200 (== 2 episodes x 100)
 [PASS] test_transitions: 100 (== 1 episodes x 100)
 [PASS] frames_per_episode: 101..101 (all == 101, no early end)
 [PASS] actions_in_bounds: [0.0, 512.0] ([0, 512])
 [PASS] contact_ratio: 52.5% (>= 35% PASS, 25-35% WARN, < 25% FAIL)
METRICS
 train_episodes = 2
 val_episodes = 0
 test_episodes = 1
 contact_ratio_train = 0.525
 contact_ratio_test = 0.500
 coverage_8x8 = 0.156
 block_angle_std_deg = 67.0
 block_step_px_mean = 3.20
 action_mean = [290.32, 283.18]
 action_std = [106.0, 95.43]
 action_clip_frac = 0.010
 noisy_clean_gap_px = 31.3
 env_success_episodes = 0
 replaced_episodes = 0
 sec_per_episode = 0.14
 projected_full_time = 33.0s
SAVED
 outputs/collect_data/pusht_train.npz
 outputs/collect_data/pusht_test.npz
 outputs/collect_data/data_health.json
 outputs/collect_data/histograms.png
 outputs/collect_data/episode_train_seed0000.gif
 outputs/collect_data/filmstrip_train_seed0000.png
 outputs/collect_data/episode_train_seed0001.gif
 outputs/collect_data/filmstrip_train_seed0001.png
 outputs/collect_data/episode_test_seed1000.gif
 outputs/collect_data/filmstrip_test_seed1000.png
 store/pusht-data-smoke
W&B: off
======================================================
```

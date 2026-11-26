# PROGRESS

**Current phase: 3 — World model (done locally; Aditya to run Kaggle `train_wm` smoke, then full).**

## Status
| Phase | Status | Local smoke | Kaggle smoke | Kaggle full |
| --- | --- | --- | --- | --- |
| 1 Foundation | done | PASS | PASS | n/a |
| 2 Expert + data | **complete** | PASS | PASS | PASS |
| 3 World model | done locally | PASS | pending | pending |
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
- **Phase 2 complete:** Kaggle `collect_data` full PASSED at commit f433e2b: 20,000 train / 3,000 test transitions, 10 val episodes, contact ratio 49.8% train / 48.2% test, coverage 0.984, artifact `pusht-data:v0`. Filmstrip gate passed (3 of 3 show the T being pushed).
- **Block angle std (fix commit ecb977d):** the full-data `block_angle_std_deg` = 154° is not a bug. Stored angles were already wrapped, and the circular std √(−2 ln R) is unbounded (R ≈ 0.027, near-uniform orientations, gives 154°). Helpers `wrap_angles_rad` / `circular_std_deg` now wrap explicitly everywhere, `block_angle_resultant_R` is reported too, and there are tests with an angle of 3π. Models only ever see the angle as (sin, cos).
- **REPORT header** is now `RUN REPORT` (was "PASTE THIS TO CLAUDE").
- **Latent collapse fix (Phase 3, 2026-11-26) — chosen: (b) λ_var = 1.0 variance hinge.** The §4 model as written collapses (z_std ≈ 0.003, far under the 0.1 alarm): every frame encodes to nearly the same latent. Local CPU comparison: 22 expert episodes (train seeds 0–19, val 190–191), 400 steps, batch 16, seed 0, 64 fixed val windows; architecture and param counts unchanged. NMSE is normalised by each variant's own latent spread. Source: `results/ablation_collapse_local.json` (made by `scripts/compare_collapse_fixes.py --export`).

  | variant | z_std | rec | img | NMSE@1 (copy-last) | NMSE@10 (copy-last) |
  | --- | --- | --- | --- | --- | --- |
  | base (§4 as written) | 0.0036 | 0.0044 | 0.0044 | 1.183 (0.067) | 17.424 (0.537) |
  | (a) subtract train mean frame | 0.0027 | 0.0044 | 0.0044 | 1.930 (0.108) | 28.038 (0.779) |
  | **(b) variance hinge, λ_var = 1.0** | **0.8232** | **0.0035** | **0.0037** | 0.009 (0.009) | **0.383 (0.408)** |
  | (c) detach dynamics inputs | 0.0022 | 0.0043 | 0.0043 | 3.433 (0.007) | 30.854 (0.254) |
  | (d) b + c | 0.8333 | 0.0033 | 0.0035 | 0.011 (0.011) | 0.486 (0.523) |

  Loss term: `λ_var · mean_d relu(1 − sqrt(var_batch(z_d) + 1e-4))` over all 13·B latents in the batch. Recorded in CLAUDE.md §4 and `configs/wm.yaml` (`loss.var: 1.0`, `var_target: 1.0`). Variants (a) and (c) stay in the code as off-by-default switches (`input_mean_frame`, `detach_dynamics_inputs`). Caveat: at 400 CPU steps (b) only matches copy-last at k = 1 and 5.
- **Decoder (Phase 3):** SiLU after the Linear 256→4096 as well as between the transposed convs (none at the end); param counts unaffected.
- **Smoke settings (Phase 3):** smoke scales warm-up and val/checkpoint intervals down (local: 50 steps, batch 4, warm-up 5, val/ckpt every 25, 16 val windows; Kaggle: 300 steps, batch 64, warm-up 30, val every 100, ckpt every 150, 256 val windows). Local smoke data has no val episodes, so smoke validates on the last train episode (REPORT says so).
- **NMSE reference (Phase 3):** in `train_wm` validation, z̄ and the spread are taken over all frames of the val episodes.
- **`ckpt-wm` layout (Phase 3):** intermediate pushes hold `checkpoint.pt` (model, optimiser, scaler, sampler RNG, logs, config, git, params, wall clock) with `complete: false, step`; a rerun with the same config_hash resumes from it (bit-identical on CPU, tested). The final push holds `model.pt` (weights + config + params + git), `train_summary.json`, `config.yaml`, `rollout_filmstrip.png`, `complete: true`. `Store.pull_own` added so smoke resumes from its own `-smoke` checkpoint.
- **Resume verified (Phase 3):** local smoke SIGKILLed right after its step-25 checkpoint; the rerun resumed at step 25 and finished with all CHECKS PASS.
- **§3 termination:** never end an episode early; step all 100 actions even if `terminated`. (gym-pusht 0.1.8's `step()` does keep working after termination.) If it ever refuses, replace that episode with an extra train seed from 2000 upwards. Validation is always seeds 190–199; test is always 1000–1029.

## Installed versions (local)
python 3.12.2, torch 2.15.0, gym-pusht 0.1.8, gymnasium 1.4.0, pymunk 6.11.1, pygame 2.6.1, opencv-python(-headless) 4.14.0.94, numpy 2.5.5, wandb 0.31.2.

## Open issues
- The local smoke filmstrip's decoded row is still a blur (50 steps, batch 4, CPU); judge the decoder on the Kaggle full run.
- Local `projected_30k_steps` (2h07m) is CPU at batch 4; the real projection comes from the Kaggle smoke.
- **Mac only:** on import, macOS prints `objc ... Class SDL... is implemented in both cv2/.dylibs/libSDL2 and pygame/.dylibs/libSDL2` to stderr, because both OpenCV wheels bundle SDL2 on macOS. It's harmless with `SDL_VIDEODRIVER=dummy` and all tests pass. It doesn't occur on Kaggle/Colab (Linux).

## Last REPORT (local smoke, train_wm, after kill + resume)
```
==================== RUN REPORT ====================
job: train_wm   mode: smoke   platform: local   device: cpu
git: 33e7fa9-dirty   time: 10.2s   params: 4,184,899
CHECKS
 [PASS] param_counts: encoder 1,738,976 / decoder 1,742,563 / dynamics 703,360 / total 4,184,899 / planner 2,442,336 (== CLAUDE.md §4)
 [PASS] losses_finite: all finite (no NaN/inf)
 [PASS] loss_fell: 1.2168 -> 0.2532 (mean of last 10 < first 10 steps)
 [PASS] z_std: 0.808 (> 0.1, collapse alarm)
 [PASS] checkpoint_roundtrip: identical latents (model.pt reload == trained model)
 [PASS] nmse_k10_vs_copy_last: 0.434 vs 0.477 (model < copy-last; WARN only in smoke)
METRICS
 data = pusht-data-smoke
 steps = 50 (batch 4)
 resumed_from_step = 25
 train_windows = 178
 val_source = train episode seed 1 (smoke: no val episodes)
 val_windows = 16
 loss_first_last = 1.2168 -> 0.2532
 rec_first_last = 0.1366 -> 0.0208
 lat_first_last = 0.1305 -> 0.0945
 img_first_last = 0.1364 -> 0.0205
 it_per_s = 3.94
 projected_30k_steps = 2h07m01s
 peak_gpu_mem = n/a (cpu)
 val_nmse_k1 = 0.011 (copy-last 0.010)
 val_nmse_k5 = 0.174 (copy-last 0.171)
 val_nmse_k10 = 0.434 (copy-last 0.477)
 val_z_std = 0.808
 val_loss = 0.2627
 lr_final = 3.65e-07
SAVED
 outputs/train_wm/artifact/model.pt
 outputs/train_wm/artifact/train_summary.json
 outputs/train_wm/artifact/config.yaml
 outputs/train_wm/artifact/rollout_filmstrip.png
 store/ckpt-wm-smoke
W&B: off
====================================================
```

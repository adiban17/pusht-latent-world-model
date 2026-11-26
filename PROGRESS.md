# PROGRESS

**Current phase: 3b — world-model fixes (done locally; Aditya to push, then run Kaggle `wm_ablate` smoke, then full).** Phase 3 `train_wm` full finished on Kaggle (`ckpt-wm:v5`); Phase 4 `train_baselines` on Kaggle waits until the 3b winner is chosen.

## Status
| Phase | Status | Local smoke | Kaggle smoke | Kaggle full |
| --- | --- | --- | --- | --- |
| 1 Foundation | done | PASS | PASS | n/a |
| 2 Expert + data | **complete** | PASS | PASS | PASS |
| 3 World model | Kaggle full done (v1) | PASS | pending | PASS |
| 3b WM fixes | done locally | PASS | pending | pending |
| 4 Baselines | done locally | PASS | pending | pending |
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
- **Phase 4 refactor guard:** `train_wm` full was running on Kaggle (commit 5dedc02) during Phase 4, so its config_hash (`d2385fb211c8`), jobs.yaml entry, artifact name `ckpt-wm` and checkpoint format are frozen; `tests/test_compat_train_wm.py` asserts them. The shared loop gives bit-identical train_wm logs/weights to the Phase 3 loop, and resumes a Phase 3 checkpoint (which lacks the new `torch_rng` key) identically (checked once against the old code).
- **Shared training code (Phase 4):** `wm/train/common.py::train_loop` (+ `resolve_config`) is the one loop for all five training jobs; `wm/jobs/_trainer.py::TrainingJob` holds the job plumbing (resume, checkpoint pushes, model.pt, summary, common CHECKS/METRICS). Checkpoints now also store torch's global RNG (`torch_rng`, `cuda_rng`) so BC's dropout resumes exactly; loading tolerates checkpoints without them.
- **Variance hinge (Phase 4):** WM-1step keeps λ_var = 1.0 (`configs/wm_1step.yaml` = `wm.yaml` except `teacher_forcing: true`, tested), so it differs from `wm` in teacher forcing only. BC and probe encoders get no hinge: their supervised targets differ per frame, so collapse can't fit them; their REPORTs show z_std as info only.
- **Pre-registered Phase 4 decisions (before any results):** (1) the probe predicts the block's **centre of gravity** (targets: pusher xy, `block_cog` xy normalised x/256 − 1, sin θ, cos θ), so "block px" matches `pose_error`; (2) WM-1step validation and filmstrips are **open-loop** (teacher forcing affects training only); (3) WM-1step's "beats copy-last at k = 10" is **WARN, never FAIL** (losing it is a legitimate baseline result, and a FAIL would leave `ckpt-wm-1step` incomplete); (4) BC samples t from every frame with an action; for t < 2 the missing context frames **repeat the episode's first frame**, as the planner does (§8); (5) BC and probe batch 256, probe 10,000 steps; all jobs use AdamW 3e-4, wd 1e-4, 500-step warm-up + cosine, fp16 AMP, grad-clip 1.0.
- **Pixel baseline (Phase 4):** trained on 4-frame windows (one-step); validation = open-loop 10-step rollouts on 256 fixed 13-frame val windows, pixel MSE_k vs copy-last; `loss` at val = MSE at k = 1. Params 4,813,731 (+15.0%, inside ±20%).
- **BC details (Phase 4):** target = (logged action − pusher)/50 (clean actions if `use_clean_actions`); goal = frame min(t + k, episode end), k ~ U{5..30}; random shift = replicate-pad 4 + random crop, one shift per sample shared by its 4 images, drawn from the sampler generator; no shift at validation. Val CHECK: MSE < constant mean-train-action baseline. Params 2,528,482.
- **Probe details (Phase 4):** trained on single train frames; val on val episodes; final errors on all 3,030 test frames (mean and median block px, block °, pusher px, plus the predict-train-mean baseline) saved in `ckpt-probe/train_summary.json` under `test`. Params 1,740,518.
- **Smoke settings (Phase 4):** local 50 steps (batch 4 wm_1step/pixel, 8 BC, 16 probe), warm-up 5, val/ckpt every 25; Kaggle 300 steps at full batch, warm-up 30, val every 100, ckpt every 150.
- **diagnose_wm (Phase 3 add-on, eval only):** job registered under Phase 3 (needs `pusht-data`, `ckpt-wm`; `config: null`, `makes: []`, so it never skips and never touches the store). It pulls the finished `ckpt-wm` (smoke falls back to `ckpt-wm-smoke`) and reports, train episodes vs val episodes (190–199): each loss term on 256 fixed windows per split (+ open-loop NMSE_k); single-frame encode→decode MSE on all pixels and on non-background pixels (any channel differs from the train mean frame by > 0.1, i.e. ~25/255), next to the mean-frame MSE on the same pixels; a ridge probe z → state (λ = 1.0, unpenalised bias, fit on ≤ 4,096 train frames, targets as the §7 probe incl. block CoG) on val frames and on imagined latents at k = 1, 5, 10 (beside real z_{t+k} and copy-last z_t); and a filmstrip of 8 val reconstructions. Writes `diagnose_wm.json` + `recon_filmstrip_val.png` to `outputs/diagnose_wm/` (and the filmstrip to W&B). Note: `var` is a batch statistic, computed over chunks of 64 windows (13·64 latents), as in validation.
- **§3 termination:** never end an episode early; step all 100 actions even if `terminated`. (gym-pusht 0.1.8's `step()` does keep working after termination.) If it ever refuses, replace that episode with an extra train seed from 2000 upwards. Validation is always seeds 190–199; test is always 1000–1029.

- **Kaggle `train_wm` full (Phase 3, v1):** finished: `ckpt-wm:v5`, step 30000, git 5dedc02, 37m15s, val NMSE@10 0.168 vs copy-last 0.230.
- **Kaggle `diagnose_wm` on `ckpt-wm` v1 (as pasted by Aditya):** ridge probe z → state on val: pusher 74.8 px (predict-mean 146.5), block 15.4 px, angle 68.2°; on train pusher 70.7 px, so the pusher is missing from z, not overfit. Non-background pixels = 3.6% of the image. NMSE@10 train 0.030 vs val 0.168 (copy-last 0.371 / 0.230). Imagined block error on val at k = 10: 29.5 px vs copy-last 29.8 px. Val loss 0.346, of which 0.336 is the variance term.
- **Phase 3b options (all off by default; `wm.yaml` / `wm_1step.yaml` unchanged, `train_wm` hash still `d2385fb211c8`, v1 smoke logs and weights bit-identical to commit d861a67):**
  - **(A) `fg_weight`:** rec and img losses = element mean of `w·(D(z) − o)²`, `w = 1 + α·mask`, α = 20, mask = any channel of |frame − train mean frame| > 25/255 (mean frame computed once from train frames, stored as buffer `fg_mean` in model.pt). No renormalisation by mean(w).
  - **(B) `proprio`:** pusher position p (`states[:, :2]/256 − 1`) is model state. φ = Linear(256+2+2 → 256) on [z, p, a]; new head `p_out = Linear(320 → 2)`, `p' = p + p_out(h')`; loss term `proprio` = MSE(p̂_{t+1..t+10}, p), weight 1.0. Warm-up uses real p; rollouts feed back p̂; decoder is still D(z). API: `warmup(z, a, p_ctx)`, `rollout(z0, h, a, p0=...) -> (z_hat, p_hat)`. **Params: dynamics 704,514, total 4,186,053, planner 2,443,490** (encoder/decoder unchanged).
  - **`p_out` is zero-initialised** (so p̂ starts as copy-last). Local check, not a result (`scripts/check_proprio_init.py`, smoke data, 200 steps, batch 8): val NMSE@10 at step 200 = A 0.056, AB random-init 0.222, AB zero-init 0.072. The random head's large early pusher error dominated the clipped gradient.
  - **(C) `keep_best`:** `train_loop(select=...)` keeps the weights with the lowest val NMSE@10 (saved in checkpoints under `best`, restored on resume); the final artifact uses them and records `best_step`, `best_nmse10`, `last_nmse10`.
- **`wm_ablate` (Phase 3b job):** variants = `configs/wm.yaml` + flags, 5,000 steps (warm-up 500, cosine), val every 250, same seed and identical batches. **Three runs, four rows:** v1, A, and AB trained once with keep_best; its last weights are row "A+B", its best-NMSE@10 weights row "A+B+C". All metrics on val episodes (256 fixed windows, all val frames): NMSE@5/@10 vs copy-last (each normalised by its own latent spread, so not compared across rows), ridge probe (λ = 1, ≤ 4,096 train frames) on real z and on imagined z at k = 10 vs copy-last, p̂ pusher px at k = 10 for B rows, non-background recon MSE (0.1 mask as in diagnose_wm), it/s. Writes `ablation_wm_fixes.json` + recon filmstrips; full mode also writes `results/ablation_wm_fixes.json` in the Kaggle checkout; pushes `results-wm-ablate` after each run so a rerun skips finished runs.
- **Pre-registered `wm_ablate` selection rule (2026-11-26, before any Kaggle result; implemented in `wm_ablate.select_winner`):** the winner is the row with the lowest imagined block px at k = 10 on val that (a) beats copy-last on block px at k = 10 AND (b) has imagined angle error at k = 10 no worse than copy-last's. If no row satisfies both, report that plainly and pick the lowest block px among rows satisfying (a). Tie-break (within 1 px): lower pusher error, using p̂ at k = 10 for B rows and the ridge pusher px on real z otherwise. Errors are means over val windows; imagined angle at k = 10 is in the table for every row.
- **Note for Phase 6 (not built):** with proprio, the planner's goal cost uses only the image latent z_g (the goal is an image), not the goal pusher position.

## Installed versions (local)
python 3.12.2, torch 2.15.0, gym-pusht 0.1.8, gymnasium 1.4.0, pymunk 6.11.1, pygame 2.6.1, opencv-python(-headless) 4.14.0.94, numpy 2.5.5, wandb 0.31.2.

## Open issues
- Kaggle v1 val loss is almost all variance hinge (0.336 of 0.346): val latents are far less spread per dimension than train ones. `wm_ablate` reports z_std per row; revisit after it.
- Getting `results/ablation_wm_fixes.json` into the repo: the Kaggle run writes it to `/kaggle/working/pusht-latent-world-model/results/` and to the W&B artifact `results-wm-ablate`; Aditya downloads that one file from the Kaggle output and drops it in `results/`.
- The local smoke filmstrip's decoded row is still a blur (50 steps, batch 4, CPU); judge the decoder on the Kaggle full run.
- Local `projected_30k_steps` (2h07m) is CPU at batch 4; the real projection comes from the Kaggle smoke.
- **Phase 4 Kaggle budget:** `train_baselines` full = wm_1step 30k + pixel 30k + BC 20k (batch 256 × 4 images) + probe 10k steps in one group. The Kaggle smoke's four `projected_*` lines tell us whether it fits one 12 h session; if not, rerunning the same cell skips finished jobs and resumes the unfinished one from its last checkpoint.
- BC smoke `val_z_std` = 0.006 (50 CPU steps). No CHECK by design; watch it on the Kaggle full run.
- **Mac only:** on import, macOS prints `objc ... Class SDL... is implemented in both cv2/.dylibs/libSDL2 and pygame/.dylibs/libSDL2` to stderr, because both OpenCV wheels bundle SDL2 on macOS. It's harmless with `SDL_VIDEODRIVER=dummy` and all tests pass. It doesn't occur on Kaggle/Colab (Linux).

## Last REPORT (local smoke, wm_ablate)
```
==================== RUN REPORT ====================
job: wm_ablate   mode: smoke   platform: local   device: cpu
git: d861a67-dirty   time: 51.6s   params: v1 4,184,899 / A 4,184,899 / AB 4,186,053
CHECKS
 [PASS] metrics_finite: all finite (no NaN/inf)
 [PASS] param_counts_v1: total 4,184,899 / planner 2,442,336 (== CLAUDE.md §4)
 [PASS] param_counts_A: total 4,184,899 / planner 2,442,336 (== CLAUDE.md §4)
 [PASS] param_counts_AB: total 4,186,053 / planner 2,443,490 (== CLAUDE.md §4 + proprio)
 [PASS] z_std_v1: 0.808 (> 0.1, collapse alarm; WARN only in the ablation)
 [PASS] z_std_A: 0.831 (> 0.1, collapse alarm; WARN only in the ablation)
 [PASS] z_std_A+B: 0.817 (> 0.1, collapse alarm; WARN only in the ablation)
 [PASS] z_std_A+B+C: 0.817 (> 0.1, collapse alarm; WARN only in the ablation)
METRICS
 data = pusht-data-smoke
 val_source = train episode seed 1 (smoke: no val episodes)
 steps = 50 per run (batch 4), same seed and batches
 val_windows = 16
 table = nmse@5 (copy) | nmse@10 (copy) | ridge real z: block px / angle / pusher px | imagined@10 block px (copy) | angle deg (copy) | pusher px (copy) | p_hat@10 px (copy) | recon fg mse | it/s
 v1     = 0.174 (0.171) | 0.434 (0.477) | 2.7 / 3.3 / 15.6 | 58.2 (37.0) | 20.3 (20.2) | 102.4 (81.1) | n/a (n/a) | 0.06015 | 4.03
 A      = 0.173 (0.182) | 0.425 (0.507) | 2.6 / 3.5 / 15.3 | 55.8 (37.0) | 15.9 (20.1) | 88.5 (81.1) | n/a (n/a) | 0.02410 | 3.75
 A+B    = 0.163 (0.156) | 0.413 (0.449) | 2.6 / 3.4 / 15.2 | 46.6 (36.6) | 30.7 (20.1) | 137.4 (80.5) | 39.1 (75.9) | 0.02414 | 3.93
 A+B+C  = 0.163 (0.156) | 0.413 (0.449) | 2.6 / 3.4 / 15.2 | 46.6 (36.6) | 30.7 (20.1) | 137.4 (80.5) | 39.1 (75.9) | 0.02414 | 3.93
 run_v1 = 4.03 it/s, projected 30k steps 2h03m59s, trained in 16.8s
 run_A = 3.75 it/s, projected 30k steps 2h13m24s, trained in 14.7s
 run_AB = 3.93 it/s, projected 30k steps 2h07m08s, trained in 14.2s
 keep_best = best step 50 of 50 (val nmse@10 best 0.413 vs last 0.413)
 winner = none (NO row beats copy-last on imagined block px at k=10)
SAVED
 outputs/wm_ablate/artifact/ablation_wm_fixes.json
 store/results-wm-ablate-smoke
W&B: off
====================================================
```

## Phase 4 local smoke REPORTs (train_baselines)
```
==================== RUN REPORT ====================
job: train_wm_1step   mode: smoke   platform: local   device: cpu
git: 5dedc02-dirty   time: 18.3s   params: 4,184,899
CHECKS
 [PASS] teacher_forcing: true (true for wm_1step)
 [PASS] param_counts: encoder 1,738,976 / decoder 1,742,563 / dynamics 703,360 / total 4,184,899 / planner 2,442,336 (== CLAUDE.md §4)
 [PASS] losses_finite: all finite (no NaN/inf)
 [PASS] loss_fell: 1.0984 -> 0.2374 (mean of last 10 < first 10 steps)
 [PASS] z_std: 0.771 (> 0.1, collapse alarm)
 [PASS] checkpoint_roundtrip: identical latents (model.pt reload == trained model)
 [WARN] nmse_k10_vs_copy_last: 0.667 vs 0.462 (model < copy-last; WARN only for the 1-step baseline)
METRICS
 data = pusht-data-smoke
 steps = 50 (batch 4)
 resumed_from_step = 0
 train_windows = 178
 val_source = train episode seed 1 (smoke: no val episodes)
 val_windows = 16
 loss_first_last = 1.0984 -> 0.2374
 rec_first_last = 0.1377 -> 0.0215
 lat_first_last = 0.0456 -> 0.0066
 img_first_last = 0.1374 -> 0.0214
 it_per_s = 3.82
 projected_30k_steps = 2h10m48s
 peak_gpu_mem = n/a (cpu)
 val_nmse_k1 = 0.012 (copy-last 0.010)
 val_nmse_k5 = 0.234 (copy-last 0.172)
 val_nmse_k10 = 0.667 (copy-last 0.462)
 val_z_std = 0.771
 val_loss = 0.2859
 lr_final = 3.65e-07
SAVED
 outputs/train_wm_1step/artifact/model.pt
 outputs/train_wm_1step/artifact/train_summary.json
 outputs/train_wm_1step/artifact/config.yaml
 outputs/train_wm_1step/artifact/rollout_filmstrip.png
 store/ckpt-wm-1step-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: train_pixel   mode: smoke   platform: local   device: cpu
git: 5dedc02-dirty   time: 3.5s   params: 4,813,731
CHECKS
 [PASS] param_counts: 4,813,731 (+15.0% vs world model 4,184,899) (== 4,813,731, within ±20%)
 [PASS] losses_finite: all finite (no NaN/inf)
 [PASS] loss_fell: 0.0160 -> 0.0020 (mean of last 10 < first 10 steps)
 [PASS] checkpoint_roundtrip: identical predictions (model.pt reload == trained model)
 [WARN] one_step_vs_copy_last: 0.00208 vs 0.00068 (val pixel MSE at k=1 < copy-last; WARN only in smoke)
METRICS
 data = pusht-data-smoke
 steps = 50 (batch 4)
 resumed_from_step = 0
 train_windows = 196
 val_source = train episode seed 1 (smoke: no val episodes)
 val_windows = 16
 loss_first_last = 0.01603 -> 0.00196
 it_per_s = 29.20
 projected_30k_steps = 17m08s
 peak_gpu_mem = n/a (cpu)
 val_pixel_mse_k1 = 0.00208 (copy-last 0.00068)
 val_pixel_mse_k5 = 0.04477 (copy-last 0.00302)
 val_pixel_mse_k10 = 0.36358 (copy-last 0.00474)
 lr_final = 3.65e-07
SAVED
 outputs/train_pixel/artifact/model.pt
 outputs/train_pixel/artifact/train_summary.json
 outputs/train_pixel/artifact/config.yaml
 outputs/train_pixel/artifact/rollout_filmstrip.png
 store/ckpt-pixel-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: train_bc   mode: smoke   platform: local   device: cpu
git: 5dedc02-dirty   time: 8.1s   params: 2,528,482
CHECKS
 [PASS] param_counts: encoder 1,738,976 / mlp 789,506 / total 2,528,482 (== CLAUDE.md §6 (≈2.53M))
 [PASS] losses_finite: all finite (no NaN/inf)
 [PASS] loss_fell: 0.3661 -> 0.3100 (mean of last 10 < first 10 steps)
 [PASS] checkpoint_roundtrip: identical actions (model.pt reload == trained model)
 [WARN] val_mse_vs_mean_action: 0.1977 vs 0.1952 (val MSE < constant mean-action baseline; WARN only in smoke)
METRICS
 data = pusht-data-smoke
 steps = 50 (batch 8)
 resumed_from_step = 0
 train_samples = 200
 val_source = train episode seed 1 (smoke: no val episodes)
 val_samples = 64
 actions = logged (noisy)
 loss_first_last = 0.3661 -> 0.3100
 it_per_s = 8.81
 projected_20k_steps = 37m51s
 peak_gpu_mem = n/a (cpu)
 val_mse = 0.1977 (mean-action baseline 0.1952)
 val_action_err_px = 29.4 (mean-action baseline 29.4)
 val_z_std = 0.006 (info only; no variance hinge)
 lr_final = 3.65e-07
SAVED
 outputs/train_bc/artifact/model.pt
 outputs/train_bc/artifact/train_summary.json
 outputs/train_bc/artifact/config.yaml
 store/ckpt-bc-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: train_probe   mode: smoke   platform: local   device: cpu
git: 5dedc02-dirty   time: 3.3s   params: 1,740,518
CHECKS
 [PASS] param_counts: encoder 1,738,976 / head 1,542 / total 1,740,518 (== CLAUDE.md §7)
 [PASS] losses_finite: all finite (no NaN/inf)
 [PASS] loss_fell: 0.3316 -> 0.0154 (mean of last 10 < first 10 steps)
 [PASS] checkpoint_roundtrip: identical outputs (model.pt reload == trained model)
 [PASS] test_block_px_vs_mean: 104.2 vs 142.0 (mean block error < predict-train-mean baseline; WARN only in smoke)
METRICS
 data = pusht-data-smoke
 steps = 50 (batch 16)
 resumed_from_step = 0
 train_frames = 202
 val_source = train episode seed 1 (smoke: no val episodes)
 test_frames = 101
 loss_first_last = 0.3316 -> 0.0154
 it_per_s = 18.97
 projected_10k_steps = 8m47s
 peak_gpu_mem = n/a (cpu)
 test_block_px = mean 104.24 / median 122.84 px (mean baseline 142.03)
 test_block_deg = mean 103.20 / median 129.04 deg (mean baseline 82.06)
 test_pusher_px = mean 161.58 / median 188.15 px (mean baseline 164.36)
 val_block_px_mean = 16.96
 test_z_std = 0.058 (info only; no variance hinge)
 lr_final = 3.65e-07
SAVED
 outputs/train_probe/artifact/model.pt
 outputs/train_probe/artifact/train_summary.json
 outputs/train_probe/artifact/config.yaml
 store/ckpt-probe-smoke
W&B: off
====================================================
```

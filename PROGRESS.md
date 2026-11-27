# PROGRESS

**Current phase: 6 (CEM planner + validation tuning), built; local `phase6` smoke PASS → Kaggle.** Phase 6 adds make_pairs, train_bc_clean, the CEM planner / MPC runner, plan_val and plan_sweep; its decisions are pre-registered below. Next: Aditya pushes, runs Kaggle `phase6` smoke (we read cem_pixel's timing and decide its flag), then `phase6` full, and drops `results/plan_val.json`, `results/plan_sweep.json` and `results/eval_pairs_{val,test}.json` into `results/`; then I freeze `configs/plan.yaml`. Phase 5b is **done** (Kaggle full; judge freeze decision below).

## Status
| Phase | Status | Local smoke | Kaggle smoke | Kaggle full |
| --- | --- | --- | --- | --- |
| 1 Foundation | done | PASS | PASS | n/a |
| 2 Expert + data | **complete** | PASS | PASS | PASS |
| 3 World model | A+B trained in `overnight` (step 30000, used by eval_rollouts full); v1 = `ckpt-wm:v5` | PASS | PASS | PASS (A+B) |
| 3b WM fixes | **complete** (A+B adopted) | PASS | PASS | PASS |
| 4 Baselines | done (`overnight`; BC = `ckpt-bc:v3`, git 2227d6f) | PASS | PASS | PASS |
| 5 Rollout eval | Kaggle full done (adc200d); judge failed → 5b | PASS | PASS | PASS (`results/rollouts.json`) |
| 5b Angle audit + judges | **DONE** (judges frozen, see Decisions) | PASS | PASS | PASS (`results/angle_audit.json`, `results/rollouts.json`) |
| 6 Planner | built; local smoke PASS | PASS | pending | pending |
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
- **Phase 3b decision (2026-11-27): adopt A+B; C not adopted.** Kaggle `wm_ablate` full at commit ffd775a. Source: `results/ablation_wm_fixes.json` (table generated from it). All columns are on val episodes; errors are means over 256 val windows; "copy-last" in the imagined columns = each row's own ridge probe applied to z_t, so it differs per row. NMSE is normalised by each row's own latent spread, so it is not compared across rows.

  | row | weights | z_std | NMSE@10 (copy-last) | ridge on real z: block px / angle ° / pusher px | imagined@10 block px (copy-last) | imagined@10 angle ° (copy-last) | p̂ pusher px @10 (copy-last) | recon fg MSE | rule (a) / (b) |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | v1 | last | 0.879 | 0.086 (0.170) | 11.8 / 80.5 / 79.3 | 32.6 (27.2) | 88.8 (84.6) | n/a | 0.0396 | no / no |
  | A | last | 0.901 | 0.155 (0.205) | 6.5 / 89.1 / 69.2 | 28.9 (30.1) | 84.3 (93.7) | n/a | 0.0114 | yes / yes |
  | **A+B** | last | 0.913 | 0.119 (0.235) | 8.4 / 87.7 / 64.4 | 28.0 (33.8) | 72.6 (92.6) | 2.2 (60.2) | 0.0101 | yes / yes |
  | A+B+C | best@2500 | 0.921 | 0.102 (0.239) | 9.2 / 90.7 / 63.3 | 33.1 (35.1) | 91.7 (96.2) | 5.3 (60.2) | 0.0104 | yes / yes |

  Selection (from the JSON): winner **A+B**; satisfies (a) block px < copy-last and (b) angle <= copy-last at k=10; tie within 1 px among ['A', 'A+B'] broken by pusher error. Mean-frame fg MSE = 0.1054. Run: full on kaggle, git ffd775a, 5,000 steps per run (batch 64), val episodes (seeds 190-199).

  By the pre-registered rule, A and A+B both satisfy (a) and (b) and are within 1 px on block error (28.9 vs 28.0), so the tie-break on pusher error picks **A+B** (p̂ 2.2 px vs A's ridge pusher 69.2 px). v1 fails (a): 32.6 px vs copy-last 27.2. **C not adopted:** keep_best chose step 2500 by val NMSE@10 (0.102, the lowest NMSE@10 in the table), but those weights are physically worse than A+B's last weights: imagined block 33.1 vs 28.0 px, angle 91.7° vs 72.6°, p̂ 5.3 vs 2.2 px. So NMSE@10 is not a reliable checkpoint-selection metric here.
- **A+B is the default recipe (2026-11-27):** `configs/wm.yaml` and `configs/wm_1step.yaml` set `fg_weight: true` (α 20, threshold 25/255), `proprio: true` (`loss.proprio: 1.0`), `keep_best: false`; they still differ only in `teacher_forcing` (tested), and `wm.yaml` equals the `wm_ablate` AB variant minus keep_best and its 5k-step schedule (tested). Params: total 4,186,053, planner 2,443,490 (CLAUDE.md §4 updated). The Phase 3 recipe is kept unchanged as `configs/wm_v1.yaml` (hash-checked: it reproduces v1's train_wm hash `d2385fb211c8`). **`ckpt-wm:v5` on W&B is the v1 model.**
- **config_hash changes (deliberate, 2026-11-27):** `train_wm` d2385fb211c8 → **d4c8e342b0d6** (guard test updated), `train_wm_1step` → dd5934291cbe. So the next `train_wm` full does not skip and does not resume from v5: it trains A+B from scratch (v5 is complete with the old hash). `wm_ablate`'s `base_config` now points at `wm_v1.yaml` (so its v1 row keeps its meaning); that changed its hash to 76b1611426f5, so a rerun would not skip — it is finished and should not be rerun. `scripts/compare_collapse_fixes.py` and `scripts/check_proprio_init.py` also read `wm_v1.yaml` now. The Phase 4 refactor guard (frozen v1 hash) is superseded.
- **`overnight` group (2026-11-27):** `jobs.yaml` group = `train_wm` → `diagnose_wm` → `train_baselines`; a member that is a group is expanded in place (`run.expand_group`), so it runs train_wm, diagnose_wm, train_wm_1step, train_pixel, train_bc, train_probe and stops at the first failure. Finished training jobs skip; `diagnose_wm` makes nothing, so it always reruns (eval only, a few minutes).
- **diagnose_wm with proprio (2026-11-27):** loss terms include `proprio`; for a proprio model it also reports the open-loop p̂ pusher error at k = 1, 5, 10 vs copy-last p_t, train and val (`p_hat_pusher` in `diagnose_wm.json`, `p_hat_pusher_{train,val}` in the REPORT).
- **Kaggle `overnight` smoke #1 (commit 3c6f9a2, as reported by Aditya):** train_wm and train_wm_1step PASS; train_pixel `[FAIL] checkpoint_roundtrip: mismatch` on CUDA (it passed on CPU); the group then stopped, so train_bc and train_probe never ran; diagnose_wm diagnosed the old v1 `ckpt-wm` (step 30000, git 5dedc02, proprio False).
- **Fix: checkpoint round trip tolerance (2026-11-27):** `wm/jobs/_trainer.py::roundtrip` puts both models in eval mode, requires bit-identical reloaded weights, and compares outputs with `torch.allclose(rtol=1e-4, atol=1e-5)`. It prints and reports the max absolute and relative difference, and the tolerance is on the REPORT line. Used by all four training jobs (train_wm/1step, pixel, BC, probe). Cause, from the code (not measured; there is no GPU locally): both models were already in eval mode, and only the pixel check runs a `ConvTranspose2d` (its up path). CUDA computes that through cuDNN's backward-data kernels, which can be nondeterministic, so two calls may differ in the last bits. The world-model check only compares `encode` (plain convs), BC and the probe have no transposed convs, and the exact match passes on CPU. **To confirm on Kaggle:** the next smoke's train_pixel line must show weights identical and a max |diff| at float-noise level. If the difference is larger, it's a real bug and the check will FAIL.
- **Fix: groups continue on failure (2026-11-27):** `run.run_group` runs every member even after a FAIL (or an exception outside the job, e.g. W&B), prints `[group <g>] <job>: PASS|SKIP|FAIL|REFUSED` after each job and a final `GROUP <g>: OK|FAILED (n of m passed or skipped) | PASS ... | SKIP ... | FAIL ...` line, and exits 1 if anything failed. Registry helpers moved from run.py into `wm/registry.py` (run.py re-exports them).
- **Fix: hash-matched checkpoints (2026-11-27):** `wm/jobs/_ckpt.py::pull_matching(ctx, artifact)` returns the first read candidate (smoke: `<name>`, then `<name>-smoke`) that is complete and has the config_hash the producing job's current config gives (`wm.registry.producer`). Otherwise it raises a FAIL naming each candidate's hash, `complete` and step. diagnose_wm uses it, adds CHECK `ckpt_config_hash` and prints `ckpt_config_hash` and `proprio`. This is the rule for every later eval job (CLAUDE.md §10). Note: the W&B store only looks at each name's `:latest` version.
- **Phase 5 rollout evaluation, pre-registered (2026-11-27, before any Kaggle result):**
  - **Config:** `configs/rollouts.yaml` (`jobs.yaml` `eval_rollouts.config` was `null`), so `config.yaml` is saved with the results.
  - **Windows:** 500 test windows of 13 frames (o[t−2..t+10], a[t−2..t+9]) from `pusht_test.npz`, a fixed seed-0 subset of all 2,670 valid starts (smoke: 20). Every method gets the same windows.
  - **Methods:** `wm` (A+B), `wm_1step`, `pixel`, `copy_last`, all open loop: wm_1step's teacher forcing affects training only. Every checkpoint comes through `_ckpt.pull_matching`.
  - **Primary metrics:** the CNN state probe on every predicted frame gives block CoG px, block angle ° and pusher px vs the true state at t+k, k = 1..10. For proprio models, pusher px also comes directly from p̂ (copy-last = p_t).
  - **References next to every table:**
    - `real` = the probe on o_{t+k} for the same windows (noise floor per k);
    - `wm_recon` = the probe on D_wm(E_wm(o_{t+k})) (autoencoder floor: decode loss without dynamics);
    - `probe_test_all` = the probe on all real test frames, recomputed in the job, with the predict-train-mean baseline.
  - **Headline NMSE (resume/README "X% lower error than the pixel baseline"):** the like-for-like row **`wm_via_frames` = E_wm(D_wm(ẑ))** against `pixel` and `wm_1step`. All of their frames are re-encoded by E_wm, normalised by the spread of the wm-encoded test frames. **Direct-ẑ NMSE (row `wm`) is secondary and labelled as such.** `wm_recon` NMSE shows the decode/re-encode floor.
  - **Secondary:** pixel MSE.
  - **CIs:** 95% percentile bootstrap, 1,000 resamples over windows, seed 0. The same resample indices serve every row, so differences are paired. Paired CIs:
    - 1 − wm_via_frames ÷ pixel and 1 − wm_via_frames ÷ wm_1step (NMSE);
    - 1 − wm ÷ pixel (block px);
    - **wm − pixel block px at k = 5 and 10**;
    - wm − copy_last block px at k = 10.
  - **Clipping:** every predicted frame (wm and wm_1step decodes, pixel predictions, `wm_recon`) is clipped to [−0.5, 0.5], the range of real normalised frames, before the probe, the encoder or pixel MSE see it. The U-Net's own autoregressive loop stays as trained (unclipped), the same as its training-time validation.
  - **CHECK severity:**
    - FAIL only for a broken evaluation: a checkpoint hash mismatch, NaN/inf, the wrong window count, or `probe_floor` (the probe on real test frames does not beat predict-mean; WARN in smoke).
    - WARN only: §9's `nmse_wm_vs_copy_last_all_k` (on `wm_via_frames`; the direct-ẑ count is in the value), `block_px_wm_vs_copy_last_k10`, and **`probe_angle_floor`** (probe mean block angle on all real test frames < 30°; otherwise the REPORT says "angle columns unreliable — judge cannot read angle from real frames").
  - **Outputs:** `rollouts.json`, 5 plots (metric vs k with CI bands; the probe-on-all-test-frames line on the probe plots; p̂ dashed on the pusher plot) and 6 filmstrips (real | wm | wm_1step | pixel, frames t..t+10). Pushed as `results-rollouts`; full mode also writes `results/rollouts.json` in the checkout.
- **Eval skip rule (Phase 5):** a job whose `needs` include `ckpt-*` stores `inputs` = {checkpoint: its producer's current config_hash} in its results metadata. `run.is_finished` treats the results as unfinished if that no longer matches (e.g. after `wm.yaml` changes), so stale evals never skip. Training jobs are unaffected (they need no checkpoints). `wm.registry.input_hashes`; Phases 6–7 reuse it. Recorded in CLAUDE.md §10.
- **Note for Phase 6 (not built):** with proprio, the planner's goal cost uses only the image latent z_g (the goal is an image), not the goal pusher position.
- **Kaggle `eval_rollouts` full (Phase 5, commit adc200d; source `results/rollouts.json`, 500 test windows, checkpoints ckpt-wm/-1step/-pixel step 30000, ckpt-probe step 10000, all at current hashes):**
  1. **The v1 probe is a broken judge for decoded frames.** On real test frames it reads the block well (all 3,030 frames: 10.7 px mean; row `real`: 10.1–10.8 px per k), but on `wm_recon` = D_wm(E_wm(real)) it gives 67.0–68.8 px, and pusher 104–116 px vs 22 px on real frames. So every probe number on wm and wm_1step frames (block 59–91 px) measures the judge, not the model. Probe rows on pixel frames (block 13.7 → 42.9 px at k = 1 → 10, copy-last 11.5 → 32.4) are less affected but also not trusted until a fair judge confirms them.
  2. **Angle is unreadable by every judge so far.** Probe block angle on real test frames: mean 76.2°, median 68.9° (predict-mean 94.7°); the ridge probe on z gets 80–91° (`results/ablation_wm_fixes.json`). A CNN on sharp 64×64 frames should read a T's angle easily, so a label or measurement bug was suspected. **Read-only pre-check (2026-11-27):** re-rendering 40 stored smoke states with `reset_to_state` reproduces the stored frames exactly (mean abs diff 0.00; with −θ, θ+90° and θ+180° it is 3.3–4.5), so frames and angle labels agree. On reading, `state_targets`, `decode_targets` and `probe_errors` are correct. The probe's test loss of 0.254 (mean over 6 outputs) means its sin/cos MSE is about 0.75 per output, worse than predicting the mean (about 0.5): it is confidently wrong on held-out frames, which points to memorisation. Phase 5b's `angle_audit` checks this properly.
  3. **Pre-registered NMSE headline (`wm_via_frames`), recorded as is:** at k = 10, wm_via_frames 0.122 [0.098, 0.150] vs pixel 0.084 [0.066, 0.104]. The reduction 1 − wm ÷ pixel is −0.454 [−0.900, −0.095]: **the pixel baseline is better** (at k = 5: −0.706 [−1.421, −0.249]). Against wm_1step, wm_via_frames is **60% lower** at k = 10 (0.600 [0.512, 0.676]; 0.305 vs 0.122) and 50% at k = 5 (0.504 [0.349, 0.619]). Direct-ẑ NMSE (secondary) at k = 10: 0.127. p̂ pusher error at k = 10: wm 2.4 px, wm_1step 11.0, copy-last 66.8.
- **Phase 5b judge decision (pre-registered 2026-11-27 by Aditya, before any judge result):**
  - **Judges are instruments.** Only the judges (`probe_mix`, the latent MLP) train on extra frames: the 20k-transition frames **plus about 50,000 frames rendered from random valid states** with `reset_to_state`. Block CoG is uniform in [85, 427]² (the T stays inside the walls), angle uniform in [−π, π), pusher not overlapping the block, round trip < 1 px. wm, wm_1step, pixel and bc are untouched and still see only the 20k transitions.
  - **Rendered data** = 12,500 four-frame micro-episodes (random start state, then 3 random actions with target = p + N(0, 30²), clipped), seeds from 5000 up. They give the pixel model its 3 context frames + action, and the true next state labels its 1-step prediction. 10% of micro-episodes are a held-out rendered validation split for early stopping.
  - **Training:** ±4 px random shift (replicate-pad + crop), with position labels shifted to match (8 px per 64-px pixel; angle unchanged); AdamW wd 1e-4; keep-best on rendered-val loss (`train_loop(select=...)`).
  - **Architecture:** the §7 encoder first. Only if probe_mix's real-frame test angle is > 30° does the job also train a `coord_gap` variant (CoordConv x/y input channels + global average pooling in place of flatten→Linear) as a documented deviation. Both are reported; coord_gap becomes the judge if it passes acceptance.
  - **probe_mix frame types** (1/4 of each batch, all labelled with the true state): real; D_wm(E_wm(x)); D_wm1step(E_wm1step(x)); pixel-model 1-step prediction (context o_{t−2..t} + a_t, labelled s_{t+1}; first frame repeated for t < 2). Derived frames are made on the fly and clipped to [−0.5, 0.5].
  - **Acceptance (probe_mix, test):** real-frame angle < 15°; real-frame block ≤ 1.25 × the v1 probe's 10.7 px (≤ 13.4 px); block and pusher px on each derived type ≤ 1.5 × the real-frame error. Angle on derived types is reported but not gated (the autoencoder may truly lose angle; the template oracle separates that from a judge failure). A failed acceptance check is a WARN; the summary records `accepted`, and eval_rollouts marks the rows of an unaccepted judge.
  - **Latent judge:** MLP 256→512→512→6 (SiLU, dropout 0.1) on E_wm(x) of the same real + rendered frames, keep-best on rendered val; reported on real E_wm(o) train / val / test.
  - **Headline rule:** "wm beats X on block px at k" is claimed only if both judges agree in sign and both paired 95% CIs exclude 0; otherwise the README says the judges disagree.
  - **Angle verdict:** latent-MLP mean angle error on real E_wm(o) test frames < 20° = **present** (rotation goals plausible), 20–45° = **weak**, > 45° = **absent** (Phase 6 must restrict or flag rotation goals). It is printed beside the template oracle and predict-mean.
  - **No existing checkpoint retrains:** the training config_hash of train_wm, train_wm_1step, train_pixel, train_bc and train_probe stays unchanged (asserted in a test).
- **Phase 5b build (2026-11-27), implementation decisions (no results yet):**
  - **Angle convention, verified locally (`angle_audit`, `tests/test_angle_audit.py`):** radians wrapped to [−π, π); 0 = bar on top, stem pointing down; the angle grows clockwise on screen (image y points down); stored (x, y) = body origin, CoG = origin + R(θ)(0, 45). Re-rendering stored states reproduces stored frames exactly; the mask-moment angle matches the label within 1.7° mean (8.4° max) over 288 grid poses; the mask centroid sits 4.7 px from `block_cog` (the area centroid is 4.3 px from gym-pusht's CoG) vs 40 px from the body origin; target encode/decode round trip 0; wraps 359°/1° → 2°. The non-learned template oracle reads real test frames within ~1° (smoke, 10 frames). **No label bug.** In the 50-step smoke the v1 probe already shows train 8° vs test 103°: memorisation. The Kaggle run gives the real numbers.
  - **Overlap check:** gym-pusht's one physics step after `reset_to_state` does *not* separate an overlapping pusher (it's kinematic), so a round trip can't detect overlap. Rendered states use pymunk's signed point-to-shape distance instead (pusher edge ≥ 2 px from the T).
  - **Hash folding (`wm/registry.py`):** a job's config_hash includes the current hashes of everything it needs except `pusht-data`. Unchanged and asserted: train_wm d4c8e342b0d6, train_wm_1step dd5934291cbe, train_pixel 2438d918b486, train_bc dd9930d91fd0, train_probe 981a0e656148, collect_data 4af4e37608e6, wm_ablate 76b1611426f5. eval_rollouts changed (1c66c59f6bbc →), so it reruns.
  - **angle_audit** also needs `ckpt-wm-1step` and `ckpt-pixel`, so the template oracle covers all four frame types (real, wm_recon, wm1step_recon, pixel_1step). That is the reference for "angle on derived types is reported, not gated".
  - **Latent judge rows:** `wm` = direct ẑ (what the planner uses); `wm_via_frames` = E_wm(D_wm(ẑ)). The headline rule pairs probe_mix's `wm` (decoded frames) with the latent judge's `wm` (direct ẑ).
  - **probe_mix fallback** (coord_gap) runs in the same job without intermediate checkpoints. If it's killed mid-fallback, a rerun redoes the fallback from scratch (the flatten run resumes as usual).
  - **probe_mix with an untrained/unknown summary:** eval_rollouts treats a probe_mix without `accepted: true` in its train_summary as not accepted, so every headline says "no claim: image judge not accepted".
  - **rollouts.json schema 2:** metric keys `<judge>/block_px|block_deg|pusher_px` for judges probe_v1 / probe_mix / latent; `judges` (status, accepted, test_all, grid); `angle_in_wm_latent`; `template_oracle_test`; `headline.block_px` claims; paired `<judge>/block_px_diff_wm_minus_{pixel,wm_1step,copy_last}`. NMSE, pixel MSE and p̂ are unchanged. 11 plots (3 judges × 3 metrics + NMSE + pixel MSE).

- **Phase 5b results (Kaggle full; sources `results/angle_audit.json` (git 0382f7a) and `results/rollouts.json` (schema 2, git 0382f7a-dirty, 500 test windows); the Phase 5 `rollouts.json` is in git history at a343e14):**
  - **angle_audit:** no label bug (rerender, orientation, CoG convention, round trip all clean); verdict "labels consistent; failure = probe generalisation (memorisation)". v1 probe angle: train 13.4° vs val 84.8°. Template oracle (non-learned, true position given), 200 frames each: real 1.4° mean, wm_recon 5.6°, wm1step_recon 3.4°, pixel_1step 1.9°.
  - **probe_mix (flatten, best step 19000): NOT accepted.** Failed pre-registered checks: real angle 46.5° (gate < 15°), real block 15.5 px (gate ≤ 13.4), wm_recon pusher 117.0 px and wm1step_recon pusher 117.5 px (gate ≤ 1.5 × real 53.5 = 80.3 px). Passed: block px on every derived type and pixel_1step pusher. Test block px by frame type: real 15.5 / wm_recon 14.1 / wm1step_recon 15.7 / pixel_1step 14.5.
  - **Latent judge** on real E_wm(o) test frames: block 7.0 px, angle 24.7°, pusher 86.6 px. **`angle_in_wm_latent` = 24.7° → weak** (thresholds < 20 present, > 45 absent; predict-mean 94.7°; template oracle 1.3° on 100 real test frames).
  - **NMSE (headline row wm_via_frames):** reduction vs wm_1step at k = 10 **0.600 [0.512, 0.676]** (k = 5: 0.504); vs pixel **−0.454 [−0.900, −0.095]** at k = 10 (k = 5: −0.706 [−1.421, −0.249]): the pixel baseline is better.
  - **Block px, wm − pixel (paired 95% CI):** latent k5 +2.9 [2.1, 3.6], k10 +3.4 [2.3, 4.5]; probe_mix k5 +2.9 [1.8, 4.1], k10 +2.4 [1.1, 4.0]: **pixel beats wm** under both judges. Means at k = 10 (latent): wm 20.9, pixel 17.5, wm_1step 32.6, copy_last 32.1, real 6.8 px. Both judges agree in sign on every headline comparison: wm worse than pixel (k5, k10), better than wm_1step and copy_last (k5, k10); the JSON's `headline` says "no claim: image judge not accepted" for all six (pre-registered rule).
  - **Pusher, p̂ at k = 10:** wm 2.4 px, wm_1step 11.0, copy_last 66.8.
  - **Imagined angle (diagnosed failure mode):** wm's imagined angle is worse than copy_last at every k under both learned judges (latent: k1 30.6° vs 25.9°, k10 66.1° vs 31.5°; probe_mix: k1 56.3° vs 42.3°, k10 79.7° vs 44.4°). Not yet confirmed by the template oracle on imagined frames (Phase 7 TODO (a)).
- **Judge freeze decision (2026-11-27, by Aditya, made after seeing the Phase 5b results; a deviation from the pre-registered headline rule, which the README must disclose):**
  - **Block position:** the **latent MLP judge is primary** (real-frame floor 7.0 px). probe_mix (flatten) is a **sign-agreement check only**: it failed its pre-registered acceptance (above) but reads block position consistently across frame types (real 15.5 / wm_recon 14.1 / wm1step_recon 15.7 / pixel_1step 14.5 px). The pre-registered rule required an accepted image judge for any headline claim; the README states this deviation explicitly.
  - **Angle:** the **template oracle** (non-learned, true position given) is the only trusted angle reader (real 1.4°, wm_recon 5.6°). Learned-judge angle columns are reported but not trusted.
  - **Pusher:** **p̂ only** (wm 2.4 px vs wm_1step 11.0 vs copy_last 66.8 at k = 10); image-judge pusher columns are not used.
  - **Key findings carried forward:** angle_in_wm_latent 24.7° (weak); NMSE reduction vs wm_1step at k = 10 0.600 [0.512, 0.676]; the pixel baseline beats wm on NMSE and block px; wm's imagined angle is worse than copy_last at all k (diagnosed failure mode).
- **Kaggle `train_bc` full (Phase 4, from the `overnight` run, as reported by Aditya):** `ckpt-bc:v3`, git 2227d6f, val action error **30.9 px** vs the constant mean-action baseline 33.3 px. Not retrained; `train_bc`'s hash stays dd9930d91fd0 (asserted in `tests/test_registry_phase6.py`).
- **Phase 6 pre-registered decisions (2026-11-27, before any Phase 6 result):**
  1. **Planner state (z, p).** Each MPC step encodes the last 3 real frames with E_wm, `h = warmup(z_{t-2..t-1}, a_{t-2..t-1}, p_ctx = real pusher at t-2, t-1)`, then `rollout(z_t, h, a_cand, p0 = p_t)`; at episode start the first frame is repeated 3× and the current pusher position fills the padding actions and p_ctx. **The cost uses only z:** ‖ẑ_H − z_g‖² + β Σ_{k=1}^{H−1} ‖ẑ_k − z_g‖², β = 0.1, z_g = E_wm(goal frame); p̂ is fed back, never scored. `cem_wm_1step` uses the same code with `ckpt-wm-1step`.
  2. **Success (primary) unchanged:** block CoG < 20 px AND |Δθ| < 20°, checked after every env step, stop at the first success, budget 40.
  3. **Position-only success (secondary, every method, Wilson 95% CI):** block CoG < 20 px after any step (episodes still stop only on full success). Rotation-only pairs can start inside 20 px, so the headline secondary is on pairs whose start CoG error is ≥ 20 px; the all-pairs count is also reported.
  4. **Rotation vs translation breakdown (secondary, every method, Wilson 95% CIs):** success and position-only success split into rotation pairs (|goal angle − start angle| ≥ 20°) and translation pairs (< 20°), with the pair count of each group stored.
  5. **Per-pair records:** start / final / min pos px, start / final angle °, steps, both success flags, rotation flag, per-step best imagined cost and terminal cost (CEM), seconds.
  6. **BC's fair second try:** `train_bc_clean` = the §6 BC on `clean_actions` (`configs/bc_clean.yaml`, only `use_clean_actions` differs; tested). Both BC variants run on the 50 val pairs; the one with more successes goes to the final test; ties → more position-only successes (start ≥ 20 px subset) → lower median final pos px → `bc` (spec default). `plan_val` records `bc_selected`; Phase 7 reads it.
  7. **`cem_pixel` (flag `methods.cem_pixel`, on for the smoke):** CEM with the pixel U-Net's autoregressive rollout from the 3 real context frames; predicted frames clipped to [−0.5, 0.5], encoded by E_wm, scored with the same cost and β as cem_wm (β = 0 gives exactly "E_wm(final frame) vs z_g"); fp16 autocast on CUDA. A full-size timing probe (N 128, 4 rounds, H 10) gives projected s/episode in the plan_val REPORT; we decide after the Kaggle smoke whether it joins the final comparison (setting the flag changes plan_val's hash, fine before the full run).
  8. **Sweep freeze rule (`plan_sweep`, cem_wm only, 9 settings: default; N 32/64/256; rounds 1/2; H 5/15; β 0):** a variant replaces the default only with ≥ 3 more successes (of 50); several → the most, ties → the cheaper N × rounds × H; if every setting has ≤ 2 successes, the same rule runs on position-only success (start ≥ 20 px). `plan_sweep` computes `selected` mechanically; I then write it into `configs/plan.yaml` (`frozen: true`); it applies to every CEM method.
  9. **Imagined-success threshold τ (for Phase 7's `model_error` / `search_error` tags, calibrated on val in plan_val):** the 95th percentile of ‖E_wm(o_i) − E_wm(o_j)‖² over frame pairs of the same val episode whose states satisfy `is_success`; imagined success = the terminal term ‖ẑ_H − z_g‖² of the step's best plan < τ.
  10. **Seeds:** planner seed 0; each episode's CPU generator = seed × 100003 + pair index (CEM sampling and the random policy).
- **Phase 6 build (2026-11-27), implementation decisions (no results yet):**
  - `configs/pairs.yaml` is separate from `plan.yaml`, so freezing the CEM values does not change `eval-pairs`' hash. Pairs: starts 0..75 step 5, offset 20, ≥ 30 px or ≥ 20°, seed 0, 200 test / 50 val (smoke 3 / 3; the local smoke has no val episodes, so its last train episode stands in, as in training).
  - plan_val needs `ckpt-pixel` (for cem_pixel) and `ckpt-bc-clean`; plan_sweep needs only `eval-pairs` and `ckpt-wm`. Both store `inputs` and fold the input hashes in (Phase 5b rule).
  - CEM returns the cheapest candidate seen over all rounds; the executed target is its first waypoint; σ = std of elites (population std) floored at 2 px; sampling uses a CPU generator so results don't depend on the device.
  - CHECKS (FAIL = broken eval): checkpoint/eval-pairs hashes, pair count, every method ran every pair, **render alignment** (frame after `reset_to_state(start / goal state)` vs stored start / goal frame, mean abs diff ≤ 0.5 uint8), **noop success == 0** (impossible by the pair filter), finite metrics. WARN only: cem_wm success > random.
  - Local smoke runs cem_pixel's episodes at N 16 / 2 rounds (CPU: a full-size call takes ~29 s); its timing probe is always full size. The Kaggle smoke runs everything full size.
  - plan_val saves 3 filmstrips (real frames 1..H | decoded imagined plan of the first step | goal) of the first cem_wm val episodes.

## Phase 7 TODO
- (a) **Template-oracle angle on imagined frames:** a column with the template oracle (non-learned, true position given) on *imagined* frames for wm (decoded ẑ), wm_1step, pixel and copy_last at k = 1, 5, 10, to confirm or refute the learned judges' "wm imagined angle worse than copy_last" finding with the only trusted angle reader.
- (b) **"-dirty" on Kaggle:** full-mode jobs write `results/*.json` into the checkout, so later jobs in the same session report git "-dirty" (e.g. `rollouts.json` shows 0382f7a-dirty). Exclude `results/` from the dirty check in `wm/repro.git_info`.

## Installed versions (local)
python 3.12.2, torch 2.15.0, gym-pusht 0.1.8, gymnasium 1.4.0, pymunk 6.11.1, pygame 2.6.1, opencv-python(-headless) 4.14.0.94, numpy 2.5.5, wandb 0.31.2.

## Open issues
- **Angle in the wm latent is weak (Phase 5b: 24.7°, latent judge).** Phase 6 reports rotation vs translation pairs and position-only success (pre-registered) so the effect on planning is visible; rotation goals are kept, not restricted.
- **Block angle is barely in z (from `results/ablation_wm_fixes.json`; Phase 5b's latent judge, a nonlinear MLP, replaces this ridge-based reading):** the ridge probe on *real* val latents gets 80–91° mean angle error in every row (a uniform random guess gives ~90°), while block position is 6.5–11.8 px. Imagined angle beats copy-last only because copy-last is near random too. This matters for rotation-heavy goals in Phase 6. The Phase 4 CNN probe and the Phase 5 rollout eval (probe on decoded frames) will show whether the angle is non-linearly present or missing; revisit before planning.
- Kaggle v1 val loss is almost all variance hinge (0.336 of 0.346): val latents are far less spread per dimension than train ones. `wm_ablate` reports z_std per row; revisit after it.
- Getting `results/ablation_wm_fixes.json` into the repo: the Kaggle run writes it to `/kaggle/working/pusht-latent-world-model/results/` and to the W&B artifact `results-wm-ablate`; Aditya downloads that one file from the Kaggle output and drops it in `results/`.
- The local smoke filmstrip's decoded row is still a blur (50 steps, batch 4, CPU); judge the decoder on the Kaggle full run.
- Local `projected_30k_steps` (2h07m) is CPU at batch 4; the real projection comes from the Kaggle smoke.
- **Phase 4 Kaggle budget:** `train_baselines` full = wm_1step 30k + pixel 30k + BC 20k (batch 256 × 4 images) + probe 10k steps in one group. The Kaggle smoke's four `projected_*` lines tell us whether it fits one 12 h session; if not, rerunning the same cell skips finished jobs and resumes the unfinished one from its last checkpoint.
- BC smoke `val_z_std` = 0.006 (50 CPU steps). No CHECK by design; watch it on the Kaggle full run.
- **Phase 6:** the goal frame also shows the pusher at goal time, and z_g carries whatever pusher information E_wm keeps, so the cost may also pull the pusher toward its goal-time position. Known; the cost stays image-only as pre-registered.
- **Phase 6 T4 estimates (from FLOP counts, not measured; the Kaggle smoke's timing lines replace them):** train_bc_clean ~15–30 min (as train_bc); plan_val without cem_pixel ~5–10 min; cem_pixel +10–30 min for 50 pairs; plan_sweep ~10–20 min; `phase6` full ~0.5–1.5 h.
- **Mac only:** on import, macOS prints `objc ... Class SDL... is implemented in both cv2/.dylibs/libSDL2 and pygame/.dylibs/libSDL2` to stderr, because both OpenCV wheels bundle SDL2 on macOS. It's harmless with `SDL_VIDEODRIVER=dummy` and all tests pass. It doesn't occur on Kaggle/Colab (Linux).

## Last REPORTs (local smoke, `phase6` group; smoke numbers mean nothing)
The Phase 5b local smoke REPORTs are in commit 0382f7a's PROGRESS.md.
GROUP phase6: OK (4 of 4 passed or skipped) | PASS make_pairs, train_bc_clean, plan_val, plan_sweep
```
==================== RUN REPORT ====================
job: make_pairs   mode: smoke   platform: local   device: cpu
git: 0382f7a-dirty   time: 0.1s   params: n/a
CHECKS
 [PASS] pairs_val: 3 of 13 candidates from 1 episodes (train episode seed 1 (smoke: no val episodes)) (== 3)
 [PASS] pairs_test: 3 of 12 candidates from 1 episodes (test episodes (pusht_test.npz)) (== 3)
METRICS
 val_pairs = 3: rotation (|turn| >= 20 deg) 3, translation 0; median move 95.7 px, median |turn| 60.0 deg
 test_pairs = 3: rotation (|turn| >= 20 deg) 3, translation 0; median move 68.0 px, median |turn| 63.2 deg
 data = pusht-data-smoke
SAVED
 outputs/make_pairs/artifact/eval_pairs_val.json
 outputs/make_pairs/artifact/eval_pairs_test.json
 store/eval-pairs-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: train_bc_clean   mode: smoke   platform: local   device: cpu
git: 0382f7a-dirty   time: 9.2s   params: 2,528,482
CHECKS
 [PASS] param_counts: encoder 1,738,976 / mlp 789,506 / total 2,528,482 (== CLAUDE.md §6 (≈2.53M))
 [PASS] losses_finite: all finite (no NaN/inf)
 [PASS] loss_fell: 0.1331 -> 0.1128 (mean of last 10 < first 10 steps)
 [PASS] checkpoint_roundtrip: actions: weights identical, max |diff| 0.00e+00 (rel 0.00e+00) (model.pt weights bit-identical; outputs allclose rtol 0.0001, atol 1e-05)
 [PASS] val_mse_vs_mean_action: 0.1212 vs 0.1212 (val MSE < constant mean-action baseline; WARN only in smoke)
METRICS
 data = pusht-data-smoke
 steps = 50 (batch 8)
 resumed_from_step = 0
 train_samples = 200
 val_source = train episode seed 1 (smoke: no val episodes)
 val_samples = 64
 actions = clean
 loss_first_last = 0.1331 -> 0.1128
 it_per_s = 11.15
 projected_20k_steps = 29m54s
 peak_gpu_mem = n/a (cpu)
 val_mse = 0.1212 (mean-action baseline 0.1212)
 val_action_err_px = 24.2 (mean-action baseline 24.4)
 val_z_std = 0.025 (info only; no variance hinge)
 lr_final = 3.65e-07
SAVED
 outputs/train_bc_clean/artifact/model.pt
 outputs/train_bc_clean/artifact/train_summary.json
 outputs/train_bc_clean/artifact/config.yaml
 store/ckpt-bc-clean-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: plan_val   mode: smoke   platform: local   device: cpu
git: 0382f7a-dirty   time: 1m31s   params: wm 4,186,053 / wm_1step 4,186,053 / bc 2,528,482 / bc_clean 2,528,482 / pixel 4,813,731
CHECKS
 [PASS] ckpt_config_hash_wm: ckpt-wm-smoke step 50: d4c8e342b0d6 (== d4c8e342b0d6, current config; complete: true)
 [PASS] ckpt_config_hash_wm_1step: ckpt-wm-1step-smoke step 50: dd5934291cbe (== dd5934291cbe, current config; complete: true)
 [PASS] ckpt_config_hash_bc: ckpt-bc-smoke step 50: dd9930d91fd0 (== dd9930d91fd0, current config; complete: true)
 [PASS] ckpt_config_hash_bc_clean: ckpt-bc-clean-smoke step 50: a4cd0cd8aef5 (== a4cd0cd8aef5, current config; complete: true)
 [PASS] ckpt_config_hash_pixel: ckpt-pixel-smoke step 50: 2438d918b486 (== 2438d918b486, current config; complete: true)
 [PASS] eval_pairs_hash: eval-pairs-smoke: 10a77495b6db (== 10a77495b6db (make_pairs current config))
 [PASS] pairs: 3 val pairs (train episode seed 1 (smoke: no val episodes)) (== 3)
 [PASS] all_methods_ran: cem_wm 3, cem_wm_1step 3, bc 3, bc_clean 3, random 3, noop 3, cem_pixel 3 (3 episodes each)
 [PASS] render_alignment_start: mean abs diff 0.000 (max 0.000) over 3 pairs (mean <= 0.5 uint8 levels: rendered start state == stored start frame)
 [PASS] render_alignment_goal: mean abs diff 0.000 (max 0.000) over 3 pairs (mean <= 0.5 uint8 levels: rendered goal state == stored goal frame)
 [PASS] noop_success_zero: 0/3 (== 0 (impossible by the pair filter; > 0 means a bug))
 [PASS] metrics_finite: all finite (no NaN/inf)
 [WARN] cem_wm_vs_random: 0 vs 0 successes (cem_wm > random; WARN only (a result, not a bug))
METRICS
 ckpt_wm = ckpt-wm-smoke step 50, git 3c6f9a2-dirty, config_hash d4c8e342b0d6, proprio True
 ckpt_wm_1step = ckpt-wm-1step-smoke step 50, git 3c6f9a2-dirty, config_hash dd5934291cbe, proprio True
 ckpt_bc = ckpt-bc-smoke step 50, git 3c6f9a2-dirty, config_hash dd9930d91fd0
 ckpt_bc_clean = ckpt-bc-clean-smoke step 50, git 0382f7a-dirty, config_hash a4cd0cd8aef5
 ckpt_pixel = ckpt-pixel-smoke step 50, git 3c6f9a2-dirty, config_hash 2438d918b486
 data = pusht-data-smoke
 pairs = 3 val pairs from eval-pairs-smoke (train episode seed 1 (smoke: no val episodes)): rotation 3, translation 0; budget 10; seed 0
 cem = N 128, H 10, rounds 4, K 12, beta 0.1, sigma 30 (floor 2), clip 50 px; cem_pixel episodes at N 16, rounds 2 (local smoke)
 success_cem_wm = 0/3 = 0.0% [0.0, 56.1]
 pos_success_cem_wm = 0/3 = 0.0% [0.0, 56.1] (start >= 20 px) | all pairs 0/3
 groups_cem_wm = rotation (3 pairs): success 0/3, position-only 0/3 | translation (0 pairs): success 0/0, position-only 0/0
 final_err_cem_wm = median pos 94.9 px / angle 60.0 deg; mean steps-to-success n/a; 0.38 s/episode
 success_cem_wm_1step = 0/3 = 0.0% [0.0, 56.1]
 pos_success_cem_wm_1step = 0/3 = 0.0% [0.0, 56.1] (start >= 20 px) | all pairs 0/3
 groups_cem_wm_1step = rotation (3 pairs): success 0/3, position-only 0/3 | translation (0 pairs): success 0/0, position-only 0/0
 final_err_cem_wm_1step = median pos 94.7 px / angle 60.0 deg; mean steps-to-success n/a; 0.37 s/episode
 success_bc = 0/3 = 0.0% [0.0, 56.1]
 pos_success_bc = 0/3 = 0.0% [0.0, 56.1] (start >= 20 px) | all pairs 0/3
 groups_bc = rotation (3 pairs): success 0/3, position-only 0/3 | translation (0 pairs): success 0/0, position-only 0/0
 final_err_bc = median pos 91.1 px / angle 58.4 deg; mean steps-to-success n/a; 0.05 s/episode
 success_bc_clean = 0/3 = 0.0% [0.0, 56.1]
 pos_success_bc_clean = 0/3 = 0.0% [0.0, 56.1] (start >= 20 px) | all pairs 0/3
 groups_bc_clean = rotation (3 pairs): success 0/3, position-only 0/3 | translation (0 pairs): success 0/0, position-only 0/0
 final_err_bc_clean = median pos 93.4 px / angle 58.4 deg; mean steps-to-success n/a; 0.05 s/episode
 success_random = 0/3 = 0.0% [0.0, 56.1]
 pos_success_random = 0/3 = 0.0% [0.0, 56.1] (start >= 20 px) | all pairs 0/3
 groups_random = rotation (3 pairs): success 0/3, position-only 0/3 | translation (0 pairs): success 0/0, position-only 0/0
 final_err_random = median pos 83.9 px / angle 57.7 deg; mean steps-to-success n/a; 0.01 s/episode
 success_noop = 0/3 = 0.0% [0.0, 56.1]
 pos_success_noop = 0/3 = 0.0% [0.0, 56.1] (start >= 20 px) | all pairs 0/3
 groups_noop = rotation (3 pairs): success 0/3, position-only 0/3 | translation (0 pairs): success 0/0, position-only 0/0
 final_err_noop = median pos 95.6 px / angle 59.9 deg; mean steps-to-success n/a; 0.01 s/episode
 success_cem_pixel = 0/3 = 0.0% [0.0, 56.1]
 pos_success_cem_pixel = 0/3 = 0.0% [0.0, 56.1] (start >= 20 px) | all pairs 0/3
 groups_cem_pixel = rotation (3 pairs): success 0/3, position-only 0/3 | translation (0 pairs): success 0/0, position-only 0/0
 final_err_cem_pixel = median pos 93.5 px / angle 60.0 deg; mean steps-to-success n/a; 19.21 s/episode
 bc_selected = bc (median final pos px: bc 91.0617 vs bc_clean 93.4133); success bc 0/3 = 0.0% [0.0, 56.1] | bc_clean 0/3 = 0.0% [0.0, 56.1]
 timing_cem_wm = 0.046 s per full-size planning call (N 128, 4 rounds, H 10; mean of 1) + env 1.8 ms/step -> projected 1.9 s/episode, 1.6 min for 50 pairs (upper bound)
 timing_cem_wm_1step = 0.034 s per full-size planning call (N 128, 4 rounds, H 10; mean of 1) + env 1.7 ms/step -> projected 1.4 s/episode, 1.2 min for 50 pairs (upper bound)
 timing_cem_pixel = 29.799 s per full-size planning call (N 128, 4 rounds, H 10; mean of 1) + env 3.3 ms/step -> projected 1192.1 s/episode, 993.4 min for 50 pairs (upper bound)
 tau = 15.3833 = p95 of ||E(o_i) - E(o_j)||^2 over 721 same-episode val frame pairs that are successes (median 1.5220; non-success median 404.5698) [train episode seed 1 (smoke: no val episodes)]
 imagined_success_cem_wm = 0/3 episodes reached terminal cost < tau; 0 of them failed for real
 imagined_success_cem_wm_1step = 0/3 episodes reached terminal cost < tau; 0 of them failed for real
 imagined_success_cem_pixel = 1/3 episodes reached terminal cost < tau; 1 of them failed for real
SAVED
 outputs/plan_val/artifact/plan_val.json
 outputs/plan_val/artifact/filmstrips
 store/results-plan-val-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: plan_sweep   mode: smoke   platform: local   device: cpu
git: 0382f7a-dirty   time: 11.6s   params: wm 4,186,053
CHECKS
 [PASS] ckpt_config_hash_wm: ckpt-wm-smoke step 50: d4c8e342b0d6 (== d4c8e342b0d6, current config; complete: true)
 [PASS] eval_pairs_hash: eval-pairs-smoke: 10a77495b6db (== 10a77495b6db (make_pairs current config))
 [PASS] pairs: 3 val pairs (train episode seed 1 (smoke: no val episodes)) (== 3)
 [PASS] all_settings_ran: 9 settings (3 episodes each)
 [PASS] metrics_finite: all finite (no NaN/inf)
METRICS
 ckpt_wm = ckpt-wm-smoke step 50, git 3c6f9a2-dirty, config_hash d4c8e342b0d6, proprio True
 data = pusht-data-smoke
 pairs = 3 val pairs (train episode seed 1 (smoke: no val episodes)); budget 10; seed 0; cem_wm only
 setting_default = success 0/3 = 0.0% [0.0, 56.1] | position-only 0/3 (start >= 20 px) | median pos 94.9 px | 0.47 s/ep | cost 5120
 setting_N=32 = success 0/3 = 0.0% [0.0, 56.1] | position-only 0/3 (start >= 20 px) | median pos 95.1 px | 0.48 s/ep | cost 1280
 setting_N=64 = success 0/3 = 0.0% [0.0, 56.1] | position-only 0/3 (start >= 20 px) | median pos 95.6 px | 0.32 s/ep | cost 2560
 setting_N=256 = success 0/3 = 0.0% [0.0, 56.1] | position-only 0/3 (start >= 20 px) | median pos 95.2 px | 0.67 s/ep | cost 10240
 setting_rounds=1 = success 0/3 = 0.0% [0.0, 56.1] | position-only 0/3 (start >= 20 px) | median pos 95.7 px | 0.17 s/ep | cost 1280
 setting_rounds=2 = success 0/3 = 0.0% [0.0, 56.1] | position-only 0/3 (start >= 20 px) | median pos 94.9 px | 0.26 s/ep | cost 2560
 setting_H=5 = success 0/3 = 0.0% [0.0, 56.1] | position-only 0/3 (start >= 20 px) | median pos 94.7 px | 0.25 s/ep | cost 2560
 setting_H=15 = success 0/3 = 0.0% [0.0, 56.1] | position-only 0/3 (start >= 20 px) | median pos 95.4 px | 0.65 s/ep | cost 7680
 setting_beta=0 = success 0/3 = 0.0% [0.0, 56.1] | position-only 0/3 (start >= 20 px) | median pos 95.7 px | 0.49 s/ep | cost 5120
 selected = default by pos_success (no variant has >= 3 more pos_success than default (0))
 frozen_cem = candidates 128, horizon 10, rounds 4, elites 12, delta_clip 50.0, sigma_init 30.0, sigma_floor 2.0, beta 0.1, warm_start True
SAVED
 outputs/plan_sweep/artifact/plan_sweep.json
 store/results-plan-sweep-smoke
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

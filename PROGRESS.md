# PROGRESS

**Current phase: 5b (angle audit and fair judges), built; local smoke PASS → Kaggle.** Kaggle `eval_rollouts` full (adc200d) showed the v1 probe is a broken judge (findings below), so Phase 5b adds an angle audit, two new judges (`probe_mix` on images, a latent MLP on E_wm) and reruns `eval_rollouts` with both. Its judge decisions are pre-registered below. Next: Aditya pushes, then runs Kaggle `phase5b` smoke → full and drops `results/angle_audit.json` and `results/rollouts.json` into `results/`.

## Status
| Phase | Status | Local smoke | Kaggle smoke | Kaggle full |
| --- | --- | --- | --- | --- |
| 1 Foundation | done | PASS | PASS | n/a |
| 2 Expert + data | **complete** | PASS | PASS | PASS |
| 3 World model | v1 done (`ckpt-wm:v5`); A+B retrain in `overnight` | PASS | pending | pending (A+B) |
| 3b WM fixes | **complete** (A+B adopted) | PASS | PASS | PASS |
| 4 Baselines | done locally | PASS | pending | pending |
| 5 Rollout eval | Kaggle full done (adc200d); judge failed → 5b | PASS | PASS | PASS (`results/rollouts.json`) |
| 5b Angle audit + judges | built; local smoke PASS | PASS | pending | pending |
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

## Installed versions (local)
python 3.12.2, torch 2.15.0, gym-pusht 0.1.8, gymnasium 1.4.0, pymunk 6.11.1, pygame 2.6.1, opencv-python(-headless) 4.14.0.94, numpy 2.5.5, wandb 0.31.2.

## Open issues
- **Phase 5b open question (answered by the Kaggle `phase5b` full run):** is block angle in the wm latent? See `angle_in_wm_latent` in the `train_latent_probe` and `eval_rollouts` REPORTs (pre-registered thresholds: < 20° present, 20–45° weak, > 45° absent). This decides whether Phase 6 can attempt rotation goals. Labels are verified consistent (no bug), so a large error there is a property of the model or the judge, not the data.
- **Block angle is barely in z (from `results/ablation_wm_fixes.json`; Phase 5b's latent judge, a nonlinear MLP, replaces this ridge-based reading):** the ridge probe on *real* val latents gets 80–91° mean angle error in every row (a uniform random guess gives ~90°), while block position is 6.5–11.8 px. Imagined angle beats copy-last only because copy-last is near random too. This matters for rotation-heavy goals in Phase 6. The Phase 4 CNN probe and the Phase 5 rollout eval (probe on decoded frames) will show whether the angle is non-linearly present or missing; revisit before planning.
- Kaggle v1 val loss is almost all variance hinge (0.336 of 0.346): val latents are far less spread per dimension than train ones. `wm_ablate` reports z_std per row; revisit after it.
- Getting `results/ablation_wm_fixes.json` into the repo: the Kaggle run writes it to `/kaggle/working/pusht-latent-world-model/results/` and to the W&B artifact `results-wm-ablate`; Aditya downloads that one file from the Kaggle output and drops it in `results/`.
- The local smoke filmstrip's decoded row is still a blur (50 steps, batch 4, CPU); judge the decoder on the Kaggle full run.
- Local `projected_30k_steps` (2h07m) is CPU at batch 4; the real projection comes from the Kaggle smoke.
- **Phase 4 Kaggle budget:** `train_baselines` full = wm_1step 30k + pixel 30k + BC 20k (batch 256 × 4 images) + probe 10k steps in one group. The Kaggle smoke's four `projected_*` lines tell us whether it fits one 12 h session; if not, rerunning the same cell skips finished jobs and resumes the unfinished one from its last checkpoint.
- BC smoke `val_z_std` = 0.006 (50 CPU steps). No CHECK by design; watch it on the Kaggle full run.
- **Mac only:** on import, macOS prints `objc ... Class SDL... is implemented in both cv2/.dylibs/libSDL2 and pygame/.dylibs/libSDL2` to stderr, because both OpenCV wheels bundle SDL2 on macOS. It's harmless with `SDL_VIDEODRIVER=dummy` and all tests pass. It doesn't occur on Kaggle/Colab (Linux).

## Last REPORTs (local smoke, `phase5b` group, commit 8ea9498; smoke numbers mean nothing)
GROUP phase5b: OK (5 of 5 passed or skipped) | PASS angle_audit, render_judge_data, train_probe_mix, train_latent_probe, eval_rollouts
```
==================== RUN REPORT ====================
job: angle_audit   mode: smoke   platform: local   device: cpu
git: a343e14-dirty   time: 15.2s   params: probe 1,740,518 / wm 4,186,053
CHECKS
 [PASS] ckpt_config_hash_probe: ckpt-probe-smoke step 50: 981a0e656148 (== 981a0e656148, current config; complete: true)
 [PASS] ckpt_config_hash_wm: ckpt-wm-smoke step 50: d4c8e342b0d6 (== d4c8e342b0d6, current config; complete: true)
 [PASS] ckpt_config_hash_wm_1step: ckpt-wm-1step-smoke step 50: dd5934291cbe (== dd5934291cbe, current config; complete: true)
 [PASS] ckpt_config_hash_pixel: ckpt-pixel-smoke step 50: 2438d918b486 (== 2438d918b486, current config; complete: true)
 [PASS] rerender_consistency: mean abs diff 0.000 over 50 stored frames (100% exact) (<= 0.5 uint8 levels)
 [PASS] angle_orientation: mask-moment angle vs state angle: mean 1.7, max 8.4 deg over 288 poses (mean <= 10 deg)
 [PASS] cog_convention: mask centroid to block_cog 4.7 px vs to body origin 40.3 px (centroid nearer block_cog on every pose)
 [PASS] target_roundtrip: max 2.32e-06 deg / 8.20e-06 px; wrap 359v1, -179v179, 540v180 -> 2.0, 2.0, 0.0 (0 / 0; wraps 2, 2, 0 deg)
 [PASS] metrics_finite: all finite (no NaN/inf)
METRICS
 verdict = labels consistent; failure = probe generalisation (memorisation: train angle << test)
 ckpt_probe = ckpt-probe-smoke step 50, git 3c6f9a2-dirty
 ckpt_wm = ckpt-wm-smoke step 50, git 3c6f9a2-dirty, proprio True
 convention = radians wrapped to [-pi, pi); 0 = bar on top, stem down; angle grows clockwise on screen (y down); stored xy = body origin, CoG = origin + R(0, 45)
 template_oracle_real = angle 1.0 deg mean / 0.8 median, 100% under 10 deg (10 test frames, true position given)
 template_oracle_wm_recon = angle 49.9 deg mean / 38.7 median, 10% under 10 deg (10 test frames, true position given)
 template_oracle_wm1step_recon = angle 92.4 deg mean / 103.9 median, 0% under 10 deg (10 test frames, true position given)
 template_oracle_pixel_1step = angle 3.5 deg mean / 2.0 median, 90% under 10 deg (10 test frames, true position given)
 template_oracle_note = non-learned reader given the true position: real = does a 64x64 frame hold the angle; derived types = is it still in that frame type
 probe_v1_train = block 15.9 px / 8.4 deg / pusher 37.6 px (200 frames; predict-mean 48.9 deg)
 probe_v1_val = block 17.6 px / 10.1 deg / pusher 40.4 px (101 frames; predict-mean 61.6 deg)
 probe_v1_test = block 104.2 px / 103.2 deg / pusher 161.6 px (101 frames; predict-mean 82.1 deg)
 probe_v1_grid = block 69.4 px / 89.9 deg / pusher 341.9 px (288 frames; predict-mean 90.0 deg)
 memorisation = train angle 8.4 vs test 103.2 deg -> yes (train < 0.5 x test)
 data = pusht-data-smoke
SAVED
 outputs/angle_audit/artifact/angle_audit.json
 outputs/angle_audit/artifact/angle_grid.png
 store/results-angle-audit-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: render_judge_data   mode: smoke   platform: local   device: cpu
git: a343e14-dirty   time: 0.6s   params: n/a
CHECKS
 [PASS] frame_counts: train 144 / val 16 frames (36 / 4 micro-episodes) (== 4 x micro-episodes; 160 total)
 [PASS] angles_uniform: resultant R 0.186 (circular std 105 deg) (R < 0.1 (full only))
METRICS
 frames_total = 160
 rejection = mean 1.08 draws per valid start (max 2)
 block_cog_coverage_8x8 = 0.344
 seeds = 5000-5039 (val = first 4)
 seconds = 0.5
SAVED
 outputs/render_judge_data/artifact/judge_train.npz
 outputs/render_judge_data/artifact/judge_val.npz
 outputs/render_judge_data/artifact/judge_micro_episodes.png
 outputs/render_judge_data/artifact/judge_data.json
 store/judge-data-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: train_probe_mix   mode: smoke   platform: local   device: cpu
git: a343e14-dirty   time: 45.7s   params: 1,740,518
CHECKS
 [PASS] param_counts: encoder 1,738,976 / head 1,542 / total 1,740,518 (== CLAUDE.md §7 (flatten))
 [PASS] param_counts_coord_gap: encoder 691,168 / head 1,542 / total 692,710 (== documented coord_gap)
 [PASS] ckpt_config_hash_wm: ckpt-wm-smoke step 50: d4c8e342b0d6 (== d4c8e342b0d6, current config; complete: true)
 [PASS] ckpt_config_hash_wm_1step: ckpt-wm-1step-smoke step 50: dd5934291cbe (== dd5934291cbe, current config; complete: true)
 [PASS] ckpt_config_hash_pixel: ckpt-pixel-smoke step 50: 2438d918b486 (== 2438d918b486, current config; complete: true)
 [PASS] losses_finite: all finite (no NaN/inf)
 [PASS] loss_fell: 0.5975 -> 0.2225 (mean of last 10 < first 10 steps)
 [PASS] checkpoint_roundtrip: outputs: weights identical, max |diff| 0.00e+00 (rel 0.00e+00) (model.pt weights bit-identical; outputs allclose rtol 0.0001, atol 1e-05)
 [WARN] accept_real_angle: 79.1 deg (< 15 deg on real test frames; WARN only)
 [WARN] accept_real_block: 123.8 px (<= 13.4 px (1.25 x v1 probe); WARN only)
 [PASS] accept_wm_recon: block 124.6 / pusher 165.1 px vs real 123.8 / 165.9 (<= 1.5 x real; WARN only)
 [PASS] accept_wm1step_recon: block 129.3 / pusher 169.7 px vs real 123.8 / 165.9 (<= 1.5 x real; WARN only)
 [PASS] accept_pixel_1step: block 123.5 / pusher 165.8 px vs real 123.8 / 165.9 (<= 1.5 x real; WARN only)
METRICS
 judge = flatten (NOT accepted); fallback triggered
 data = pusht-data-smoke + judge-data-smoke: 346 train frames (real train split + rendered); rendered val 16
 steps = 50 (batch 16, best step 50)
 resumed_from_step = 0
 loss_first_last = 0.5975 -> 0.2225
 it_per_s = 5.98
 projected_20k_steps = 55m44s
 peak_gpu_mem = n/a (cpu)
 test_block_px_flatten = real 123.8 | wm_recon 124.6 | wm1step_recon 129.3 | pixel_1step 123.5 px
 test_block_deg_flatten = real 79.1 | wm_recon 71.7 | wm1step_recon 71.2 | pixel_1step 78.8 deg
 test_pusher_px_flatten = real 165.9 | wm_recon 165.1 | wm1step_recon 169.7 | pixel_1step 165.8 px
 real_val_flatten = block 104.9 px / 52.7 deg / pusher 129.4 px
 grid_flatten = block 81.5 px / 89.9 deg / pusher 316.2 px (288 rendered poses)
 test_block_px_coord_gap = real 131.3 | wm_recon 135.1 | wm1step_recon 135.7 | pixel_1step 132.1 px
 test_block_deg_coord_gap = real 83.5 | wm_recon 82.7 | wm1step_recon 82.6 | pixel_1step 82.8 deg
 test_pusher_px_coord_gap = real 177.0 | wm_recon 176.5 | wm1step_recon 177.0 | pixel_1step 178.3 px
 real_val_coord_gap = block 118.6 px / 60.9 deg / pusher 154.7 px
 grid_coord_gap = block 92.4 px / 90.0 deg / pusher 300.4 px (288 rendered poses)
 lr_final = 3.65e-07
SAVED
 outputs/train_probe_mix/artifact/model.pt
 outputs/train_probe_mix/artifact/train_summary.json
 outputs/train_probe_mix/artifact/config.yaml
 store/ckpt-probe-mix-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: train_latent_probe   mode: smoke   platform: local   device: cpu
git: a343e14-dirty   time: 8.3s   params: 397,318
CHECKS
 [PASS] param_counts: total 397,318 (== documented latent MLP)
 [PASS] ckpt_config_hash_wm: ckpt-wm-smoke step 50: d4c8e342b0d6 (== d4c8e342b0d6, current config; complete: true)
 [PASS] losses_finite: all finite (no NaN/inf)
 [PASS] loss_fell: 0.2092 -> 0.1055 (mean of last 10 < first 10 steps)
 [PASS] checkpoint_roundtrip: outputs: weights identical, max |diff| 0.00e+00 (rel 0.00e+00) (model.pt weights bit-identical; outputs allclose rtol 0.0001, atol 1e-05)
 [PASS] real_test_block_vs_mean: 49.4 vs 142.0 px (block error on real E_wm(o) test < predict-train-mean; WARN only in smoke)
METRICS
 angle_in_wm_latent = 135.2 deg -> absent (Phase 6 must restrict or flag rotation goals) (latent MLP on real E_wm(o) test frames; predict-mean 82.1 deg; thresholds < 20 present, > 45 absent)
 ckpt_wm = ckpt-wm-smoke step 50, git 3c6f9a2-dirty, config_hash d4c8e342b0d6, proprio True
 data = pusht-data-smoke + judge-data-smoke: 346 train latents; rendered val 16
 steps = 50 (batch 64, best step 25)
 resumed_from_step = 0
 loss_first_last = 0.2092 -> 0.1055
 it_per_s = 243.37
 projected_10k_steps = 41.1s
 peak_gpu_mem = n/a (cpu)
 real_train = block 24.6 px / 14.5 deg / pusher 67.3 px (202 frames; predict-mean block 105.2 px / 48.8 deg / pusher 118.9 px)
 real_val = block 25.7 px / 23.4 deg / pusher 84.1 px (101 frames; predict-mean block 120.5 px / 61.6 deg / pusher 132.6 px)
 real_test = block 49.4 px / 135.2 deg / pusher 158.2 px (101 frames; predict-mean block 142.0 px / 82.1 deg / pusher 164.4 px)
 rendered_val = block 61.0 px / 140.1 deg / pusher 125.9 px (16 frames; predict-mean block 134.1 px / 82.2 deg / pusher 197.8 px)
 grid = block 46.3 px / 88.7 deg / pusher 300.2 px (288 frames; predict-mean block 91.4 px / 90.0 deg / pusher 351.5 px)
 lr_final = 1.22e-06
SAVED
 outputs/train_latent_probe/artifact/model.pt
 outputs/train_latent_probe/artifact/train_summary.json
 outputs/train_latent_probe/artifact/config.yaml
 store/ckpt-latent-probe-smoke
W&B: off
====================================================
==================== RUN REPORT ====================
job: eval_rollouts   mode: smoke   platform: local   device: cpu
git: a343e14-dirty   time: 38.2s   params: wm 4,186,053 / wm_1step 4,186,053 / pixel 4,813,731 / probe_v1 1,740,518 / probe_mix 1,740,518 / latent 397,318
CHECKS
 [PASS] ckpt_config_hash_wm: ckpt-wm-smoke step 50: d4c8e342b0d6 (== d4c8e342b0d6, current config; complete: true)
 [PASS] ckpt_config_hash_wm_1step: ckpt-wm-1step-smoke step 50: dd5934291cbe (== dd5934291cbe, current config; complete: true)
 [PASS] ckpt_config_hash_pixel: ckpt-pixel-smoke step 50: 2438d918b486 (== 2438d918b486, current config; complete: true)
 [PASS] ckpt_config_hash_probe_v1: ckpt-probe-smoke step 50: 981a0e656148 (== 981a0e656148, current config; complete: true)
 [PASS] ckpt_config_hash_probe_mix: ckpt-probe-mix-smoke step 50: 1571bba42879 (== 1571bba42879, current config; complete: true)
 [PASS] ckpt_config_hash_latent: ckpt-latent-probe-smoke step 50: 0d92e89daece (== 0d92e89daece, current config; complete: true)
 [PASS] metrics_finite: all finite (no NaN/inf)
 [PASS] windows: 20 of 89 test windows (== 20)
 [PASS] judge_floor_probe_mix: block 123.8 px vs predict-mean 142.0 (judge on all real test frames < predict-train-mean; WARN only in smoke)
 [WARN] probe_angle_floor_probe_mix: 79.1 deg - angle columns unreliable for this judge (< 30 deg mean on all real test frames; WARN only)
 [PASS] judge_floor_latent: block 49.4 px vs predict-mean 142.0 (judge on all real test frames < predict-train-mean; WARN only in smoke)
 [WARN] probe_angle_floor_latent: 135.2 deg - angle columns unreliable for this judge (< 30 deg mean on all real test frames; WARN only)
 [WARN] judge_accepted_probe_mix: NOT accepted (rows unreliable) (pre-registered acceptance in train_probe_mix; WARN only (no headline claim without it))
 [WARN] nmse_wm_vs_copy_last_all_k: wm_via_frames < copy_last at 0 of 10 k (direct z_hat: 0 of 10) (all k; WARN only (a result, not a bug))
 [WARN] block_px_wm_vs_copy_last_k10_probe_mix: 124.9 vs 124.6 px (wm < copy_last; WARN only)
 [PASS] block_px_wm_vs_copy_last_k10_latent: 55.8 vs 63.0 px (wm < copy_last; WARN only)
METRICS
 ckpt_wm = ckpt-wm-smoke step 50, git 3c6f9a2-dirty, config_hash d4c8e342b0d6, proprio True
 ckpt_wm_1step = ckpt-wm-1step-smoke step 50, git 3c6f9a2-dirty, config_hash dd5934291cbe, proprio True
 ckpt_pixel = ckpt-pixel-smoke step 50, git 3c6f9a2-dirty, config_hash 2438d918b486
 ckpt_probe_v1 = ckpt-probe-smoke step 50, git 3c6f9a2-dirty, config_hash 981a0e656148
 ckpt_probe_mix = ckpt-probe-mix-smoke step 50, git a343e14-dirty, config_hash 1571bba42879
 ckpt_latent = ckpt-latent-probe-smoke step 50, git a343e14-dirty, config_hash 0d92e89daece
 data = pusht-data-smoke
 windows = 20 test windows of 13 frames (seed 0); bootstrap 1000 paired resamples, 95% CI
 angle_in_wm_latent = 135.2 deg -> absent (Phase 6 must restrict or flag rotation goals) | template oracle 0.9 deg (10 real test frames, true position given) | predict-mean 82.1 deg
 judge_probe_v1 = deprecated (judge failure); all real test frames: block 104.2 px / 103.2 deg / pusher 161.6 px
 judge_probe_mix = NOT accepted (rows unreliable), pool flatten; all real test frames: block 123.8 px / 79.1 deg / pusher 165.9 px
 judge_latent = reported (no gate); all real test frames: block 49.4 px / 135.2 deg / pusher 158.2 px
 grid_angle_deg = probe_v1 89.9 (before) | probe_mix 89.9 | latent 88.7 (288 rendered poses)
 note = probe_mix rows real / wm_recon / wm1step_recon and latent row real = floors; latent 'wm' = direct z_hat, 'wm_via_frames' = E_wm(D_wm(z_hat)); headline NMSE = wm_via_frames
 probe_mix_block_px_k1 = wm 126.0 [111.1, 140.4] | wm_1step 131.2 [115.2, 146.9] | pixel 126.5 [115.9, 137.4] | copy_last 126.9 [116.4, 137.6] | real 126.9 [116.4, 137.6] | wm_recon 127.4 [112.5, 142.1] | wm1step_recon 131.8 [115.9, 147.5]
 probe_mix_block_px_k5 = wm 124.4 [110.6, 138.2] | wm_1step 130.3 [114.9, 145.6] | pixel 123.6 [113.1, 134.3] | copy_last 125.3 [115.4, 135.5] | real 125.3 [115.0, 135.7] | wm_recon 126.6 [111.6, 141.4] | wm1step_recon 131.6 [115.6, 147.4]
 probe_mix_block_px_k10 = wm 124.9 [113.9, 135.3] | wm_1step 132.3 [118.8, 145.3] | pixel 121.1 [110.5, 131.8] | copy_last 124.6 [116.0, 133.2] | real 124.6 [115.4, 133.6] | wm_recon 127.3 [113.2, 140.6] | wm1step_recon 132.9 [117.6, 147.1]
 probe_mix_block_deg_k1 = wm 69.4 [47.2, 92.5] | wm_1step 67.2 [46.3, 89.1] | pixel 79.3 [54.8, 104.1] | copy_last 78.8 [54.4, 103.5] | real 78.8 [54.4, 103.5] | wm_recon 69.2 [47.1, 92.4] | wm1step_recon 67.1 [46.2, 89.0]
 probe_mix_block_deg_k5 = wm 66.0 [42.3, 90.8] | wm_1step 64.1 [40.8, 87.0] | pixel 75.9 [50.0, 100.9] | copy_last 73.2 [48.2, 98.8] | real 72.9 [47.8, 98.7] | wm_recon 64.5 [40.9, 87.7] | wm1step_recon 63.4 [40.8, 85.8]
 probe_mix_block_deg_k10 = wm 63.6 [39.7, 88.4] | wm_1step 60.6 [37.9, 84.3] | pixel 67.6 [41.1, 95.8] | copy_last 66.1 [41.0, 92.6] | real 66.3 [40.8, 92.9] | wm_recon 59.7 [37.0, 83.3] | wm1step_recon 60.6 [39.8, 82.0]
 probe_mix_pusher_px_k1 = wm 162.6 [137.0, 186.9] | wm_1step 167.0 [137.7, 194.5] | pixel 166.0 [145.2, 185.5] | copy_last 165.7 [144.5, 185.7] | real 165.8 [144.5, 185.8] | wm_recon 163.9 [137.5, 188.7] | wm1step_recon 167.6 [138.1, 195.3]
 probe_mix_pusher_px_k5 = wm 166.6 [143.1, 189.6] | wm_1step 172.7 [145.2, 199.4] | pixel 173.7 [155.9, 191.3] | copy_last 170.3 [150.4, 189.7] | real 170.5 [150.5, 189.9] | wm_recon 169.6 [144.0, 194.6] | wm1step_recon 174.1 [145.9, 201.7]
 probe_mix_pusher_px_k10 = wm 168.4 [148.0, 187.1] | wm_1step 177.5 [152.2, 201.3] | pixel 189.8 [168.8, 210.3] | copy_last 172.3 [153.2, 190.3] | real 172.4 [153.4, 190.6] | wm_recon 173.1 [148.6, 196.4] | wm1step_recon 178.8 [151.1, 204.8]
 latent_block_px_k1 = wm 57.3 [39.5, 76.9] | wm_via_frames 123.1 [104.6, 141.8] | wm_1step 156.1 [136.0, 175.4] | pixel 57.8 [40.1, 76.8] | copy_last 59.0 [40.4, 78.3] | real 57.1 [38.3, 76.7]
 latent_block_px_k5 = wm 53.1 [35.4, 71.3] | wm_via_frames 106.8 [90.0, 124.7] | wm_1step 145.0 [125.1, 165.0] | pixel 50.2 [35.5, 65.1] | copy_last 58.8 [41.2, 78.2] | real 49.7 [33.2, 67.8]
 latent_block_px_k10 = wm 55.8 [39.1, 72.6] | wm_via_frames 94.2 [81.4, 106.9] | wm_1step 136.1 [118.7, 153.9] | pixel 117.5 [102.6, 129.7] | copy_last 63.0 [45.7, 80.9] | real 39.2 [27.4, 53.7]
 latent_block_deg_k1 = wm 124.4 [104.4, 142.9] | wm_via_frames 105.9 [88.8, 123.5] | wm_1step 73.9 [56.2, 90.3] | pixel 122.2 [101.8, 141.5] | copy_last 121.8 [101.0, 141.2] | real 121.7 [101.3, 140.9]
 latent_block_deg_k5 = wm 146.9 [137.7, 155.2] | wm_via_frames 129.0 [117.1, 139.5] | wm_1step 84.4 [67.3, 101.5] | pixel 123.8 [102.8, 142.9] | copy_last 118.4 [96.4, 139.1] | real 119.0 [96.4, 140.2]
 latent_block_deg_k10 = wm 155.4 [148.6, 161.8] | wm_via_frames 141.2 [129.9, 151.2] | wm_1step 100.7 [84.9, 117.4] | pixel 109.6 [86.5, 132.2] | copy_last 112.3 [89.2, 134.7] | real 129.1 [111.3, 145.0]
 latent_pusher_px_k1 = wm 169.7 [144.0, 193.7] | wm_via_frames 184.9 [149.2, 217.6] | wm_1step 196.3 [156.9, 232.7] | pixel 171.2 [145.2, 195.8] | copy_last 170.8 [144.3, 195.7] | real 171.2 [144.8, 195.7]
 latent_pusher_px_k5 = wm 167.5 [146.0, 188.7] | wm_via_frames 186.4 [158.2, 214.0] | wm_1step 201.0 [166.5, 233.5] | pixel 175.8 [150.3, 200.2] | copy_last 173.7 [146.6, 200.2] | real 171.9 [145.7, 198.5]
 latent_pusher_px_k10 = wm 163.6 [147.1, 179.4] | wm_via_frames 185.4 [164.0, 205.5] | wm_1step 204.2 [174.3, 231.9] | pixel 188.4 [159.8, 213.2] | copy_last 174.8 [148.6, 200.7] | real 166.2 [140.7, 193.1]
 probe_v1_block_px_k10 = [deprecated (judge failure)] wm 82.9 | wm_1step 115.1 | pixel 107.9 | copy_last 111.5 | real 111.2 | wm_recon 105.2
 probe_v1_block_deg_k10 = [deprecated (judge failure)] wm 136.6 | wm_1step 87.7 | pixel 80.9 | copy_last 84.3 | real 89.1 | wm_recon 96.1
 probe_v1_pusher_px_k10 = [deprecated (judge failure)] wm 136.4 | wm_1step 156.7 | pixel 133.4 | copy_last 159.5 | real 161.9 | wm_recon 150.0
 p_hat_pusher_px_k1 = wm 8.7 [6.6, 10.9] | wm_1step 9.4 [7.4, 11.3] | copy_last 9.4 [7.8, 11.1]
 p_hat_pusher_px_k5 = wm 29.6 [22.4, 37.7] | wm_1step 33.1 [26.5, 40.1] | copy_last 38.3 [31.8, 43.8]
 p_hat_pusher_px_k10 = wm 56.6 [44.3, 69.5] | wm_1step 55.7 [42.3, 68.0] | copy_last 67.4 [56.6, 76.9]
 nmse_k1 = wm_via_frames 1.444 [1.337, 1.554] | wm_1step 1.986 [1.696, 2.258] | pixel 0.017 [0.012, 0.023] | copy_last 0.006 [0.004, 0.008] | wm 0.037 [0.031, 0.044] | wm_recon 1.510 [1.396, 1.626]
 nmse_k5 = wm_via_frames 1.597 [1.513, 1.681] | wm_1step 2.051 [1.775, 2.317] | pixel 0.789 [0.551, 1.029] | copy_last 0.103 [0.075, 0.135] | wm 0.691 [0.601, 0.777] | wm_recon 1.516 [1.378, 1.660]
 nmse_k10 = wm_via_frames 2.320 [2.177, 2.472] | wm_1step 2.460 [2.268, 2.640] | pixel 9.599 [8.064, 11.110] | copy_last 0.355 [0.254, 0.454] | wm 2.831 [2.503, 3.150] | wm_recon 1.576 [1.431, 1.713]
 pixel_mse_k1 = wm 0.03136 [0.03053, 0.03215] | wm_1step 0.03597 [0.03510, 0.03690] | pixel 0.00187 [0.00158, 0.00218] | copy_last 0.00053 [0.00040, 0.00067] | wm_recon 0.03361 [0.03284, 0.03440]
 pixel_mse_k5 = wm 0.03204 [0.03111, 0.03292] | wm_1step 0.03719 [0.03645, 0.03798] | pixel 0.04043 [0.03389, 0.04704] | copy_last 0.00253 [0.00182, 0.00326] | wm_recon 0.03405 [0.03330, 0.03487]
 pixel_mse_k10 = wm 0.03410 [0.03318, 0.03504] | wm_1step 0.04127 [0.04079, 0.04179] | pixel 0.21821 [0.17042, 0.26622] | copy_last 0.00355 [0.00268, 0.00442] | wm_recon 0.03457 [0.03365, 0.03572]
 nmse_reduction_vs_pixel = k5 -1.026 [-1.850, -0.590] | k10 0.758 [0.703, 0.798]
 nmse_reduction_vs_wm_1step = k5 0.221 [0.128, 0.289] | k10 0.057 [-0.042, 0.139]
 probe_mix_block_px_wm_minus_pixel = k5 0.8 [-2.6, 4.2] px | k10 3.8 [2.5, 5.2] px
 probe_mix_block_px_wm_minus_wm_1step = k5 -5.9 [-8.0, -3.7] px | k10 -7.4 [-10.1, -4.6] px
 probe_mix_block_px_wm_minus_copy_last = k5 -0.9 [-4.8, 2.9] px | k10 0.3 [-1.8, 2.3] px
 latent_block_px_wm_minus_pixel = k5 2.9 [-2.6, 7.8] px | k10 -61.7 [-76.1, -45.2] px
 latent_block_px_wm_minus_wm_1step = k5 -92.0 [-103.7, -81.8] px | k10 -80.3 [-88.6, -73.0] px
 latent_block_px_wm_minus_copy_last = k5 -5.8 [-9.6, -1.2] px | k10 -7.2 [-12.7, -1.4] px
 headline_wm_vs_pixel_k5 = no claim: image judge not accepted (image inconclusive, latent inconclusive)
 headline_wm_vs_pixel_k10 = no claim: image judge not accepted (image wm worse, latent wm better)
 headline_wm_vs_wm_1step_k5 = no claim: image judge not accepted (image wm better, latent wm better)
 headline_wm_vs_wm_1step_k10 = no claim: image judge not accepted (image wm better, latent wm better)
 headline_wm_vs_copy_last_k5 = no claim: image judge not accepted (image inconclusive, latent wm better)
 headline_wm_vs_copy_last_k10 = no claim: image judge not accepted (image inconclusive, latent wm better)
SAVED
 outputs/eval_rollouts/artifact/rollouts.json
 outputs/eval_rollouts/artifact/plots
 outputs/eval_rollouts/artifact/filmstrips
 store/results-rollouts-smoke
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

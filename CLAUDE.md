# CLAUDE.md — PushT Latent World Model + CEM Planning

@PROGRESS.md

This file is the build spec. Read it fully at the start of every session, and re-read the sections a phase points to before building it. `PROGRESS.md` (imported above) says which phase we are in.

## Who you're working with
Aditya is strong in ML/DL but new to robotics, world models and Claude Code. He uses the Code tab of the Claude desktop app and prefers **not** to use the terminal: run every command yourself, then tell him in one or two plain sentences what you did and what's next. Only ask him for things you cannot do: pushing with GitHub Desktop, running Kaggle/Colab notebooks, pasting their output back, creating accounts, entering API keys. While building, don't quiz him or add long explanations; he'll ask when he wants them.

## Goal
Train an action-conditioned latent world model (4-layer CNN encoder + GRU dynamics) on PushT from pixels, then reach goal images with CEM model-predictive control (MPC). Compare against:
1. a pixel-space frame predictor, on multi-step prediction error;
2. goal-conditioned behaviour cloning (BC), on planning success.

All models train on the same 20,000 transitions.

## Hard rules
1. **Never invent, hard-code or "estimate" results.** Every number in the README or any table comes from `results/*.json` written by an evaluation script. If a run hasn't happened, the table says `pending`.
2. **The local machine is an M1 Mac. CPU only. Never use MPS.** Device = `cuda` if available, else `cpu`. Locally, every job runs in smoke mode only; `run.py --mode full` must refuse to run on the local platform (except `make_report`).
3. **Real runs happen on Kaggle (fallback: Colab), T4 GPU,** through the single runner notebook and `run.py` (see §10). The notebook's code never changes between phases; only `JOB`, `MODE` and `EXTRA` in cell 2 change.
4. **Never push.** Commit after each phase (and each fix) with a clear message; Aditya pushes with GitHub Desktop.
5. **Reproducibility:** seed python/numpy/torch; save `config.yaml`, git short hash (+ dirty flag), parameter counts and wall-clock time with every checkpoint and every results JSON.
6. **Plain PyTorch.** No Lightning, no Hydra. Type hints, short docstrings, small files. Clarity over cleverness; this code will be explained in an interview.
7. **Tests:** every phase adds CPU tests under `tests/` (pytest). Run them before every commit.
8. **Headless rendering:** `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` before anything imports pygame.
9. **End of every phase:** update `PROGRESS.md` (status table, decisions, open issues, last REPORT), then commit.

## Python environment
- **Mac:** if `python3` is older than 3.10, install `uv` and create `.venv` with Python 3.11 (`uv venv --python 3.11`). Install with `uv pip install -e ".[local,dev]"`. Use `.venv/bin/python` for every command.
- **Kaggle/Colab:** `pip install -e .` only. **Do not list torch/torchvision as core dependencies**: Kaggle and Colab ship CUDA builds and reinstalling them wastes time or breaks CUDA.
- `pyproject.toml` dependencies: core = `numpy`, `gymnasium`, `gym-pusht`, `opencv-python-headless`, `imageio`, `matplotlib`, `pyyaml`, `scipy`, `wandb`; extra `local` = `torch`; extra `dev` = `pytest`.

## Repo layout
```
pusht-latent-world-model/
├── CLAUDE.md
├── PROGRESS.md
├── README.md                    # generated in Phase 8 from results/*.json
├── pyproject.toml
├── run.py                       # the only entry point: python run.py --job <name> --mode smoke|full [--force]
├── jobs.yaml                    # job registry (§10)
├── configs/
│   ├── data.yaml  wm.yaml  wm_1step.yaml  pixel.yaml  bc.yaml  probe.yaml  plan.yaml
├── wm/
│   ├── platform.py              # detect local / kaggle / colab; output dirs
│   ├── store.py                 # artifact store: local folder on Mac, W&B Artifacts on Kaggle/Colab
│   ├── report.py                # REPORT block + CHECKS
│   ├── envs/pusht.py            # make_env, render64, get_state, reset_to_state, pose_error, is_success
│   ├── data/expert.py           # ScriptedPusher
│   ├── data/collect.py          # episodes -> npz, health report, GIFs, filmstrips
│   ├── data/windows.py          # GPU-resident window sampler (WM, pixel) + goal sampler (BC)
│   ├── models/encoder.py  decoder.py  dynamics.py  world_model.py
│   ├── models/pixel_unet.py
│   ├── models/bc.py
│   ├── models/probe.py
│   ├── planning/cem.py          # batched CEM planner
│   ├── planning/mpc.py          # run_episode(...) closed-loop runner + policies
│   ├── train/common.py          # optimiser, schedule, AMP, checkpoint/resume, logging
│   ├── train/train_wm.py  train_pixel.py  train_bc.py  train_probe.py
│   ├── eval/make_pairs.py       # eval_pairs_{val,test}.json
│   ├── eval/rollouts.py         # NMSE / probe / pixel-MSE curves
│   ├── eval/planning.py         # success, Wilson CI, McNemar, failure tags
│   ├── jobs/                    # one module per job in jobs.yaml
│   └── viz/                     # GIFs, filmstrips, plots
├── scripts/make_readme_tables.py
├── notebooks/runner.ipynb       # the three cells of §10, nothing else
└── tests/
```
`.gitignore`: `.venv/`, `outputs/`, `store/`, `wandb/`, `results_bundle.zip`, `*.npz`, `*.pt`, `__pycache__/`.

## 1. Environment (`wm/envs/pusht.py`)
- `gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")`.
- Observation: `obs["pixels"]` is 96×96×3 uint8; `obs["agent_pos"]` is (x, y). Resize pixels to **64×64** with `cv2.resize(..., interpolation=cv2.INTER_AREA)`.
- Action: absolute target (x, y) in [0, 512]². Control rate: package default (10 Hz).
- `get_state(env) -> np.float32[5]`: pusher x, y, block x, y, block angle, read from `env.unwrapped`. **Verify the attribute names in the installed version** and document them in a comment.
- `reset_to_state(env, state)`: try `env.reset(options={"reset_to_state": state})`; if unsupported, set it through the unwrapped env (e.g. a `_set_state` method). Test: reset → get_state round-trip error < 1 px and < 0.5°.
- `pose_error(state, goal_state) -> (pos_px, ang_deg)`: distance between the block's **centres of gravity**; angle difference wrapped to [−180°, 180°].
- **Decision (Phase 2, pre-registered before any results):** the block position used by `pose_error`, `is_success`, the expert and the data statistics is the T's **centre of gravity**, `block_cog(state) = (block_x, block_y) + R(angle)·(0, 45)`. The stored state's (x, y) is gym-pusht's body origin (the middle of the top edge of the bar, ~45 px from the centre), and it stays exactly as defined in Phase 1. With the CoG, rotating the T in place gives ~0 px position error.
- `is_success(state, goal_state)`: `pos_px < 20 and abs(ang_deg) < 20`. Fixed; never tune these thresholds.
- Normalisation: image `x/255 − 0.5` (CHW float); action/position `x/256 − 1`; angle → (sin, cos).

## 2. Scripted expert (`wm/data/expert.py`)
Per episode: random sub-goal point for the block (uniform in [100, 412]², redrawn every 25 steps); noise σ ∈ {5, 15, 30} px and ε ∈ {0.05, 0.15}, sampled per episode; lateral offset u ∈ [−25, 25] px, sampled per segment.

Each step, with block position b (the T's centre of gravity, §1), pusher position p:
1. d = unit(subgoal − b); n = d rotated 90°.
2. Contact point c = b − 55·d + u·n.
3. If p is on the far side of the block (dot(p − b, d) > 0), first target a detour point b + 90·(±n), choosing the side nearer p; once reached, continue to c.
4. If |p − c| > 15 px: target = p + clip_norm(c − p, 25). Else push: target = p + 20·d.
5. Noise: target += N(0, σ²); with probability ε, target = p + U(−60, 60)².
6. Clip to [0, 512]. Return (noisy_action, clean_action).

Tuning is time-boxed to 90 minutes (Phase 2). Only adjust the numeric parameters above unless Aditya says otherwise.

## 3. Data collection (`wm/data/collect.py`, job `collect_data`)
- Full: train file `pusht_train.npz`, 200 episodes × 100 steps (seeds 0–199) = **20,000 transitions**; episodes with seeds 190–199 are validation. Test file `pusht_test.npz`, 30 episodes × 100 steps (seeds 1000–1029) = 3,000 transitions.
- **Decision (Phase 1):** each episode stores **101 frames** (100 actions + the final frame), so train = exactly 20,000 transitions (20,200 frames) and test = exactly 3,000 transitions (3,030 frames).
- Smoke: 2 train episodes + 1 test episode.
- **Decision (Phase 1): never end an episode early.** Keep stepping for all 100 actions even if the env reports `terminated`. If gym-pusht refuses to step after termination, replace that episode with an extra train seed from 2000 upwards. Validation is always seeds 190–199; test is always seeds 1000–1029.
- Arrays: `frames` uint8 [F,64,64,3]; `states` float32 [F,5]; `actions` float32 [F,2] (action applied at that frame; zeros on each episode's last frame); `clean_actions` float32 [F,2]; `episode_starts`, `episode_lengths`, `episode_seeds` int64 [E] (**decision (Phase 2):** `episode_seeds` added so val episodes are found by seed). `np.savez_compressed`.
- Health outputs: `data_health.json` with contact ratio (share of steps where the block moves > 0.5 px or turns > 0.5°), coverage (share of an 8×8 grid over the table the block visited; angle std), action stats; histograms PNG; 3 GIFs and 3 filmstrip PNGs (8 evenly spaced frames side by side) of random episodes.
- REPORT CHECKS: exact transition counts; contact ratio ≥ 35% PASS, 25–35% WARN, < 25% FAIL.

## 4. World model (`wm/models/*`, jobs `train_wm`, `train_wm_1step`)
| Part | Spec |
| --- | --- |
| Encoder | 4 × [Conv2d(k=4, s=2, p=1) + SiLU], channels 3→32→64→128→256 (64×64 → 4×4); flatten 4096; Linear 4096→256; `LayerNorm(256, elementwise_affine=False)` |
| Dynamics | φ = Linear(256+2 → 256) + SiLU; `GRUCell(256, 320)`; out = Linear(320 → 256) giving Δ; ẑ_next = LN(ẑ + Δ) (same non-affine LayerNorm) |
| Decoder | Linear 256→4096, reshape 256×4×4, ConvTranspose2d(k=4, s=2, p=1) 256→128→64→32→3 with SiLU between (none at the end) |

Expected parameter counts (assert in a test; print at startup) for the default recipe (`wm.yaml` = A+B, see the Phase 3b decision below): encoder 1,738,976; decoder 1,742,563; dynamics 704,514; **total 4,186,053**; planner path (encoder + dynamics) 2,443,490. The v1 model as specified in the table above (`wm_v1.yaml`, no proprio) has dynamics 703,360, total 4,184,899, planner 2,442,336.

Window = 13 frames o[t−2..t+10] and 12 actions a[t−2..t+9]; context C = 3, horizon H = 10.
- z_j = E(o_j) for all 13 frames.
- Warm-up: h = 0; for j in (t−2, t−1): h = GRU(φ([z_j, a_j]), h).
- Rollout: ẑ = z_t; for k = 0..9: h = GRU(φ([ẑ, a_{t+k}]), h); ẑ = LN(ẑ + out(h)); store ẑ_{t+k+1}.
- `wm_1step.yaml` sets `teacher_forcing: true`: in the rollout feed the real z_{t+k} instead of ẑ.

Loss (element means): `λ_rec·MSE(D(z_j), o_j)` over all 13 frames + `λ_lat·mean_k MSE(ẑ_{t+k}, z_{t+k}.detach())` + `λ_img·mean_k MSE(D(ẑ_{t+k}), o_{t+k})`; λ_rec = 1.0, λ_lat = 0.1, λ_img = 1.0 (config).

**Decision (Phase 3, 2026-11-26):** add a VICReg-style variance hinge `λ_var·mean_d relu(1 − std_batch(z_d))` (std over all 13·B latents of the batch, +1e-4 inside the sqrt), λ_var = 1.0. Without it the spec'd model collapses (z_std ≈ 0.003: every frame encodes to nearly the same latent). Chosen from a 5-variant local comparison (`results/ablation_collapse_local.json`, `scripts/compare_collapse_fixes.py`). Architecture and parameter counts are unchanged.

**Phase 3b options (2026-11-26; config flags, all off in code by default and in `wm_v1.yaml`):** (A) `fg_weight`: rec/img loss weight w = 1 + 20·mask, mask = |frame − train mean frame| > 25/255 in any channel; (B) `proprio`: pusher position p is model state, dynamics input [z, p, a], extra head Linear(320 → 2) (zero-init) predicts p' = p + Δp with MSE weight 1.0, rollouts feed back p̂, `rollout(..., p0=p) -> (z_hat, p_hat)`; total params 4,186,053 (planner 2,443,490); (C) `keep_best`: keep the checkpoint with the best val NMSE@10 and record `best_step`. Job `wm_ablate` compares them (PROGRESS.md has the pre-registered selection rule).

**Decision (Phase 3b, 2026-11-27): adopt A+B; not C.** Kaggle `wm_ablate` full (commit ffd775a, source `results/ablation_wm_fixes.json`; table in PROGRESS.md) picked **A+B** by the pre-registered rule: it beats copy-last on imagined block px at k = 10 on val and its imagined angle error is no worse than copy-last's. C (`keep_best`, selected by val NMSE@10) was not adopted: the NMSE-selected weights were physically worse than the last ones, so NMSE@10 is not a reliable checkpoint-selection metric here. `wm.yaml` and `wm_1step.yaml` set `fg_weight: true`, `proprio: true`, `keep_best: false` (they still differ only in `teacher_forcing`); the Phase 3 recipe is kept as `wm_v1.yaml`, and Kaggle `ckpt-wm:v5` is that v1 model. Consequences: the planner path takes the pusher position (`warmup(z, a, p_ctx)`, `rollout(z0, h, a, p0=p) -> (z_hat, p_hat)`); the planning cost still uses only the image latent z_g.

Training: batch 64 windows; AdamW lr 3e-4, wd 1e-4; 500-step linear warm-up then cosine; 30,000 steps (full); fp16 autocast + GradScaler; grad-clip 1.0; validation every 1,000 steps on val episodes; checkpoint every 5,000 steps (pushed to the store, so a killed run resumes) + final. Log each loss term and the **collapse alarm** `z_std` = mean over dims of the batch std of z (CHECK: > 0.1).

Smoke: 50 steps, batch 4 locally; on Kaggle 300 steps at full batch on the real data. Smoke REPORT includes it/s after warm-up, projected time for 30,000 steps, peak GPU memory, first vs last loss.

Full REPORT includes final losses, val NMSE at k = 1, 5, 10 for the model and for copy-last (CHECK: model < copy-last at k = 10), wall-clock, z_std, and a filmstrip of one decoded 10-step rollout beside the real frames.

Data loading: load the whole npz onto the GPU as uint8 tensors once; sample (episode, start) indices and gather windows on-GPU. No DataLoader.

Planning API: `wm.encode(frames)`, `wm.warmup(z_ctx, a_ctx, p_ctx) -> h`, `wm.rollout(z0, h, actions[B,H,2], p0=p) -> (z_hat[B,H,256], p_hat[B,H,2])` (batched; a v1 model without proprio takes no p and returns z_hat only), `wm.decode(z)`.

## 5. Pixel baseline (`wm/models/pixel_unet.py`, job `train_pixel`)
- Input: 3 context frames stacked (9 ch) + action tiled as 2 constant channels = 11 ch, 64×64.
- U-Net: down convs (k=4, s=2, p=1) 11→48→96→192→384; bottleneck Conv 3×3 on [384 + 2 tiled action channels] → 384; up ConvTranspose (k=4, s=2, p=1) with skip concatenation 384→192, (192+192)→96, (96+96)→48, (48+48)→3. Output Δ image; prediction = last frame + Δ. ≈4.8M params (CHECK: within ±20% of 4,184,899; **decision (Phase 1):** widened from ±15% because the spec as written gives 4,813,731, just above the +15% bound of 4,812,634).
- Train one-step, teacher-forced, pixel MSE; batch 64; same optimiser/schedule; 30,000 steps.
- Rollout: autoregressive, sliding the 3-frame window with predicted frames.

## 6. Goal-conditioned BC (`wm/models/bc.py`, job `train_bc`)
- Inputs: 3 context frames + goal frame, each through one shared encoder (same design as §4, fresh weights) → 4×256; plus pusher (x, y) normalised → 1,026 → MLP 512 → 512 → 2 (SiLU, dropout 0.1). ≈2.53M params.
- Output: (dx, dy)/50 px relative to the current pusher position; executed target = p + 50·output, clipped to [0, 512].
- Goal relabelling: goal = real frame at t + k, k ~ U{5..30}, clipped to the episode end.
- Train: MSE on the logged (noisy) action in the same delta form; batch 256; AdamW 3e-4; 20,000 steps; random shift ±4 px (pad 4, random crop) applied identically to all 4 images of a sample.
- Config option `use_clean_actions: true` (stretch ablation).

## 7. State probe (`wm/models/probe.py`, job `train_probe`)
Same encoder design → Linear 256→6: pusher (x, y), block (x, y) normalised, sin θ, cos θ. Train on real train frames, 10,000 steps, MSE. REPORT its error on real test frames (block px, block °, pusher px): this is the judge's noise floor.

Job group `train_baselines` = `train_wm_1step`, `train_pixel`, `train_bc`, `train_probe`, in that order, each skipped if already complete.

**Checkpoint round trip (every training job):** the reloaded `model.pt` weights must be bit-identical, and its outputs must match the trained model's within `torch.allclose(rtol=1e-4, atol=1e-5)`, both models in eval mode. The REPORT line shows the max absolute and relative difference and the tolerance. (Exact output equality failed on CUDA for the pixel U-Net, most likely because `ConvTranspose2d` can be nondeterministic there.)

## 8. CEM planner and MPC (`wm/planning/`)
| Setting | Value |
| --- | --- |
| candidates N | 128 (one candidate is always the current mean) |
| horizon H | 10 |
| rounds | 4 |
| elites K | 12 |
| decision variables | δ ∈ R^{H×2} in px; clip each to ±50 |
| waypoints | w_k = clip(p + Σ_{j≤k} δ_j, 0, 512); model action = w_k/256 − 1 |
| init | μ = warm start (previous best, shifted by one, last = 0) or zeros; σ = 30 px; σ floor 2 px |
| cost | ‖ẑ_H − z_g‖² + β Σ_{k=1}^{H−1} ‖ẑ_k − z_g‖², β = 0.1 |
| update | μ, σ = mean, std of elites |

- Context each step: last 3 real frames and the 2 actions between them; at episode start, repeat the first frame 3× and use the pusher's current position as the padding actions.
- `run_episode(env, start_state, goal_frame, goal_state, policy, budget=40)`: reset_to_state; loop: act, step, check `is_success` every step, stop on success. Return success, steps, final errors, per-step best imagined cost, real frames, and (CEM) imagined plan latents for visualisation.
- Policies with one interface: `CEMPolicy(wm)`, `BCPolicy`, `RandomPolicy` (δ ~ N(0, 30²) through the same waypoint transform), `NoopPolicy` (target = current pusher position).
- `configs/plan.yaml` holds these values; Phase 6 freezes them after validation tuning.

## 9. Evaluation (`wm/eval/`)
**Pairs (`make_pairs.py`):** per episode of a split, candidate starts t ∈ {0, 5, …, 75}, goal = t + 20; keep pairs where the block moves ≥ 30 px or turns ≥ 20° between start and goal. Seed 0: 200 test pairs → `eval_pairs_test.json`; 50 pairs from validation episodes → `eval_pairs_val.json`. Store start state, goal state, goal frame index. Goal-distance sweep regenerates pairs with offsets 10 and 30.

**Rollouts (`rollouts.py`, job `eval_rollouts`):** 500 test windows (seed 0; smoke 20). Methods `wm`, `wm_1step`, `pixel`, `copy_last`. Per k = 1..10:
- NMSE_k = mean ‖ẑ_{t+k} − z_{t+k}‖² / mean ‖z − z̄‖² (z̄ = mean latent over test frames; all latents from the `wm` encoder; pixel-model frames are encoded with it).
- Probe errors (block px, block °) on predicted frames (`wm` via decoder; `pixel` directly; `copy_last` = last real frame).
- Pixel MSE.
- 95% bootstrap CIs (1,000 resamples over windows).
Write `rollouts.json`, plots of each metric vs k, 6 filmstrips (real | wm decoded | pixel). **Phase 5 (pre-registered in PROGRESS.md):** primary metrics are the probe's block CoG px / angle ° / pusher px (+ p̂ pusher px); headline NMSE is `wm_via_frames` = E_wm(D_wm(ẑ)); predicted frames are clipped to [−0.5, 0.5]; filmstrips are real | wm | wm_1step | pixel; settings in `configs/rollouts.yaml`. REPORT: table at k = 1, 5, 10 for every method + relative reduction (1 − wm ÷ pixel) at k = 5, 10. CHECK: wm NMSE < copy_last NMSE at every k.

**Planning (`planning.py`):** methods `cem_wm`, `cem_wm_1step`, `bc`, `random`, `noop`. Per method: success rate, Wilson 95% CI, median final pos/angle error, mean steps-to-success, seconds per episode.
- Job `plan_val`: all methods, 50 val pairs, 1 seed (smoke: 3 pairs, budget 10).
- Job `plan_sweep`: val pairs only; from the default, vary one at a time: N ∈ {32, 64, 128, 256}, rounds ∈ {1, 2, 4}, H ∈ {5, 10, 15}, β ∈ {0, 0.1}; `cem_wm` only. REPORT: table of success vs setting.
- Job `final_eval` (Phase 7): uses the frozen config; `plan_test` = all methods × 200 test pairs × 3 planner seeds, each pair used exactly once per method and seed; McNemar exact test `cem_wm` vs `bc` (`scipy.stats.binomtest` on discordant pairs), per seed and pooled; goal-distance sweep (offsets 10/20/30, `cem_wm` and `bc`, 100 pairs each, 1 seed); failure tags for each failed `cem_wm` episode — `model_error` (best imagined cost fell below the val-calibrated "imagined success" threshold, real outcome failed), `search_error` (never below it), `contact_switch` (the needed push direction changes by > 90° during the episode), `rotation_heavy` (|Δθ| ≥ 30° and Δpos < 30 px); a gallery of 6 failed episodes (real frames beside the decoded imagined plan); finally `results_bundle.zip` with every results JSON, plot and gallery image, saved to the output dir and the store.
- **Tuning discipline:** only `plan_val`/`plan_sweep` touch val pairs; only `final_eval` touches test pairs; nothing is tuned after `final_eval`.

**Stretch (only if asked):** context C = 1 ablation, pixel baseline with 10-step rollout loss, BC on clean actions, 2 extra training seeds for `wm` and `bc`.

## 10. Runner: platform, store, jobs, REPORT
**Platform (`wm/platform.py`):** `kaggle` if env var `KAGGLE_KERNEL_RUN_TYPE` exists; `colab` if `google.colab` is importable; else `local`. Output dir: `/kaggle/working/outputs`, `/content/outputs`, `./outputs`.

**Store (`wm/store.py`):** `push(name, path, metadata)`, `pull(name) -> path | None`, `exists(name)`.
- Local: folder `./store/<name>/`.
- Kaggle/Colab: W&B Artifacts in project `pusht-world-model` (needs `WANDB_API_KEY`; fail with a clear message if missing).
- Smoke mode writes `<name>-smoke`. Smoke mode reads `<name>` if it exists, else `<name>-smoke`. Full mode reads and writes `<name>` only. Locally everything is smoke.
- Metadata always includes `complete`, `step` (if training), `config_hash`, `git`.

**Resume/skip:** a job whose output already exists with the same `config_hash` and `complete: true` prints "already finished, skipping" (unless `--force`). An eval job (one whose `needs` include `ckpt-*`) also records `inputs` (each needed checkpoint's producer config_hash) and only skips if they are still current. A training job whose latest checkpoint is incomplete resumes from it.

**Groups:** a group (`jobs.yaml` `groups`; a member may itself be a group) runs its jobs in order. A job that FAILs is recorded and the remaining jobs still run; at the end one `GROUP <name>: OK|FAILED (...) | PASS ... | SKIP ... | FAIL ...` line is printed, and the exit code is non-zero if any job failed.

**Loading trained checkpoints (rule for every eval job: `diagnose_wm`, `eval_rollouts`, `plan_val`, `plan_sweep`, `final_eval`):** load `ckpt-wm`, `ckpt-wm-1step` (and any other `ckpt-*`) only through `wm/jobs/_ckpt.py::pull_matching`, which takes the first read candidate that is `complete: true` and whose `config_hash` equals what the producing job's *current* config gives; if none matches, FAIL with a message naming what was found. The REPORT prints the checkpoint's config_hash (and, for world models, the `proprio` flag). Without this, smoke mode's "read `<name>` first" rule would silently pick an older recipe.

**Jobs (`jobs.yaml` → `wm/jobs/*`):**
| Job | Phase | Needs | Makes |
| --- | --- | --- | --- |
| `env_check` | 1 | – | `env-check` (tiny round-trip file) |
| `collect_data` | 2 | – | `pusht-data` |
| `train_wm` | 3 | `pusht-data` | `ckpt-wm` |
| `wm_ablate` | 3b | `pusht-data` | `results-wm-ablate` (+ `results/ablation_wm_fixes.json`) |
| `train_baselines` | 4 | `pusht-data` | `ckpt-wm-1step`, `ckpt-pixel`, `ckpt-bc`, `ckpt-probe` |
| `overnight` (group) | 4 | `pusht-data` | `train_wm` → `diagnose_wm` → `train_baselines`, each skipped if complete |
| `eval_rollouts` | 5 | data + `ckpt-wm`, `ckpt-wm-1step`, `ckpt-pixel`, `ckpt-probe` | `results-rollouts` |
| `plan_val` | 6 | data + `ckpt-wm`, `ckpt-wm-1step`, `ckpt-bc` (+ makes `eval-pairs` if missing) | `results-plan-val` |
| `plan_sweep` | 6 | as `plan_val` | `results-plan-sweep` |
| `final_eval` | 7 | everything above | `results-final`, `results-bundle` |
| `make_report` | 8 | local `results/` | README |

`env_check` checks: platform, Python/torch versions, device and GPU name, a 64×64 render saved as `frame.png`, reset_to_state round trip, store push→pull round trip, a 2048×2048 matmul timing.

**REPORT (`wm/report.py`):** every job ends by printing exactly this layout and saving it to `<out_dir>/<job>/REPORT.md`; exit code ≠ 0 if any FAIL:
```
==================== RUN REPORT ====================
job: <job>   mode: <smoke|full>   platform: <local|kaggle|colab>   device: <cpu | GPU name>
git: <short hash>[-dirty]   time: <wall clock>   params: <if relevant>
CHECKS
 [PASS|WARN|FAIL] <check>: <value> (<threshold>)
METRICS
 <name> = <value>
SAVED
 <artifact name:version or local path> ...
W&B: <run URL or "off">
====================================================
```

**Runner notebook (`notebooks/runner.ipynb`):** exactly three code cells:
```python
# CELL 1: setup. Edit GITHUB_USER once, then never touch.
GITHUB_USER = "your-github-username"
REPO = "pusht-latent-world-model"
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
REPO_DIR = f"{BASE}/{REPO}"
if os.path.isdir(REPO_DIR):
    !git -C {REPO_DIR} pull -q
else:
    !git clone -q https://github.com/{GITHUB_USER}/{REPO}.git {REPO_DIR}
os.chdir(REPO_DIR)
!pip install -q -e .
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
except Exception:
    try:
        from google.colab import userdata
        os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
    except Exception:
        print("WARNING: no WANDB_API_KEY found; runs can't save or load artifacts")
```
```python
# CELL 2: the only cell you edit
JOB   = "env_check"
MODE  = "smoke"     # "smoke" first, then "full"
EXTRA = ""          # leave empty unless told otherwise (e.g. "--force")
```
```python
# CELL 3: run. Never edit.
!python run.py --job {JOB} --mode {MODE} {EXTRA}
```

## 11. Phases and acceptance
| Phase | Builds | Local smoke command | Accept when |
| --- | --- | --- | --- |
| 1 Foundation | env, platform, store, report, run.py, jobs.yaml, runner notebook, PROGRESS.md | `run.py --job env_check --mode smoke` | all CHECKS pass; `frame.png` shows the T and pusher |
| 2 Expert + data | §2, §3 | `run.py --job collect_data --mode smoke` | Aditya confirms the pusher pushes the T in ≥ 2 of 3 filmstrips; counts exact |
| 3 World model | §4 | `run.py --job train_wm --mode smoke` | param test passes; loss falls; resume works |
| 4 Baselines | §4 (1-step), §5, §6, §7 | `run.py --job train_baselines --mode smoke` | smoke tests pass; pixel params within ±20% |
| 5 Rollout eval | §9 Rollouts | `run.py --job eval_rollouts --mode smoke` | JSON, plots, filmstrips produced |
| 6 Planner | §8, §9 Pairs/Planning | `run.py --job plan_val --mode smoke` | CEM toy-quadratic test passes; one CPU episode runs end to end |
| 7 Final eval | §9 `final_eval` | `run.py --job final_eval --mode smoke` | smoke bundle contains all tables with CIs and McNemar fields |
| 8 Report | `make_report`, README | `run.py --job make_report` | every README number traces to a JSON file |

After each phase: tests pass, PROGRESS.md updated, commit, then a two-line status for Aditya (what works, what he does next).

# Latent world model + CEM planning on PushT, from pixels

An action-conditioned latent world model (CNN encoder, GRU dynamics, decoder; {{f:params_m}} parameters) trained on a {{f:transitions}}-transition PushT dataset, used for goal-image reaching with CEM model-predictive control, and compared with a pixel-space predictor and goal-conditioned behaviour cloning. All test-set comparisons were pre-registered before they were run.

**TL;DR**
- **Multi-step training works:** training the dynamics on {{f:horizon}}-step open-loop rollouts instead of teacher forcing gives {{f:nmse_red_1step_mean}}% lower {{f:horizon}}-step latent prediction error (95% CI {{f:nmse_red_1step_lo}} to {{f:nmse_red_1step_hi}}%) and {{f:pt_cem_wm_1step_succ_diff}} pp planning success ({{f:pt_cem_wm_1step_pos_diff}} pp position-only) over {{f:n_test_pairs}} held-out goals × {{f:n_seeds}} seeds, both significant after Holm correction (exact McNemar, Holm p = {{f:pt_cem_wm_1step_succ_holm}} and {{f:pt_cem_wm_1step_pos_holm}}).
- **Latent planning is fast but weaker than pixel planning:** latent CEM takes {{f:cem_wm_call}} s per planning call vs {{f:cem_pixel_call}} s for pixel-space CEM (about {{f:speedup}}× faster), at {{f:pt_cem_pixel_succ_diff}} pp success.
- **Behaviour cloning beats planning:** goal-conditioned BC reaches {{f:bc_clean_success_mean}}% success vs {{f:cem_wm_success_mean}}% for the latent planner.
- **Diagnosis:** imagined rollouts lose the block's orientation (pre-registered verdict: {{f:ang_verdict}}; {{f:ang_wm_k10}}° error at k = 10 vs {{f:ang_wm_recon_k10}}° for plain reconstruction), and {{f:cem_wm_model_error_pct}}% of the planner's failures ({{f:cem_wm_believed_failed}} of {{f:cem_wm_failed}}) are model errors: the planner believed it had reached the goal.

![Test success per method](results/final/plots/success.png)
*Test success (dark) and position-only success (light) per method, averaged over planner seeds, with 95% pair-clustered bootstrap CIs.*

![Imagined angle error vs prediction step](results/final/plots/imagined_angle_vs_k.png)
*Block-orientation error read by a non-learned template oracle on imagined frames: the world model's rollout (blue) loses the angle step by step, while decoding a real latent (grey, dotted) does not.*

![cem_wm episodes that reached the position but not the angle](results/final/gallery/cem_wm_position_only.png)
*Latent-planner test episodes tagged "position only" (columns: start, goal, final frame, decoded imagined terminal frame of the last plan): the block ends at the right place with the wrong orientation, and the imagined frame is a blur; the green T is gym-pusht's fixed marker, not the evaluation goal.*

## 1. Task and data
**Environment.** `gym_pusht/PushT-v0` (gym-pusht): a round pusher pushes a T-shaped block on a table. Observations are 96×96 RGB renders, resized to 64×64; an action is an absolute target position for the pusher. The model never sees the true state; states are used only for data statistics, success checks and evaluation.

**Data.** A noisy scripted expert pushes the block towards a random sub-goal, redrawn every {{f:subgoal_every}} steps. Each episode samples Gaussian action noise σ ∈ {{{f:noise_sigmas}}} px and a random-jump probability ε ∈ {{{f:random_eps}}}, so the data mixes clean pushes with clumsy ones. Every episode runs exactly {{f:steps_per_ep}} actions (never ended early) and stores {{f:frames_per_ep}} frames. Splits are by episode seed:

| split | episode seeds | episodes | transitions (actions) |
| --- | --- | --- | --- |
| train | {{f:train_seeds}} | {{f:train_eps}} | {{f:train_transitions}} |
| validation | {{f:val_seeds}} | {{f:val_eps}} | {{f:val_transitions}} |
| **train + val** | | **{{f:episodes}}** | **{{f:transitions}}** |
| test (separate file) | {{f:test_seeds}} | {{f:test_eps}} | {{f:test_transitions}} |

The counts are checked against the training checkpoints: {{f:train_windows}} train and {{f:val_windows}} validation windows of thirteen frames, and {{f:test_frames}} test frames. The world model, the pixel model and BC train on the train split only and select nothing on test. Two state probes used only as measuring instruments (section 3) also saw extra frames rendered from random states.

## 2. Method
### World model
| part | design |
| --- | --- |
| encoder E | four stride-two conv layers with SiLU, 64×64 → 4×4, then Linear → z ∈ R^{{f:latent_dim}} and a non-affine LayerNorm |
| dynamics | φ = Linear([z, p, a]) + SiLU → GRUCell (hidden {{f:hidden_dim}}) → residual step ẑ' = LN(ẑ + Δ(h)); a head predicts the pusher position p̂' = p̂ + Δp(h) |
| decoder D | Linear → 4×4 feature map → four transposed convs → 64×64 image |

Parameters: encoder {{f:params_encoder}}, dynamics {{f:params_dynamics}}, decoder {{f:params_decoder}}, total **{{f:params_total}}**; the planner uses encoder + dynamics ({{f:params_planner}}).

**Training** uses windows of {{f:context}} context frames and a {{f:horizon}}-step future. The GRU warms up on the context, then rolls out **open loop**: each predicted ẑ is fed back as the next input. The loss is a weighted sum of
- frame reconstruction D(E(o)) for every frame (weight {{f:lambda_rec}});
- latent prediction ‖ẑ − sg(E(o))‖² (weight {{f:lambda_lat}});
- decoded-prediction error D(ẑ) vs the real future frames (weight {{f:lambda_img}});
- a **VICReg-style variance hinge** on the batch std of every latent dimension (weight {{f:lambda_var}}), which stops representation collapse (section 5);
- pusher-position prediction (weight {{f:lambda_proprio}}).

Reconstruction and image terms are **foreground-weighted** (pixels that differ from the mean training frame count {{f:fg_alpha}}× more), because the block and pusher cover only a few per cent of the image. Training: {{f:train_steps}} steps, batch {{f:batch}}, AdamW with warm-up + cosine, fp16, about {{f:train_wm_min}} minutes on a Kaggle T4.

**Baselines.**
- `wm_1step`: the same model trained with teacher forcing (the real E(o) is fed at every step); it differs from `wm` in nothing else.
- `pixel`: a U-Net ({{f:params_pixel}} parameters) predicting the next frame from the last {{f:context}} frames and the action, rolled out autoregressively.
- `bc_clean`: goal-conditioned behaviour cloning (shared CNN encoder on {{f:context}} context frames + the goal frame, MLP head, {{f:params_bc}} parameters) trained on the expert's clean actions. It was picked over BC on the noisy logged actions by validation success ({{f:pv_bc_clean}} vs {{f:pv_bc}} of {{f:n_val_pairs}}), by a pre-registered rule.
- `random` (random waypoints) and `noop` (stay put).

### CEM model-predictive control
At every environment step the planner encodes the last {{f:context}} real frames, warms up the GRU, and searches over {{f:H_cem}}-step plans of pusher waypoints (per-step moves clipped to ±{{f:dclip}} px). The cost uses only the image latent of the goal frame z_g:

cost = ‖ẑ_H − z_g‖² + β · Σ_{k<H} ‖ẑ_k − z_g‖²,  β = {{f:beta}}.

CEM: {{f:N}} candidates, {{f:rounds}} rounds, {{f:K}} elites, initial σ {{f:sigma0}} px, warm start from the previous plan. Only the first waypoint is executed, then it replans. These settings were frozen after a validation sweep (section 5). `cem_wm_1step` uses the same planner with the teacher-forced model. `cem_pixel` uses the same planner and cost with the pixel U-Net as dynamics, its predicted frames encoded by E.

## 3. Evaluation protocol
- **Goal pairs.** A start state and the real frame {{f:goal_offset}} steps later in the same test episode, kept only if the block moves ≥ {{f:min_move}} px or turns ≥ {{f:min_turn}}° between them: {{f:n_test_pairs}} test pairs ({{f:n_turning}} turning, {{f:n_nonturning}} not), {{f:n_val_pairs}} validation pairs.
- **Success:** block centre of gravity within {{f:succ_px}} px **and** angle within {{f:succ_deg}}° of the goal pose, checked after every step, budget {{f:budget}} steps. **Position-only success:** the {{f:succ_px}} px condition alone.
- **Seeds:** every method runs planner seeds {{f:seeds}}; `cem_pixel` runs seed {{f:pixel_seeds}} only (compute; pre-registered).
- **Pre-registration.** Every test-set comparison, threshold, bin and verdict rule was written into `PROGRESS.md` and committed before the run that produced it. Only validation pairs were used for any choice. Nothing was tuned after the test run.
- **Primary tests ({{f:n_tests}}):** `cem_wm` vs `bc_clean`, `cem_wm_1step` and `cem_pixel`, each on success and position-only success. Exact McNemar test on seed {{f:seed0}}, Holm correction over the {{f:n_tests}} p-values, claim only if Holm p < {{f:alpha}}. The effect size is a paired bootstrap CI over pairs ({{f:boot}} resamples) of each method's per-pair success averaged over its seeds. Pooled rates use a pair-clustered bootstrap. Everything else is secondary.
- **Measuring the imagined angle.** Learned state probes failed as judges. The first CNN probe memorised orientation: mean angle error {{f:v1_angle_train}}° on train frames but {{f:v1_angle_val}}° on validation and {{f:v1_angle_test}}° on test (an angle audit found no label bug: re-rendered stored states match the stored frames with a mean absolute difference of {{f:rerender_diff}}). A second probe trained on real, decoded and rendered frames still failed its pre-registered acceptance (real-frame angle {{f:pmix_angle}}° vs a {{f:pmix_gate_angle}}° gate). So orientation is read by a **non-learned template oracle**: render the T at the true position every {{f:ang_step}}° and pick the best match, searching small shifts. On real test frames its mean error is {{f:ang_real_k10}}°. Block position in latents is read by a small MLP on E(o) (real-frame floor {{f:latent_judge_block}} px).

## 4. Results (test set)
### Main table
{{t:main}}

τ is the imagined-success threshold (validation-calibrated, {{f:tau_wm}} for `wm`): a CEM episode "believed" it succeeded if the best plan's terminal cost fell below τ.

### Primary tests (pre-registered)
{{t:primary}}

Five of the six tests support a claim:
- the multi-step model plans better than the teacher-forced one;
- BC and pixel-space CEM both beat latent CEM on success.

The sixth, `cem_wm` vs `bc_clean` on position-only success, gets **no claim**: Holm p = {{f:pt_bc_clean_pos_holm}}, even though its bootstrap CI ({{f:pt_bc_clean_pos_lo}} to {{f:pt_bc_clean_pos_hi}} pp) excludes zero. On position alone the gap to BC shrinks ({{f:pt_bc_clean_pos_diff}} pp vs {{f:pt_bc_clean_succ_diff}} pp on full success), while pixel-space CEM stays clearly ahead ({{f:pt_cem_pixel_pos_diff}} pp). So much of the gap to BC is orientation, not position.

### Multi-step prediction ({{f:rollout_windows}} test windows)
NMSE = mean squared latent error divided by the latent variance; for a like-for-like comparison every method's predicted frames are re-encoded by E (the headline row). Block position is read by the latent judge; the pusher error comes from the model's own p̂.

{{t:rollouts}}

- **vs teacher forcing:** at k = 10 the multi-step model's NMSE is {{f:nmse_red_1step_mean}}% lower than `wm_1step`'s (95% CI {{f:nmse_red_1step_lo}} to {{f:nmse_red_1step_hi}}%). It tracks the pusher to {{f:phat_wm}} px vs {{f:phat_wm_1step}} px (copy-last {{f:phat_copy_last}} px).
- **vs the pixel model (negative):** the pixel U-Net predicts better. The NMSE "reduction" vs pixel is {{f:nmse_red_pixel_mean}}% (CI {{f:nmse_red_pixel_lo}} to {{f:nmse_red_pixel_hi}}%), i.e. `wm`'s error is higher, and the block-error difference `wm` − pixel is {{f:blockd_pixel_mean}} px (CI {{f:blockd_pixel_lo}} to {{f:blockd_pixel_hi}}).

### Imagined orientation (template oracle, search reader, {{f:ang_windows}} test windows)
{{t:angle}}

Paired differences (degrees, `wm` minus the other row; positive = `wm` worse):

{{t:angle_paired}}

**Verdict (pre-registered, k = 10): "imagined rollouts lose the angle" = {{f:ang_verdict}}.** Both conditions hold:
- (A) the rollout adds error beyond reconstruction ({{f:angd_wm_minus_wm_recon_mean}}°, CI {{f:angd_wm_minus_wm_recon_lo}} to {{f:angd_wm_minus_wm_recon_hi}});
- (B) it is worse than simply copying the last frame, both as read by the oracle ({{f:angd_wm_minus_copy_last_mean}}°) and against the true angle change ({{f:angd_wm_minus_copy_last_exact_mean}}°).

Multi-step training helps a little (`wm` − `wm_1step` = {{f:angd_wm_minus_wm_1step_mean}}°, CI {{f:angd_wm_minus_wm_1step_lo}} to {{f:angd_wm_minus_wm_1step_hi}}). The pixel model keeps the angle far better ({{f:ang_pixel_k10}}°).

### Speed
{{t:speed}}

### Failure tags (from the final state; all seeds, seed 0 in brackets)
{{t:tags}}

`cem_wm` ends "position only" more often than any other method ({{f:cem_wm_tag_position_only}} episodes vs {{f:bc_clean_tag_position_only}} for `bc_clean`). Of its {{f:cem_wm_failed}} failures, {{f:cem_wm_believed_failed}} were believed successes: the bottleneck is mostly the model, not the search.

### Exploratory: goal difficulty and goal distance (no multiple-comparison correction; not claims)
Success / position-only per bin of block movement (px) and absolute turn (°), pooled over seeds:

{{t:bins_move}}

{{t:bins_turn}}

Goal-distance sweep (seed {{f:seed0}}; goal {{f:sw10_offset}} or {{f:sw30_offset}} steps ahead instead of {{f:goal_offset}}):

{{t:sweep}}

The pattern: latent CEM is competitive on short moves and degrades with distance, while BC stays roughly flat.
- Moves in {{f:bin_short_label}} px: `cem_wm` {{f:bin_cem_wm_short}}% vs `bc_clean` {{f:bin_bc_clean_short}}%.
- Moves {{f:bin_long_label}} px: `cem_wm` {{f:bin_cem_wm_long}}% vs `bc_clean` {{f:bin_bc_clean_long}}%.
- Goal-distance sweep, `cem_wm` vs `bc_clean` successes (of {{f:sw10_n}} pairs each): {{f:sw10_cem_wm}} vs {{f:sw10_bc_clean}} at {{f:sw10_offset}} steps ahead; {{f:sw30_cem_wm}} vs {{f:sw30_bc_clean}} at {{f:sw30_offset}} steps.

## 5. What went wrong, and what I learned
- **Representation collapse.** The model as first specified collapsed: every frame encoded to nearly the same latent (z std {{f:col_base_zstd}}). The prediction losses were then trivially small. A local comparison of five fixes ({{f:col_steps}} CPU steps each, `results/ablation_collapse_local.json`) picked a VICReg-style variance hinge (z std {{f:col_b_zstd}}). Lesson: a latent-prediction loss alone has a collapsed optimum; watch z std from step one.
- **The first latent did not contain the pusher.** A ridge probe on the first model's latents missed the pusher by {{f:abl_v1_pusher}} px. The pusher is a small blob, and a plain pixel loss barely notices it. A pre-registered ablation ({{f:abl_steps}} steps per variant) adopted foreground weighting + an explicit pusher state, giving p̂ errors of {{f:abl_ab_phat}} px at k = 10 vs {{f:abl_ab_phat_copy}} px for copy-last. A third option, keeping the checkpoint with the best validation NMSE, was rejected: those weights were physically worse.
- **The judge memorised.** The first evaluation probe looked fine on training frames and was confidently wrong on held-out ones (section 3). The audit ruled out a label bug. The fix was to stop trusting learned judges for orientation and use a non-learned template oracle. Lesson: validate the instrument on held-out and out-of-distribution frames before reading any model through it.
- **A sharper planning cost did not help.** On validation the planner reached the position but not the angle (`cem_wm` {{f:pv_cem_wm}} successes but {{f:pv_cem_wm_pos}} position-only of {{f:n_val_pairs}}, and {{f:pv_imag_fail}} of {{f:pv_imag}} believed successes failed for real). A pre-registered diagnostic replaced the latent cost with a foreground-weighted pixel cost on decoded frames. Result: {{f:pca_pixel}} vs {{f:pca_latent}} successes, a null (it needed {{f:sweep_margin}} more to join the final test), at {{f:pca_spe}} s per episode. The CEM sweep also changed nothing: best variant {{f:ps_best}} vs default {{f:ps_default}} successes.
- **The angle is lost in the rollout.** The null above pointed away from the cost and toward the prediction. The test-set oracle confirmed it:
  - orientation survives encoding and decoding ({{f:ang_wm_recon_k10}}°) but not imagination ({{f:ang_wm_k1}}° → {{f:ang_wm_k5}}° → {{f:ang_wm_k10}}° at k = 1, k = 5, k = 10);
  - a latent judge also rates orientation in real latents as only "{{f:latent_judge_verdict}}" ({{f:latent_judge_angle}}° error).
  - Lesson: a planner can only optimise what its model imagines; check what the rollout keeps before tuning the search.

## 6. Deviations from the pre-registration (disclosed)
From `PROGRESS.md`, one line each:
- **Judge freeze after the rollout evaluation:** the pre-registered rule allowed a block-position headline only with an accepted image judge. None was accepted, and I then chose the latent MLP judge as primary for block position, with the image probe as a sign check only. This was decided after seeing results.
- **Orientation reader:** after the same results, only the non-learned template oracle is trusted for angle. Learned-judge angle columns stay in `results/rollouts.json` but are not used.
- **Imagined-success threshold for `cem_wm_1step` on validation:** the validation run compared `cem_wm_1step`'s costs (its own encoder space) with `wm`'s τ. The final evaluation uses each encoder's own τ, so the validation imagined counts for `cem_wm_1step` are not comparable.
- **Movement-bin wording:** the first movement bin is written {{f:bin_short_label}}. All {{f:pos_pairs}} test pairs start at least {{f:succ_px}} px away, so it is effectively [{{f:succ_px}}, {{f:bin_edge1}}).
- **Bin CIs:** the first plan draft said Wilson intervals; the committed pre-registration and the results use the pair-clustered bootstrap over seeds (as the pooled rates do) with seed-{{f:seed0}} Wilson counts alongside, because Wilson intervals on pooled seeds would overstate precision.
- **Spec changes made during the build, each before the results it affects:**
  - the variance hinge;
  - foreground weighting + pusher state (chosen by the pre-registered ablation);
  - block position measured at the T's centre of gravity;
  - the final frame stored per episode;
  - a widened parameter-count tolerance for the pixel U-Net;
  - the checkpoint round-trip check relaxed from exact to `allclose` (CUDA transposed convolutions are not bit-reproducible).

## 7. Limitations and future work
Limitations of what was done:
- `cem_pixel` ran one planner seed. Its comparisons use seed {{f:pixel_seeds}} against the other methods' {{f:n_seeds}} seeds.
- The non-turning group has {{f:n_nonturning}} pairs, too few to interpret.
- One dataset, one training seed per model, 64×64 frames.
- The bins and the goal-distance sweep are exploratory.

**Not done (ideas only, no results):**
- an orientation-aware objective (e.g. an angle head or an oriented-edge loss on the block) so the latent keeps the angle;
- higher image resolution, where the T's orientation spans more pixels;
- stronger dynamics (stochastic latents, attention over longer context) to slow the drift;
- shorter planning horizons with more frequent replanning, or a cost that switches to BC near the goal;
- extra training seeds for `wm` and BC.

## 8. Reproduce
**Local (Mac, CPU, smoke only):**
```bash
uv venv --python 3.11 && uv pip install -e ".[local,dev]"
.venv/bin/python -m pytest -q
.venv/bin/python run.py --job env_check --mode smoke
```
Locally every job runs in smoke mode; `--mode full` is refused (except `make_report`).

**Kaggle (T4) via `notebooks/runner.ipynb`, three cells:**
- cell one clones or pulls the repo and installs it (`pip install -e .`, keeping Kaggle's CUDA torch);
- cell two holds `JOB`, `MODE` and `EXTRA`, the only things you edit;
- cell three runs `python run.py --job {JOB} --mode {MODE} {EXTRA}`.

Add `WANDB_API_KEY` as a Kaggle secret: artifacts (data, checkpoints, results) live in W&B project `pusht-world-model`, so a later job, or a killed and restarted one, picks them up. Run each job in **smoke** mode first, in an interactive draft session (minutes; prints projected times). Then run **full** with *Save Version → Save & Run All*, so it keeps running after the browser closes. Finished jobs with the same config are skipped; training resumes from its last checkpoint.

Order: `env_check` → `collect_data` → `wm_ablate` (chose the recipe) → `overnight` (`train_wm`, `diagnose_wm`, baselines) → `phase5b` (angle audit, judges, `eval_rollouts`) → `phase6` (`make_pairs`, `train_bc_clean`, `plan_val`, `plan_sweep`) → `plan_cost_ablation` → `phase7` (`eval_imagined_angle`, `final_eval`) → `make_report` (locally, writes this README). Run times on a T4, from each results file's `wall_clock_s`:

{{t:runtimes}}

W&B artifacts: `pusht-data`, `judge-data`, `eval-pairs`, `ckpt-wm`, `ckpt-wm-1step`, `ckpt-pixel`, `ckpt-bc`, `ckpt-bc-clean`, `ckpt-probe`, `ckpt-probe-mix`, `ckpt-latent-probe`, and one `results-*` artifact per evaluation job.

## 9. Repository layout
```
run.py               the only entry point: python run.py --job <name> --mode smoke|full
jobs.yaml            job registry (needs / makes / groups)
configs/             one YAML per job; plan.yaml holds the frozen CEM settings
wm/envs/             gym-pusht wrapper: 64x64 render, state, reset_to_state, success
wm/data/             scripted expert, data collection, GPU window samplers, judge frames
wm/models/           encoder, dynamics, decoder, world model, pixel U-Net, BC, probes
wm/train/            the shared training loop (AMP, checkpoints, resume)
wm/planning/         batched CEM and the closed-loop MPC runner
wm/eval/             rollouts, planning metrics, statistics, template oracle, README rendering
wm/jobs/             one module per job
results/             committed results JSONs; results/final/ = the test evaluation
docs/                README template and the number-source list
scripts/             README / summary regeneration and one-off diagnostics
notebooks/           the three-cell Kaggle/Colab runner
tests/               CPU tests (pytest)
```
This README is generated: `docs/README.template.md` holds the text and `python run.py --job make_report` fills in every number from `results/`. `docs/readme_sources.md` lists each number with the file and key it came from.

## References
- gym-pusht: Hugging Face, `gym-pusht` PushT environment, https://github.com/huggingface/gym-pusht.
- C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, S. Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023 (introduced the PushT task).
- R. Y. Rubinstein. *The Cross-Entropy Method for Combinatorial and Continuous Optimization.* Methodology and Computing in Applied Probability 1(2), 1999.
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson. *Learning Latent Dynamics for Planning from Pixels* (PlaNet). ICML 2019 (CEM-MPC in a learned latent space).
- A. Bardes, J. Ponce, Y. LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR 2022 (the variance term).
- K. Cho et al. *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.* EMNLP 2014 (GRU).
- O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
- Q. McNemar. *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika 12(2), 1947.
- S. Holm. *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of Statistics 6(2), 1979.
- E. B. Wilson. *Probable Inference, the Law of Succession, and Statistical Inference.* Journal of the American Statistical Association 22(158), 1927.

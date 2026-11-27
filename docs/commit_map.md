# Commit hash map

On 2026-11-27 the commit messages of the history were rewritten (an attribution
trailer was removed). File contents, authors and dates are unchanged, but every
commit after the initial one got a new hash. The results JSONs, Kaggle/W&B
artifacts, README sources and PROGRESS.md record the **old** hashes as
provenance; this table maps each to its commit in the current history.

| old hash | new hash | commit |
| --- | --- | --- |
| 528bf69 | 1dc0a1a | Phase 8: README |
| 43acbc9 | c8776e3 | Phase 7 results: final evaluation |
| 67c7ffe | 56657e6 | Phase 7: final_eval + eval_imagined_angle (pre-registered) |
| cc4aff0 | 4738173 | Phase 6b results: cost-space null |
| 406d7be | 5903629 | Phase 6b: cost_space diagnostic (pre-registered) |
| 79c5429 | ca4e96c | Phase 6 results + freeze |
| 5458942 | cda6a7e | Phase 6 smoke passed + cem_pixel pre-registration |
| 7c351ca | c062624 | Phase 5b results + judge freeze decision |
| e3b8671 | 6043504 | Phase 6: CEM planner + validation tuning |
| 0382f7a | 65894cd | Phase 5b: local smoke REPORTs in PROGRESS.md |
| 8ea9498 | 1d1f7b6 | Phase 5b: angle audit and fair judges |
| a343e14 | 96c7d8e | Phase 5: add Kaggle eval_rollouts full results and findings |
| adc200d | 5319b1a | Phase 5: rollout evaluation |
| 7065e99 | 3883bde | Phase 3b: add wm_ablate full results and table |
| 2227d6f | 649d751 | Fix: CUDA roundtrip tolerance, group continue-on-fail, ckpt hash matching |
| 3c6f9a2 | 2000e70 | Phase 3b: adopt A+B; overnight group |
| ffd775a | 989e260 | Phase 3b: world-model fixes + ablation job |
| d861a67 | 96bec0c | Phase 3: diagnose_wm job |
| 8bdbfee | f749c2e | Phase 4: baselines |
| 5dedc02 | 3421df8 | Phase 3: world model |
| 33e7fa9 | 1d749ed | PROGRESS: Phase 2 complete (Kaggle full passed) |
| ef051da | 996b5e4 | Change REPORT header to RUN REPORT |
| ecb977d | 4099ea2 | Fix: wrap block angle stats |
| f433e2b | ff3a146 | Phase 2: expert and data |
| f85bb8b | b36636d | Phase 1: foundation |
| c305e26 | c305e26 | Initial commit (unchanged) |

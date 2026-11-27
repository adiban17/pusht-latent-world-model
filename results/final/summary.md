# Final evaluation summary (full, kaggle)

Generated from `final_eval.json` (git 67c7ffe, config_hash b0356a915445) and `imagined_angle.json`. Test pairs: 200 (eval-pairs); budget 40; seeds [0, 1, 2] (cem_pixel [0]); BC arm = bc_clean (`bc_selected` from results-plan-val (git 5458942-dirty)). Frozen CEM config: N 128, H 10, rounds 4, K 12, beta 0.1.

Success = block CoG < 20 px AND |angle| < 20°, checked after every step. Position-only = CoG < 20 px after any step (pairs starting >= 20 px away). Pooled rates average over a method's seeds with a pair-clustered bootstrap 95% CI.

## Main table

| method | seeds | success, pooled [95% CI] | success seed 0 [Wilson] | position-only, pooled | turning pairs: success | non-turning pairs: success | median final pos px / angle ° | median final angle ° once position reached | imagined success (of them failed) | s / episode | s / planning call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noop | 3 | 0.0% [0.0, 0.0] | 0/200 [0.0, 1.9] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] (n 180) | 0.0% [0.0, 0.0] (n 20, low power) | 70.6 / 43.9 | n/a (n 0) | – | 0.15 | – |
| random | 3 | 10.5% [7.8, 13.3] | 24/200 [8.2, 17.2] | 17.2% [13.7, 20.8] | 8.7% [6.3, 11.3] (n 180) | 26.7% [15.0, 40.0] (n 20, low power) | 54.2 / 39.9 | 18.0 (n 103) | – | 0.14 | – |
| bc_clean | 3 | 67.5% [60.5, 74.0] | 135/200 [60.7, 73.6] | 70.0% [63.5, 76.5] | 72.8% [66.1, 79.4] (n 180) | 20.0% [5.0, 40.0] (n 20, low power) | 18.8 / 15.6 | 13.6 (n 420) | – | 0.15 | – |
| **cem_wm** | 3 | 34.7% [28.3, 41.0] | 67/200 [27.3, 40.3] | 59.3% [53.0, 65.7] | 34.8% [28.7, 41.3] (n 180) | 33.3% [13.3, 53.3] (n 20, low power) | 19.2 / 29.6 | 19.1 (n 356) | 565 (361 failed) | 0.75 | 0.022 |
| cem_wm_1step | 3 | 25.2% [19.8, 30.8] | 48/200 [18.6, 30.4] | 37.7% [31.7, 43.7] | 23.7% [18.1, 29.6] (n 180) | 38.3% [20.0, 56.7] (n 20, low power) | 31.3 / 30.7 | 17.4 (n 226) | 330 (242 failed) | 0.84 | 0.021 |
| cem_pixel | 1 | 49.5% [42.5, 56.5] | 99/200 [42.6, 56.4] | 72.0% [65.5, 78.0] | 47.2% [40.0, 54.4] (n 180) | 70.0% [50.0, 90.0] (n 20, low power) | 15.8 / 19.4 | 17.3 (n 144) | 197 (98 failed) | 10.20 | 0.437 |

τ (imagined-success threshold, val-calibrated per encoder): wm 21.218, wm_1step 16.644

## Primary tests (pre-registered)

Exact McNemar on seed 0, Holm over the 6 tests (claim if Holm p < 0.05); effect = paired bootstrap over pairs of the per-pair mean over each method's seeds. Everything else is secondary/exploratory.

| comparison | metric | seed 0 rates | discordant (a only / b only) | McNemar p | Holm p | claim | paired bootstrap difference a − b |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cem_wm vs bc_clean | success | 33.5% vs 67.5% | 15 / 83 | 1.4e-12 | 8.39e-12 | bc_clean better | -32.8 [-41.2, -24.5] pp (seeds 3 vs 3) |
| cem_wm vs cem_wm_1step | success | 33.5% vs 24.0% | 38 / 19 | 0.0163 | 0.0327 | cem_wm better | +9.5 [+2.3, +16.7] pp (seeds 3 vs 3) |
| cem_wm vs cem_pixel | success | 33.5% vs 49.5% | 11 / 43 | 1.4e-05 | 5.61e-05 | cem_pixel better | -14.8 [-21.2, -8.7] pp (seeds 3 vs 1) |
| cem_wm vs bc_clean | pos_success | 60.5% vs 70.0% | 34 / 53 | 0.053 | 0.053 | no claim (Holm p = 0.053) | -10.7 [-19.3, -2.0] pp (seeds 3 vs 3) |
| cem_wm vs cem_wm_1step | pos_success | 60.5% vs 37.0% | 68 / 21 | 5.99e-07 | 2.99e-06 | cem_wm better | +21.7 [+13.8, +29.3] pp (seeds 3 vs 3) |
| cem_wm vs cem_pixel | pos_success | 60.5% vs 72.0% | 13 / 36 | 0.0014 | 0.00421 | cem_pixel better | -12.7 [-18.8, -6.7] pp (seeds 3 vs 1) |

## Success vs goal difficulty (secondary)

Cells: success / position-only, pooled [95% CI].

**move_px**

| method | [0, 45) (n 43) | [45, 70) (n 56) | [70, 95) (n 47) | >= 95 (n 54) |
| --- | --- | --- | --- | --- |
| noop | 0.0% [0.0, 0.0] / 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] / 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] / 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] / 0.0% [0.0, 0.0] |
| random | 27.9% [20.2, 35.7] / 38.8% [30.2, 47.3] | 13.7% [8.9, 19.0] / 25.0% [17.9, 32.7] | 2.8% [0.7, 5.7] / 7.1% [3.5, 11.3] | 0.0% [0.0, 0.0] / 0.6% [0.0, 1.9] |
| bc_clean | 65.1% [51.2, 79.1] / 67.4% [53.5, 81.4] | 60.7% [48.2, 73.2] / 64.3% [51.8, 76.8] | 78.7% [66.0, 89.4] / 78.7% [66.0, 89.4] | 66.7% [53.7, 79.6] / 70.4% [57.4, 81.5] |
| cem_wm | 62.8% [48.8, 76.0] / 86.8% [76.7, 95.3] | 45.2% [33.3, 57.1] / 69.6% [58.9, 80.4] | 21.3% [10.6, 32.6] / 48.9% [36.2, 62.4] | 13.0% [5.6, 21.0] / 35.8% [24.1, 47.5] |
| cem_wm_1step | 37.2% [23.3, 51.9] / 51.2% [37.2, 65.1] | 25.0% [15.5, 35.1] / 37.5% [26.8, 49.4] | 18.4% [9.9, 28.4] / 34.8% [22.7, 46.8] | 21.6% [12.3, 32.1] / 29.6% [19.1, 41.4] |
| cem_pixel | 83.7% [72.1, 93.0] / 90.7% [81.4, 97.7] | 57.1% [44.6, 69.6] / 85.7% [76.8, 94.6] | 34.0% [21.3, 48.9] / 72.3% [59.6, 85.1] | 27.8% [16.7, 40.7] / 42.6% [29.6, 55.6] |

**turn_deg** (absolute)

| method | [0, 20) (n 20) | [20, 40) (n 74) | [40, 60) (n 46) | [60, 180] (n 60) |
| --- | --- | --- | --- | --- |
| noop | 0.0% [0.0, 0.0] / 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] / 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] / 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] / 0.0% [0.0, 0.0] |
| random | 26.7% [15.0, 40.0] / 31.7% [20.0, 45.0] | 18.5% [13.5, 23.9] / 27.9% [21.6, 34.2] | 4.3% [1.4, 8.0] / 13.8% [7.2, 21.0] | 0.0% [0.0, 0.0] / 1.7% [0.0, 3.9] |
| bc_clean | 20.0% [5.0, 40.0] / 25.0% [5.0, 45.0] | 73.0% [62.2, 82.4] / 74.3% [63.5, 83.8] | 69.6% [56.5, 82.6] / 73.9% [60.9, 87.0] | 75.0% [63.3, 85.0] / 76.7% [65.0, 86.7] |
| cem_wm | 33.3% [13.3, 53.3] / 63.3% [41.7, 83.3] | 48.6% [38.3, 59.5] / 66.7% [56.3, 76.6] | 39.9% [27.5, 52.9] / 61.6% [48.6, 73.9] | 13.9% [6.7, 22.2] / 47.2% [35.6, 58.9] |
| cem_wm_1step | 38.3% [20.0, 56.7] / 43.3% [23.3, 63.3] | 31.1% [22.1, 41.0] / 41.9% [32.0, 51.8] | 24.6% [13.8, 36.2] / 45.7% [33.3, 58.0] | 13.9% [6.1, 22.2] / 24.4% [14.4, 35.0] |
| cem_pixel | 70.0% [50.0, 90.0] / 85.0% [70.0, 100.0] | 70.3% [59.5, 79.7] / 83.8% [74.3, 91.9] | 47.8% [34.8, 60.9] / 78.3% [65.2, 89.1] | 18.3% [8.3, 28.3] / 48.3% [35.0, 61.7] |

## Failure tags (from the final state)

| method | success: all seeds (seed 0) | position_only: all seeds (seed 0) | angle_only: all seeds (seed 0) | both_failed: all seeds (seed 0) | believed success but failed |
| --- | --- | --- | --- | --- | --- |
| noop | 0 (0) | 0 (0) | 60 (20) | 540 (180) | – |
| random | 63 (24) | 18 (3) | 56 (20) | 463 (153) | – |
| bc_clean | 405 (135) | 6 (2) | 69 (23) | 120 (40) | – |
| cem_wm | 208 (67) | 116 (43) | 39 (13) | 237 (77) | 361 |
| cem_wm_1step | 151 (48) | 42 (13) | 73 (22) | 334 (117) | 242 |
| cem_pixel | 99 (99) | 38 (38) | 5 (5) | 58 (58) | 98 |

Gallery (seed 0, up to 4 episodes per tag picked by `default_rng(0)`; columns start | goal | final | decoded terminal imagined frame of the last plan (cem_wm only)): `gallery/cem_wm_success.png` pairs [45, 108, 122, 168]; `gallery/cem_wm_position_only.png` pairs [62, 103, 116, 145]; `gallery/cem_wm_angle_only.png` pairs [56, 127, 128, 129]; `gallery/cem_wm_both_failed.png` pairs [53, 83, 117, 161]; `gallery/bc_clean_success.png` pairs [48, 108, 130, 173]; `gallery/bc_clean_position_only.png` pairs [2, 101]; `gallery/bc_clean_angle_only.png` pairs [60, 90, 93, 145]; `gallery/bc_clean_both_failed.png` pairs [65, 95, 117, 144]

## Imagined angle (template oracle on imagined frames)

Source: `imagined_angle.json` (full, kaggle, git 67c7ffe); 500 test windows; template oracle, candidates every 5°.

**Search reader (primary)**, mean angle error ° [95% CI] (share of windows at the shift limit)

| row | k = 1 | k = 5 | k = 10 |
| --- | --- | --- | --- |
| real | 1.3 [1.2, 1.4] (edge 0%) | 1.2 [1.2, 1.3] (edge 0%) | 1.3 [1.2, 1.3] (edge 0%) |
| wm_recon | 5.4 [4.2, 6.9] (edge 0%) | 5.8 [4.3, 7.4] (edge 0%) | 5.7 [4.3, 7.3] (edge 0%) |
| wm | 7.0 [5.6, 8.7] (edge 0%) | 36.2 [32.1, 40.7] (edge 0%) | 60.2 [55.8, 65.6] (edge 0%) |
| wm_1step | 6.3 [4.9, 7.9] (edge 0%) | 40.8 [36.1, 45.8] (edge 0%) | 69.9 [65.1, 75.0] (edge 3%) |
| pixel | 1.9 [1.7, 2.0] (edge 0%) | 10.4 [8.2, 12.8] (edge 0%) | 24.6 [21.0, 28.5] (edge 0%) |
| copy_last | 3.4 [2.6, 4.4] (edge 0%) | 28.4 [24.3, 32.7] (edge 1%) | 46.7 [41.8, 51.9] (edge 5%) |
| copy_last_exact | 1.8 [1.6, 2.1] | 8.8 [8.0, 9.7] | 17.0 [15.7, 18.4] |

**True-position reader (secondary)**, mean angle error ° [95% CI]

| row | k = 1 | k = 5 | k = 10 |
| --- | --- | --- | --- |
| real | 1.3 [1.2, 1.4] | 1.2 [1.2, 1.3] | 1.3 [1.2, 1.3] |
| wm_recon | 5.0 [3.9, 6.5] | 5.1 [3.8, 6.6] | 5.1 [3.7, 6.6] |
| wm | 7.7 [6.0, 9.5] | 36.3 [32.1, 40.8] | 59.2 [54.4, 64.7] |
| wm_1step | 6.8 [5.3, 8.4] | 40.9 [36.4, 46.1] | 67.4 [62.5, 72.2] |
| pixel | 1.9 [1.8, 2.1] | 9.9 [7.7, 12.3] | 24.6 [20.9, 28.5] |
| copy_last | 3.5 [2.7, 4.4] | 38.6 [33.8, 43.6] | 56.5 [51.5, 61.9] |
| copy_last_exact | 1.8 [1.6, 2.1] | 8.8 [8.0, 9.7] | 17.0 [15.7, 18.4] |

**Oracle-found shift** (world px, mean / median)

| row | k = 1 | k = 5 | k = 10 |
| --- | --- | --- | --- |
| real | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| wm_recon | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| wm | 0.1 / 0.0 | 0.7 / 0.0 | 1.8 / 0.0 |
| wm_1step | 0.1 / 0.0 | 1.3 / 0.0 | 4.4 / 0.0 |
| pixel | 0.0 / 0.0 | 0.1 / 0.0 | 0.3 / 0.0 |
| copy_last | 0.7 / 0.0 | 5.2 / 0.0 | 7.0 / 0.0 |

**Paired differences** (search reader, °, wm − other)

| difference | k = 1 | k = 5 | k = 10 |
| --- | --- | --- | --- |
| wm_minus_copy_last | +3.7 [+2.1, +5.4] | +7.8 [+2.5, +13.4] | +13.5 [+6.9, +20.5] |
| wm_minus_wm_1step | +0.8 [-1.1, +2.7] | -4.6 [-10.0, +0.4] | -9.6 [-16.0, -3.1] |
| wm_minus_pixel | +5.2 [+3.8, +6.8] | +25.8 [+21.5, +30.4] | +35.7 [+30.4, +41.4] |
| wm_minus_wm_recon | +1.6 [-0.0, +3.4] | +30.5 [+26.2, +35.3] | +54.5 [+49.6, +59.8] |
| wm_minus_copy_last_exact | +5.2 [+3.8, +6.8] | +27.4 [+23.4, +31.8] | +43.2 [+38.7, +48.6] |

**Verdict (pre-registered, k = 10): imagined rollouts lose the angle: confirmed** (A: rollout adds error beyond wm_recon = True; B: worse than copy_last = True, search True, exact True).

## Goal-distance sweep (exploratory)

| goal offset | pairs | candidates | median move px | median |turn| ° | cem_wm: success / position-only (seed 0) | bc_clean: success / position-only (seed 0) |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 100 | 228 | 48.2 | 28.2 | 64/100 [54.2, 72.7] / 85/100 [76.7, 90.7] | 83/100 [74.5, 89.1] / 84/100 [75.6, 89.9] |
| 30 | 100 | 380 | 93.0 | 56.0 | 20/100 [13.3, 28.9] / 56/100 [46.2, 65.3] | 51/100 [41.3, 60.6] / 52/100 [42.3, 61.5] |

## Speed

| method | s / episode (mean) | s / full-size planning call |
| --- | --- | --- |
| noop | 0.15 | – |
| random | 0.14 | – |
| bc_clean | 0.15 | – |
| cem_wm | 0.75 | 0.022 |
| cem_wm_1step | 0.84 | 0.021 |
| cem_pixel | 10.20 | 0.437 |

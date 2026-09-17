# Camera-ready changes in response to review (submission 196)

Reviews: 2x strong accept, 1x weak accept with methodological concerns
(reviewer 3). This document tracks how each point in `feedback.md` was
addressed for the camera-ready.

## 1. Random-beats-Greedy anomaly (reviewer 3, priority fix)

**Diagnosis.** The environment's oversubscription rescale
(`_apply_curtailment` in `experiment/campus_env.py`) forces
`sum(served) = avail` whenever a controller's requested service exceeds
available power, *identically regardless of how much curtailment was
requested*. The original `GreedyFormula` controller (kept in the code for
comparison) chose each cluster's curtailment level by rounding the exact
continuous deficit-closing fraction to the nearest of six discrete levels.
That rounding has no consistent bias direction: under-requesting curtailment
is corrected for free by the rescale (no reward penalty), but
over-requesting directly and irreversibly wastes deliverable energy. A
controller that samples curtailment broadly (e.g. Random) is more likely to
land in the "free rescale" regime than a controller specifically tuned to
sit at the boundary — exactly where the rounding penalty bites hardest.

**Empirical confirmation** (final equalised-budget run, `experiment/results/summary.csv` +
`experiment/diagnose_results.json`, real dataset, 60 paired episodes):

| Controller | Reward | Oversubscribed steps | Wasted kW when not oversubscribed |
|---|---|---|---|
| Random | 132.65 | 58.3% | 42.2 |
| RoundRobin | 119.59 | 58.2% | 40.9 |
| StaticPriority | 104.17 | 53.4% | 43.1 |
| LowestPriorityFirst | 143.55 | 73.1% | 27.5 |
| GreedyFormula (original) | 125.48 | 56.3% | **53.5** |
| **Greedy (exact lookahead, fixed)** | **138.90** | 60.8% | n/a |

GreedyFormula wastes the most capacity of any controller when it isn't
rescued by the auto-clamp — direct evidence of the rounding bias. Critically,
GreedyFormula *still* trails Random under the fully corrected, equalised-budget
protocol used for the final numbers above, which rules out the training-budget
confound (point 2 below) as an alternative explanation — the effect persists
even though GreedyFormula is a rule-based heuristic with no training budget at
all.

**Fix.** `GreedyCtrl` in `experiment/campus_env.py` now enumerates all 216
discrete actions at each step, applies the environment's own reward
function to each (vectorised for speed — no Python loop, no retraining
involved), and picks the reward-maximising one. This is the correct
operationalisation of "greedy" and removes the discretisation bias entirely.
With the fix, Greedy (138.90) beats Random (132.65) again, resolving the
anomaly.

The original `GreedyFormula` is kept in the codebase (not deleted) so the
before/after comparison above is reproducible.

**Unexpected downstream consequence: the per-regime finding reverses.** The
original paper's secondary headline claim was that SAC's advantage over Greedy
is largest under Emergency-level deficit (2.6x). With the corrected Greedy,
this reverses: Greedy now leads SAC in 4 of 5 regimes, *including* Emergency
(47.83 vs 45.26) — see `experiment/results/sensitivity.csv`. The mechanism is
the same rounding bias: it scales with the size of the required cut, so it hurt
GreedyFormula most precisely under severe deficit, which is exactly the
condition the original paper highlighted as RL's strongest use case. This is
reported transparently in the camera-ready rather than preserved.

## 2. Unequal training budgets

All 14 RL variants now train for an equalised 60,000 environment steps
(`CFG.TRAIN_TIMESTEPS` in `experiment/campus_env.py`), removing the
30k-vs-60k split between "top-tier" and other variants.

## 3. Distinct test-episode diversity

`experiment/evaluate.py` now reports, alongside the 60-episode paired
comparison, how many distinct start-hours the 60 evaluation episodes draw
from out of the total number of possible 168h windows in the dataset, and
how many adjacent episode pairs have overlapping demand windows. See
`results/meta.json` -> `n_distinct_windows`.

Result: 59 of 60 episodes land on distinct start hours (1 exact collision)
out of 3,479 possible start hours in the 5-month dataset, but 56 of the 59
sorted-adjacent episode pairs have starts within 168h of each other and thus
temporally overlapping demand histories — disclosed as a residual limitation
in the paper (Section IV) rather than resolved, since fixing it would need
either more historical data than is available or synthetic extrapolation.

## 4. Single training seed

The five headline controllers (SAC, TD3, PPO-Tuned, MaskablePPO, PPO-Large)
are trained across 5 seeds (42-46) and reported as mean +/- across-seed
standard deviation (`results/multiseed_summary.csv`), in addition to the
primary seed-42 comparison table.

**This surfaced a real finding, not just a robustness check.** On seed 42
(used for the headline pairwise comparison), SAC (141.44) beats TD3 (135.60).
Averaged across all 5 seeds, TD3 (145.01 +/- 6.02) is nominally *above* SAC
(138.56 +/- 4.67) — the ranking of the two strongest continuous-control
methods flips depending on which seed (or how many) you report. This is
disclosed explicitly in the paper rather than only reporting the
seed-42 snapshot.

## 5. No multiple-comparison correction

`experiment/evaluate.py` applies a Holm-Bonferroni step-down correction
(`holm_correction`) to the family of pairwise paired t-tests between the
best RL controller and every other controller. Reported as `p_holm` in
`results/stat_tests.csv`.

## 6. Reward-weight sensitivity

Carried forward from the original notebook's Section 16
(`eval_with_weights` grid over w_C, w_F), re-run against the corrected,
equalised-budget best RL agent. See `results/weight_sensitivity.csv` and
`paper/figures/fig5_weight_heatmap.png`.

## 7. Eight feeders vs. three clusters

Clarified in the paper text (Discussion): the 8 feeders describe physical
wiring topology (retained for dataset provenance); the 3 clusters are the
operational control granularity actually used by the campus Works and
Physical Planning Department for shedding policy. These are not competing
granularities.

## 8. PPO-BC cloning accuracy / Hybrid underperformance

Not separately re-diagnosed for the camera-ready (noted as a direction for
follow-up work; out of scope given the review's priority ordering).

## Other corrections found during the rerun (not reviewer-flagged)

- Paper text said "six months (4344 hours)" / mean demand 386 kW; the
  actual `final.csv` provided for this rerun spans 1 Jan - 31 May 2024
  (3,648 hours, ~5 months) with mean total demand 240.6 kW. Text corrected
  to match the real dataset (confirmed with the author as the same file
  used for the original submission).

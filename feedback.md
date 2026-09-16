SUBMISSION: 196
TITLE: Continuous-Action Reinforcement Learning for Resilient Campus Microgrid Load Shedding

----------------------- REVIEW 1 ---------------------

SUBMISSION: 196
TITLE: Continuous-Action Reinforcement Learning for Resilient Campus Microgrid Load Shedding

----------- Overall evaluation -----------
SCORE: 3 (strong accept)
----- TEXT:
The paper presents a rigorous, well‑executed, and highly relevant contribution to reinforcement learning for microgrid resilience. It is methodologically strong, experimentally thorough, and clearly written. The work is mature, reproducible, and addresses a real‑world Nigerian campus microgrid with statistically grounded benchmarking. This is exactly the type of high‑quality research IEEE NIGERCON aims to showcase.

Detailed Review and Justification
1. Technical Quality and Soundness — Excellent
The paper demonstrates exceptional technical rigor:

Benchmarks 14 RL algorithms and 5 deterministic controllers under identical conditions.

Uses three OpenAI Gym environments to isolate the effect of action‑space representation.

Provides paired statistical tests, confidence intervals, and effect sizes (Cohen’s d).

Implements SAC, TD3, PPO variants, DQN, QR‑DQN, TRPO, A2C, and hybrid controllers.

Uses a six‑month real dataset from the University of Lagos campus microgrid.

Includes a well‑defined MDP, reward function, fairness metric (Jain’s index), and supply‑deficit regimes.

The methodology is not only sound but exemplary in its completeness and reproducibility.

2. Novelty and Contribution — Very High
The paper closes three major gaps in prior literature:

First benchmark to compare RL against the operator‑grade Greedy controller.

First to isolate action‑space representation (discrete vs continuous) as the decisive factor.

First to evaluate controllers under deep Emergency‑level deficits, where adaptive control matters most.

The finding that continuous‑action SAC matches deterministic baselines and outperforms Greedy by 2.6× in Emergency conditions is a significant and novel insight.

3. Relevance to NIGERCON — Excellent
The work directly addresses:

African energy resilience

Microgrid control

Reinforcement learning applications

Real‑world Nigerian infrastructure challenges

This is highly aligned with NIGERCON’s mission to highlight impactful engineering research for the African context.

4. Experimental Evaluation — Outstanding
The evaluation is one of the strongest aspects of the paper:

60 paired episodes across five deficit regimes

Detailed per‑regime analysis (Normal → Emergency)

Clear demonstration of SAC’s robustness under severe deficits

Fairness metrics included

Training reproducibility ensured via fixed seeds

Computational feasibility demonstrated (21‑minute SAC training)

This is a model example of how RL research should be evaluated.

5. Clarity and Organization — Excellent
The paper is exceptionally well written:

Clear motivation grounded in Nigerian grid realities

Strong literature review covering RL, load shedding, and action‑space design

Well‑structured MDP formulation

Clean tables and figures

Logical flow from problem → method → results → implications

The writing is concise, technical, and professional.

6. Weaknesses — Minor
Only minor points:

The paper is long and dense; some sections could be slightly condensed.

A brief discussion on real‑world deployment challenges (sensor noise, communication delays) would strengthen the practical perspective.

The fairness term could be discussed more deeply in relation to campus policies.

None of these detract from the overall quality.


----------------------- REVIEW 2 ---------------------

SUBMISSION: 196
TITLE: Continuous-Action Reinforcement Learning for Resilient Campus Microgrid Load Shedding

----------- Overall evaluation -----------
SCORE: 3 (strong accept)
----- TEXT:
Highly Recommended


----------------------- REVIEW 3 ---------------------

SUBMISSION: 196
TITLE: Continuous-Action Reinforcement Learning for Resilient Campus Microgrid Load Shedding

----------- Overall evaluation -----------
SCORE: -1 (weak accept)
----- TEXT:
This is an ambitious and well-instrumented benchmark — fourteen RL algorithms against five heuristics, evaluated on a real 55-building dataset with paired statistical testing and a genuinely useful ablation (isolating action-space representation from algorithm choice). The core finding (continuous action space matters more than algorithm family) is plausible and interesting. However, several results in the paper — most notably the Random baseline's ranking — point to a possible fundamental flaw in the environment/reward design that undermines the headline claims, and there are confounds in the experimental setup that weaken the causal story.

The core anomaly (Random baseline outperforming the Greedy heuristic and 11/14 RL variants) is not a minor caveat — it undermines the validity of the entire benchmark's headline comparisons. Combined with the training-budget confound directly affecting the paper's star result (SAC), the central claim ("action representation, not algorithm, is decisive") is not currently well-supported by the evidence presented, even though the underlying idea is credible and worth pursuing.


 Overview
- Benchmarks 14 RL algorithms vs. 5 rule-based controllers for campus microgrid load shedding, using a real 55-building Nigerian dataset (University of Lagos).
- Central claim: continuous action space (SAC) matches the best heuristic and beats Greedy by a wide margin, especially under severe supply deficit (Emergency regime).
- Well-instrumented: paired statistical tests, per-regime breakdown, open Gym environment with 3 action-space variants isolating one design axis.

 Strengths
- Real, non-synthetic case study grounded in an actual under-resourced grid (strong motivation and relevance).
- Genuinely useful experimental design: identical dynamics across three action-space flavors isolates representation from algorithm choice.
- Includes operator-grade Greedy as a baseline — often missing in prior RL-for-load-shedding literature.
- Per-regime (Normal→Emergency) breakdown surfaces where adaptive control actually helps, rather than relying on aggregate reward alone.
- Reproducibility details provided (fixed seeds, stable-baselines3, step counts, replay buffer size, etc.).

Weaknesses
1. Random policy beats Greedy and most RL variants (134.4 vs. 128.3, near-perfect fairness) — strongly suggests the reward/environment (oversubscription rescaling) makes served power largely insensitive to the chosen action, confounding the whole benchmark.
2. Unequal training budgets: SAC/TD3/PPO-Tuned get 60k steps vs. 30k for other discrete variants — directly confounds the "continuous > discrete" conclusion with a "more training > less training" effect.
3. Suspiciously large effect sizes (Cohen's d = 10.23) imply near-zero variance in paired differences, likely from limited distinct episode/demand-window diversity across the 60 test episodes.
4. Single training seed (42) for all 14 RL variants — no seed-robustness reporting, a near-minimum standard for RL benchmarking claims.
5. No multiple-comparison correction across ~18 pairwise significance tests against SAC.
6. Reward weights unjustified, no sensitivity analysis — plausible driver of the Random-beats-Greedy anomaly.
7. Eight feeders mentioned in intro but silently dropped in favor of 3 clusters — unexplained granularity mismatch.
8. PPO-BC's 33% cloning accuracy and the Hybrid controller's underperformance are noted but not diagnosed.

Suggestions
1. Priority fix: Explain/resolve why Random outperforms Greedy — audit whether oversubscription rescaling neutralizes action choice; this must be addressed before the benchmark's conclusions can be trusted.
2. Equalize training steps across all RL variants, or explicitly ablate step count as its own variable.
3. Report how many distinct demand trajectories the 60 test episodes actually span; if reused, add more held-out weeks or randomize demand windows too.
4. Train key controllers (SAC, TD3, top PPO variants) across ≥3–5 seeds and report variance.
5. Apply a Bonferroni/Holm/FDR correction to pairwise tests.
6. Add a reward-weight sensitivity analysis, at least for ωC and ωF, given their apparent influence on the anomalous rankings.
7. Clarify the 8-feeder vs. 3-cluster relationship, or remove the feeder framing from the introduction if unused.
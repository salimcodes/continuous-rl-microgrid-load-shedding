"""Cheap diagnostic: heuristic-only comparison, no NN training.

Reproduces the Random-vs-Greedy anomaly with the ORIGINAL formula-based
Greedy, quantifies the oversubscription mechanism, then shows the effect of
swapping in the exact one-step-lookahead Greedy.
"""
import json
import numpy as np
from campus_env import (CFG, load_data, CampusEnvDiscrete,
                         RandomCtrl, RoundRobinCtrl, StaticPriorityCtrl,
                         LowestPriorityFirstCtrl, GreedyFormulaCtrl, GreedyCtrl)

df, cluster_loads, priority_df, building_cols = load_data(CFG)
print(f"data span: {df.Timestamp.min()} -> {df.Timestamp.max()}  ({len(df)} rows)")
print(f"mean total demand: {cluster_loads['TotalDemand'].mean():.1f} kW")

results = {}
diag = {}

# --- original formula-based Greedy, with diagnostics on --------------------
env = CampusEnvDiscrete(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001, diagnostics=True)
for cls in [RandomCtrl, RoundRobinCtrl, StaticPriorityCtrl, LowestPriorityFirstCtrl, GreedyFormulaCtrl]:
    env.diag_log = []
    ctrl = cls(env)
    m = ctrl.evaluate(n_episodes=CFG.EVAL_EPISODES, seed=CFG.RANDOM_SEED + 12345)
    results[ctrl.name] = {k: float(np.mean(v)) for k, v in m.items()} | {k + "_std": float(np.std(v)) for k, v in m.items()}
    log = env.diag_log
    n = len(log)
    oversub_frac = np.mean([d["oversubscribed"] for d in log])
    mean_slack = np.mean([d["slack"] for d in log if not d["oversubscribed"]]) if any(not d["oversubscribed"] for d in log) else 0.0
    diag[ctrl.name] = dict(oversub_frac=float(oversub_frac), mean_wasted_kw_when_undersub=float(mean_slack), n_steps=n)
    print(f"{ctrl.name:20s} reward={results[ctrl.name]['reward']:8.3f}  "
          f"oversub%={oversub_frac*100:5.1f}  wasted_kW_when_not_oversub={mean_slack:6.2f}")

# --- exact lookahead Greedy --------------------------------------------------
env2 = CampusEnvDiscrete(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001, diagnostics=True)
env2.diag_log = []
greedy2 = GreedyCtrl(env2)
m2 = greedy2.evaluate(n_episodes=CFG.EVAL_EPISODES, seed=CFG.RANDOM_SEED + 12345)
results["Greedy"] = {k: float(np.mean(v)) for k, v in m2.items()} | {k + "_std": float(np.std(v)) for k, v in m2.items()}
log2 = env2.diag_log
oversub_frac2 = np.mean([d["oversubscribed"] for d in log2]) if log2 else float("nan")
diag["Greedy"] = dict(oversub_frac=float(oversub_frac2), n_steps=len(log2))
print(f"{'Greedy (exact)':20s} reward={results['Greedy']['reward']:8.3f}  oversub%={oversub_frac2*100:5.1f}")

with open("diagnose_results.json", "w") as f:
    json.dump(dict(results=results, diag=diag,
                    data_span=[str(df.Timestamp.min()), str(df.Timestamp.max())],
                    mean_demand=float(cluster_loads["TotalDemand"].mean())), f, indent=2)
print("\nSaved diagnose_results.json")

"""Regenerate the paper's figures from experiment/results/*.csv."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
OUT = Path("../paper/figures")
OUT.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                      "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

summary = pd.read_csv(RESULTS / "summary.csv")

# --- Fig: reward by controller, ranked -------------------------------------
order = summary["Controller"].tolist()
means = summary["Reward_mean"].values
errs = 1.96 * summary["Reward_std"].values / np.sqrt(60)

def colour_for(n):
    if n == "Greedy": return "#2ca02c"
    if n == "GreedyFormula": return "#98df8a"
    if n in ["Random", "RoundRobin", "StaticPriority", "LowestPriorityFirst"]: return "#7f7f7f"
    if "PPO" in n: return "#d62728"
    if n in ["A2C", "TRPO"]: return "#ff7f0e"
    if n in ["DQN", "QR-DQN"]: return "#9467bd"
    if n in ["SAC", "TD3"]: return "#17becf"
    if "Hybrid" in n: return "#e377c2"
    return "#1f77b4"

cols = [colour_for(n) for n in order]
plt.figure(figsize=(13, 5.2))
plt.bar(order, means, yerr=errs, color=cols, edgecolor="black", capsize=4)
plt.ylabel("Mean episode reward (95% CI)")
plt.title("Reward — every controller, ranked (corrected, equalised-budget run)")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(OUT / "p10_reward_bars.png")
plt.close()

# --- Fig: per-regime reward for top-4 controllers ---------------------------
sens = pd.read_csv(RESULTS / "sensitivity.csv")
REG_ORDER = ["Normal", "Minor", "Moderate", "Major", "Emergency"]
pivot = sens.pivot(index="Scenario", columns="Controller", values="Reward").reindex(REG_ORDER)
plt.figure(figsize=(9, 5))
pivot.plot(kind="bar", ax=plt.gca(), edgecolor="black")
plt.ylabel("Mean episode reward")
plt.title("Reward by deficit regime — top 4 controllers")
plt.xticks(rotation=20)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT / "fig2_regime_reward.png")
plt.close()

# --- Fig: multi-seed robustness for headline controllers --------------------
ms = pd.read_csv(RESULTS / "multiseed.csv")
plt.figure(figsize=(9, 5))
ctrls = ms["Controller"].unique().tolist()
data = [ms.loc[ms.Controller == c, "RewardMean"].values for c in ctrls]
try:
    bp = plt.boxplot(data, tick_labels=ctrls, patch_artist=True)
except TypeError:
    bp = plt.boxplot(data, labels=ctrls, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#17becf")
plt.ylabel("Per-seed mean episode reward")
plt.title("Seed-robustness — 5 seeds per headline controller")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT / "fig3_seed_robustness.png")
plt.close()

# --- Fig: oversubscription audit (Greedy vs GreedyFormula vs Random) -------
import json
diag = json.load(open(RESULTS.parent / "diagnose_results.json")) if (RESULTS.parent / "diagnose_results.json").exists() else None
if diag:
    names = list(diag["diag"].keys())
    oversub = [diag["diag"][n]["oversub_frac"] * 100 for n in names]
    plt.figure(figsize=(8, 4.5))
    plt.bar(names, oversub, color="#9467bd", edgecolor="black")
    plt.ylabel("Steps oversubscribed (%)")
    plt.title("Oversubscription frequency by controller")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "fig4_oversub_audit.png")
    plt.close()

# --- Fig: reward-weight sensitivity heatmap ---------------------------------
import seaborn as sns
heat = pd.read_csv(RESULTS / "weight_sensitivity.csv")
piv = heat.pivot(index="w_critical", columns="w_fairness", values="reward")
plt.figure(figsize=(7, 5))
sns.heatmap(piv, annot=True, fmt=".2f", cmap="magma", cbar_kws={"label": "Mean reward"})
plt.title("Reward sensitivity to (w_C, w_F) — best RL agent")
plt.xlabel("w_F"); plt.ylabel("w_C")
plt.tight_layout()
plt.savefig(OUT / "fig5_weight_heatmap.png")
plt.close()

print("Figures written to", OUT.resolve())

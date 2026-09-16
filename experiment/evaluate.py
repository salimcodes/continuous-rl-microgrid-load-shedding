"""Evaluate every controller on the same 60 paired episodes, run stats with
Holm correction, per-regime sensitivity, reward-weight sensitivity, and
diagnostic reporting. Reads models trained by train.py.
"""
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
try:
    from sb3_contrib import QRDQN, TRPO, RecurrentPPO, MaskablePPO
    SB3_CONTRIB = True
except Exception:
    SB3_CONTRIB = False

from campus_env import (CFG, load_data, CampusEnvDiscrete, CampusEnvContinuous,
                         CampusEnvMasked, PowerScenario, jain,
                         RandomCtrl, RoundRobinCtrl, StaticPriorityCtrl,
                         LowestPriorityFirstCtrl, GreedyCtrl, GreedyFormulaCtrl)
from controllers_rl import RLCtrl, HybridController
from train import MODEL_DIR, HEADLINE_MULTISEED, SEEDS_MULTI, model_path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df, cluster_loads, priority_df, building_cols = load_data(CFG)

MODEL_CLASS = {
    "PPO": PPO, "PPO-Large": PPO, "PPO-Curriculum": PPO, "PPO-Tuned": PPO, "PPO-BC": PPO,
    "A2C": A2C, "DQN": DQN, "SAC": SAC, "TD3": TD3,
}
ENV_CLASS = {
    "PPO": CampusEnvDiscrete, "PPO-Large": CampusEnvDiscrete, "PPO-Curriculum": CampusEnvDiscrete,
    "PPO-Tuned": CampusEnvDiscrete, "PPO-BC": CampusEnvDiscrete, "A2C": CampusEnvDiscrete,
    "DQN": CampusEnvDiscrete, "SAC": CampusEnvContinuous, "TD3": CampusEnvContinuous,
}
if SB3_CONTRIB:
    MODEL_CLASS.update({"QR-DQN": QRDQN, "RecurrentPPO": RecurrentPPO,
                         "MaskablePPO": MaskablePPO, "TRPO": TRPO})
    ENV_CLASS.update({"QR-DQN": CampusEnvDiscrete, "RecurrentPPO": CampusEnvDiscrete,
                       "MaskablePPO": CampusEnvMasked, "TRPO": CampusEnvDiscrete})

ALL_RL_NAMES = list(MODEL_CLASS.keys())


def load_model(name, seed):
    path = model_path(name, seed)
    if not path.exists():
        return None
    return MODEL_CLASS[name].load(str(path))


def cohens_d(x, y):
    diff = x - y
    return float(diff.mean() / (diff.std(ddof=1) + 1e-9))


def ci95(x):
    n = len(x); m = x.mean(); s = x.std(ddof=1)
    h = s * st.t.ppf(0.975, n - 1) / np.sqrt(n)
    return float(m - h), float(m + h)


def holm_correction(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the original order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj


def main():
    results = {}
    n_distinct_windows = {}

    # ---------------- heuristics (deterministic, one pass) -----------------
    eval_env = CampusEnvDiscrete(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001)
    heuristics = [RandomCtrl, RoundRobinCtrl, StaticPriorityCtrl, LowestPriorityFirstCtrl]
    for cls in heuristics:
        ctrl = cls(eval_env)
        print(f"[eval] {ctrl.name}")
        results[ctrl.name] = ctrl.evaluate(n_episodes=CFG.EVAL_EPISODES, seed=CFG.RANDOM_SEED + 12345)

    greedy_env = CampusEnvDiscrete(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001)
    print("[eval] Greedy (exact lookahead)")
    results["Greedy"] = GreedyCtrl(greedy_env).evaluate(n_episodes=CFG.EVAL_EPISODES, seed=CFG.RANDOM_SEED + 12345)
    gf_env = CampusEnvDiscrete(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001)
    print("[eval] GreedyFormula (original, kept for the audit comparison)")
    results["GreedyFormula"] = GreedyFormulaCtrl(gf_env).evaluate(n_episodes=CFG.EVAL_EPISODES, seed=CFG.RANDOM_SEED + 12345)

    # number of distinct (non-identical) 168h start windows the 60 episodes span
    starts = []
    probe_env = CampusEnvDiscrete(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001)
    for ep in range(CFG.EVAL_EPISODES):
        probe_env.reset(seed=CFG.RANDOM_SEED + 12345 + ep)
        starts.append(probe_env._cursor)
    n_distinct_windows["n_episodes"] = CFG.EVAL_EPISODES
    n_distinct_windows["n_distinct_start_hours"] = int(len(set(starts)))
    max_start = len(cluster_loads) - CFG.EPISODE_LENGTH - 1
    n_distinct_windows["n_possible_start_hours"] = int(max_start)
    # overlap: two 168h windows starting within 168h of each other share hours
    starts_sorted = sorted(starts)
    overlapping_pairs = sum(1 for i in range(1, len(starts_sorted))
                             if starts_sorted[i] - starts_sorted[i - 1] < CFG.EPISODE_LENGTH)
    n_distinct_windows["overlapping_adjacent_pairs"] = int(overlapping_pairs)

    # ---------------- RL variants: seed-42 primary run ----------------------
    for name in ALL_RL_NAMES:
        m = load_model(name, 42)
        if m is None:
            print(f"[warn] no seed-42 model found for {name}, skipping")
            continue
        env_cls = ENV_CLASS[name]
        env = env_cls(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001)
        print(f"[eval] {name} (seed 42)")
        results[name] = RLCtrl(env, m, name).evaluate(n_episodes=CFG.EVAL_EPISODES, seed=CFG.RANDOM_SEED + 12345)

    # PPO-Curriculum / PPO-BC (trained via bespoke loops, still PPO class / discrete env)
    for name in ["PPO-Curriculum", "PPO-BC"]:
        m = load_model(name, 42)
        if m is None:
            continue
        env = CampusEnvDiscrete(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001)
        print(f"[eval] {name}")
        results[name] = RLCtrl(env, m, name).evaluate(n_episodes=CFG.EVAL_EPISODES, seed=CFG.RANDOM_SEED + 12345)

    # Hybrid uses PPO-Tuned seed-42 model + exact Greedy
    ppo_tuned_42 = load_model("PPO-Tuned", 42)
    if ppo_tuned_42 is not None:
        h_env = CampusEnvDiscrete(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001)
        print("[eval] Hybrid-PPO+Greedy")
        results["Hybrid-PPO+Greedy"] = HybridController(h_env, ppo_tuned_42).evaluate(
            n_episodes=CFG.EVAL_EPISODES, seed=CFG.RANDOM_SEED + 12345)

    # ---------------- summary table -----------------------------------------
    summary = []
    for name, m in results.items():
        summary.append(dict(Controller=name, Reward_mean=float(m["reward"].mean()),
                             Reward_std=float(m["reward"].std()),
                             Served_mean=float(m["served"].mean()),
                             Curtail_mean=float(m["curtail"].mean()),
                             Jain_mean=float(m["fairness"].mean()),
                             Runtime_mean=float(m["runtime"].mean())))
    summary_df = pd.DataFrame(summary).sort_values("Reward_mean", ascending=False).reset_index(drop=True)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(summary_df.round(3))

    # ---------------- stats: best RL vs everyone, Holm-corrected -----------
    rl_only = [n for n in ALL_RL_NAMES + ["PPO-Curriculum", "PPO-BC", "Hybrid-PPO+Greedy"] if n in results]
    best_rl = max(rl_only, key=lambda n: results[n]["reward"].mean())
    best_r = results[best_rl]["reward"]
    rows, raw_p = [], []
    others = [n for n in results if n != best_rl]
    for name in others:
        other = results[name]["reward"]
        if not np.any(best_r - other):
            rows.append(dict(Baseline=name, Delta=0.0, t=np.nan, p_raw=1.0, d=0.0))
            raw_p.append(1.0)
            continue
        t_stat, t_p = st.ttest_rel(best_r, other)
        rows.append(dict(Baseline=name, Delta=float((best_r - other).mean()),
                          t=float(t_stat), p_raw=float(t_p), d=cohens_d(best_r, other)))
        raw_p.append(float(t_p))
    adj = holm_correction(raw_p)
    for row, a in zip(rows, adj):
        row["p_holm"] = float(a)
        row["sig_holm_0.05"] = bool(a < 0.05)
    stat_df = pd.DataFrame(rows).sort_values("Delta", ascending=False)
    stat_df.to_csv(RESULTS_DIR / "stat_tests.csv", index=False)
    print(f"\nBest RL controller: {best_rl}  (mean reward {best_r.mean():.3f})")
    print(stat_df.round(4))

    # ---------------- multi-seed robustness for headline controllers -------
    ms_rows = []
    for name in HEADLINE_MULTISEED:
        for seed in SEEDS_MULTI:
            m = load_model(name, seed)
            if m is None:
                continue
            env_cls = ENV_CLASS[name]
            env = env_cls(cluster_loads, priority_df, seed=CFG.RANDOM_SEED + 1001)
            r = RLCtrl(env, m, name).evaluate(n_episodes=CFG.EVAL_EPISODES, seed=CFG.RANDOM_SEED + 12345)["reward"]
            ms_rows.append(dict(Controller=name, Seed=seed, RewardMean=float(r.mean()), RewardStd=float(r.std())))
            print(f"[multiseed] {name} seed={seed} mean={r.mean():.3f}")
    ms_df = pd.DataFrame(ms_rows)
    ms_df.to_csv(RESULTS_DIR / "multiseed.csv", index=False)
    ms_summary = (ms_df.groupby("Controller")["RewardMean"]
                  .agg(["mean", "std", "count"]).reset_index()
                  .rename(columns={"mean": "AcrossSeedMean", "std": "AcrossSeedStd", "count": "NSeeds"}))
    ms_summary.to_csv(RESULTS_DIR / "multiseed_summary.csv", index=False)
    print(ms_summary.round(3))

    # ---------------- per-regime sensitivity (top-4) ------------------------
    top4 = []
    for n in [best_rl, "Greedy", "PPO", "StaticPriority"]:
        if n in results and n not in top4:
            top4.append(n)
    sens_rows = []
    for name in top4:
        for scen in PowerScenario.REG:
            if name in ["Random", "RoundRobin", "StaticPriority", "LowestPriorityFirst"]:
                env = CampusEnvDiscrete(cluster_loads, priority_df, scenario=scen, seed=CFG.RANDOM_SEED + 99)
                ctrl_map = {"Random": RandomCtrl, "RoundRobin": RoundRobinCtrl,
                            "StaticPriority": StaticPriorityCtrl, "LowestPriorityFirst": LowestPriorityFirstCtrl}
                ctrl = ctrl_map[name](env)
            elif name == "Greedy":
                env = CampusEnvDiscrete(cluster_loads, priority_df, scenario=scen, seed=CFG.RANDOM_SEED + 99)
                ctrl = GreedyCtrl(env)
            elif name in ("SAC", "TD3"):
                env = CampusEnvContinuous(cluster_loads, priority_df, scenario=scen, seed=CFG.RANDOM_SEED + 99)
                ctrl = RLCtrl(env, load_model(name, 42), name)
            elif name == "MaskablePPO":
                env = CampusEnvMasked(cluster_loads, priority_df, scenario=scen, seed=CFG.RANDOM_SEED + 99)
                ctrl = RLCtrl(env, load_model(name, 42), name)
            elif name == "Hybrid-PPO+Greedy":
                env = CampusEnvDiscrete(cluster_loads, priority_df, scenario=scen, seed=CFG.RANDOM_SEED + 99)
                ctrl = HybridController(env, load_model("PPO-Tuned", 42))
            else:
                env = CampusEnvDiscrete(cluster_loads, priority_df, scenario=scen, seed=CFG.RANDOM_SEED + 99)
                ctrl = RLCtrl(env, load_model(name, 42), name)
            m = ctrl.evaluate(n_episodes=15, seed=CFG.RANDOM_SEED + 4242)
            sens_rows.append(dict(Controller=name, Scenario=scen, Reward=float(m["reward"].mean()),
                                   Served=float(m["served"].mean()), Fairness=float(m["fairness"].mean())))
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)
    print(sens_df.round(3))

    # ---------------- reward-weight sensitivity for best RL -----------------
    def eval_with_weights(w_critical, w_fairness, n_eps=8):
        env_cls = (CampusEnvMasked if best_rl == "MaskablePPO" else
                   (CampusEnvContinuous if best_rl in ["SAC", "TD3"] else CampusEnvDiscrete))
        env = env_cls(cluster_loads, priority_df,
                      reward_weights={"critical": w_critical, "fairness": w_fairness},
                      seed=CFG.RANDOM_SEED + 50)
        model = load_model(best_rl, 42)
        return RLCtrl(env, model, best_rl).evaluate(n_episodes=n_eps, seed=CFG.RANDOM_SEED + 50)["reward"].mean()

    w_crit_grid = [0.5, 1.0, 2.0, 4.0]
    w_fair_grid = [0.0, 0.5, 1.0, 2.0]
    heat_rows = []
    for wc in w_crit_grid:
        for wf in w_fair_grid:
            val = eval_with_weights(wc, wf)
            heat_rows.append(dict(w_critical=wc, w_fairness=wf, reward=val))
    heat_df = pd.DataFrame(heat_rows)
    heat_df.to_csv(RESULTS_DIR / "weight_sensitivity.csv", index=False)

    # ---------------- persist everything ------------------------------------
    with open(RESULTS_DIR / "meta.json", "w") as f:
        json.dump(dict(best_rl=best_rl, n_distinct_windows=n_distinct_windows,
                        data_span=[str(df.Timestamp.min()), str(df.Timestamp.max())],
                        mean_demand_kw=float(cluster_loads["TotalDemand"].mean())), f, indent=2)

    raw_results = {name: {k: v.tolist() for k, v in m.items()} for name, m in results.items()}
    with open(RESULTS_DIR / "raw_episode_results.json", "w") as f:
        json.dump(raw_results, f)

    print("\nALL EVALUATION DONE")


if __name__ == "__main__":
    main()

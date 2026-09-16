"""
Campus load-shedding environment, controllers, and evaluation harness.

Ported from rl_zoo_campus_load_shedding.ipynb for the camera-ready rerun.
Changes relative to the notebook (see paper/CHANGES.md for full rationale):

  1. GreedyCtrl is replaced by an exact one-step-lookahead greedy controller
     that enumerates all 216 discrete actions and picks the reward-maximising
     one, instead of a hand-rolled formula that rounded curtailment levels to
     the nearest 0.1 increment. The rounding formula had no consistent bias
     direction (sometimes over-shed, sometimes under-shed), and because the
     environment auto-clamps any over-subscribed request to exactly the
     available supply, under-shedding was accidentally "free" while
     over-shedding directly wasted deliverable energy -- an artifact
     reviewer 3 correctly flagged as making the Random-vs-Greedy comparison
     unsound. The exact lookahead is the correct operationalisation of
     "greedy" and removes the discretisation bias entirely.
  2. BaseController.evaluate() and the env optionally record oversubscription
     diagnostics (requested vs. available power per step) so the audit
     reviewer 3 asked for can be reported quantitatively.
"""
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    DATA_PATH: str = "final.csv"
    RANDOM_SEED: int = 42

    EPISODE_LENGTH: int = 168
    N_CLUSTERS: int = 3
    N_CURTAIL_LEVELS: int = 6

    TRAIN_TIMESTEPS: int = 60_000   # equalised training budget for ALL RL variants
    EVAL_EPISODES: int = 60

    W_ENERGY: float = 1.0
    W_CRITICAL: float = 2.0
    W_CURTAIL: float = 0.5
    W_REPEAT: float = 0.2
    W_FAIRNESS: float = 0.5


CFG = Config()

CURTAIL_LEVELS = np.array([0.0, 0.10, 0.20, 0.30, 0.40, 0.50])

CLUSTER_MAP = {
    "Feeder1_High Rise Block A": 2, "Feeder1_High Rise Auxiliary": 2,
    "Feeder1_High Rise Block B": 2, "Feeder1_High Rise Block C": 2,
    "Feeder1_College of Medicine Quarters": 2, "Feeder1_Professor Quarters": 2,
    "Feeder1_Alvan Ikoku": 2, "Feeder1_Agric Ode": 1, "Feeder1_Ozolua Road": 2,
    "Feeder2_Computer Center": 0, "Feeder2_Health Centre": 0,
    "Feeder2_Moremi Hall": 2, "Feeder2_Kosoko Drive": 2, "Feeder2_Tinubu Close": 2,
    "Feeder2_CITS Market": 2, "Feeder3_Engineering": 0, "Feeder3_HV Laboratory": 0,
    "Feeder3_Auditorium Chiller": 1, "Feeder3_Arts Block South": 1,
    "Feeder4_Main Library": 0, "Feeder4_Conference Center": 1,
    "Feeder4_Business Admin": 1, "Feeder4_Senate House": 0,
    "Feeder5_Faculty of Sciences": 1, "Feeder5_Chemical Engineering Building": 1,
    "Feeder5_Jaja Hall of Residence": 2, "Feeder5_Mariere Hall of Residence": 2,
    "Feeder5_Arts Theatre": 1, "Feeder5_Main Auditorium": 1,
    "Feeder5_Jaja Shopping Complex": 2, "Feeder5_Mariere Shopping Complex": 2,
    "Feeder6_Hydraulic Research Center": 1, "Feeder6_Guest House": 2,
    "Feeder6_Tayo Aderinokun Hall": 2, "Feeder6_Advanced Legal Studies": 1,
    "Feeder6_Faculty Law": 1, "Feeder6_Faculty of Arts Blocks": 1,
    "Feeder7_Access Bank": 2, "Feeder7_Works & Physical Planning Building": 0,
    "Feeder7_Faculty of Social Sciences": 1,
    "Feeder7_Distance Learning Institute": 1, "Feeder7_Honours Hostel": 2,
    "Feeder7_International School University of Lagos": 1,
    "Feeder7_Unilag Consults": 1, "Feeder7_Ransome Kuti Quarters": 2,
    "Feeder7_Henry Carr Post Graduate Hall of Residence": 2,
    "Feeder7_Biobaku Hall of Residence": 2, "Feeder8_New Hall Hostels": 2,
    "Feeder8_New Hall Complex": 2, "Feeder8_Main Mosque": 2,
    "Feeder8_Chapel-1": 2, "Feeder8_Chapel-2": 2, "Feeder8_Sports Centre": 2,
    "Feeder8_Faculty of Environmental Sciences": 1, "Feeder8_Multipurpose Hall": 1,
}


def priority_for(b: str):
    bl = b.lower()
    if "health centre" in bl:               return 10, True,  0.00
    if "hv laboratory" in bl:               return  9, True,  0.10
    if "computer center" in bl:             return  9, True,  0.10
    if "engineering" in bl and "chemical" not in bl: return 9, True, 0.10
    if "library" in bl:                     return  8, False, 0.20
    if "senate" in bl:                      return  8, False, 0.20
    if "works & physical" in bl:            return  8, True,  0.20
    if any(k in bl for k in ["faculty", "auditorium", "auditorium chiller"]): return 7, False, 0.30
    if any(k in bl for k in ["arts theatre", "conference", "business admin"]): return 7, False, 0.30
    if "distance learning" in bl or "advanced legal" in bl: return 7, False, 0.30
    if "international school" in bl or "unilag consults" in bl: return 7, False, 0.30
    if "hydraulic research" in bl:          return 7, False, 0.30
    if "agric ode" in bl:                   return 6, False, 0.30
    if any(k in bl for k in ["hall", "hostel", "high rise"]): return 5, False, 0.40
    if any(k in bl for k in ["quarters", "guest house", "kosoko", "tinubu", "alvan", "ozolua"]): return 5, False, 0.40
    if any(k in bl for k in ["shopping", "cits market", "access bank"]): return 3, False, 0.50
    if "mosque" in bl or "chapel" in bl:    return 2, False, 0.50
    if "sports centre" in bl or "multipurpose" in bl: return 3, False, 0.50
    return 5, False, 0.40


def load_data(cfg=CFG):
    df = pd.read_csv(cfg.DATA_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    building_cols = [c for c in df.columns if c != "Timestamp"]
    total_demand = df[building_cols].sum(axis=1)

    cluster_loads = pd.DataFrame({"Timestamp": df["Timestamp"]})
    for k in range(cfg.N_CLUSTERS):
        cols = [b for b in building_cols if CLUSTER_MAP[b] == k]
        cluster_loads[f"Cluster{k}"] = df[cols].sum(axis=1)
    cluster_loads["TotalDemand"] = total_demand

    rows = []
    for b in building_cols:
        pr, crit, mc = priority_for(b)
        rows.append({"Building": b, "Cluster": CLUSTER_MAP[b],
                      "Priority": pr, "Critical": crit, "MaxCurtailment": mc})
    priority_df = pd.DataFrame(rows).sort_values("Priority", ascending=False).reset_index(drop=True)
    return df, cluster_loads, priority_df, building_cols


# --------------------------------------------------------------------------- #
# Power-deficit generator
# --------------------------------------------------------------------------- #
class PowerScenario:
    REG = {"Normal": (0.95, 1.05), "Minor": (0.80, 0.95),
           "Moderate": (0.60, 0.80), "Major": (0.40, 0.60),
           "Emergency": (0.20, 0.40)}

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(CFG.RANDOM_SEED)

    def sample(self, demand, scen="Random"):
        if scen == "Random":
            scen = self.rng.choice(list(self.REG))
        lo, hi = self.REG[scen]
        r = self.rng.uniform(lo, hi)
        return demand * r, max(0.0, 1.0 - r)


def jain(x):
    x = np.asarray(x, dtype=float)
    return (x.sum() ** 2) / (len(x) * np.square(x).sum() + 1e-9)


def cyclic_hour(h):
    th = 2 * np.pi * h / 24
    return np.sin(th), np.cos(th)


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
class CampusEnvBase(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cluster_data, priority_df, cfg=CFG,
                 scenario="Random", episode_length=None,
                 reward_weights=None, seed=None, diagnostics=False):
        super().__init__()
        self.cluster_data = cluster_data.reset_index(drop=True)
        self.priority_df = priority_df
        self.cfg = cfg
        self.scenario = scenario
        self.episode_length = episode_length or cfg.EPISODE_LENGTH
        self.diagnostics = diagnostics

        d = dict(energy=cfg.W_ENERGY, critical=cfg.W_CRITICAL,
                 curtail=cfg.W_CURTAIL, repeat=cfg.W_REPEAT,
                 fairness=cfg.W_FAIRNESS)
        if reward_weights:
            d.update(reward_weights)
        self.w = d

        self._crit_share = np.zeros(cfg.N_CLUSTERS, dtype=np.float32)
        self._max_cap = np.zeros(cfg.N_CLUSTERS, dtype=np.float32)
        for k in range(cfg.N_CLUSTERS):
            sub = priority_df[priority_df["Cluster"] == k]
            crit_mass = sub.loc[sub["Critical"], "Priority"].sum()
            total = sub["Priority"].sum() + 1e-9
            self._crit_share[k] = crit_mass / total
            self._max_cap[k] = sub["MaxCurtailment"].max() if len(sub) else 0.5

        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(10,), dtype=np.float32)

        self._demand_max = float(self.cluster_data["TotalDemand"].max())
        self._cluster_max = {k: float(self.cluster_data[f"Cluster{k}"].max())
                              for k in range(cfg.N_CLUSTERS)}

        self._scen_gen = PowerScenario(np.random.default_rng(seed or cfg.RANDOM_SEED))
        self._rng = np.random.default_rng(seed or cfg.RANDOM_SEED)
        self.diag_log = []  # oversubscription diagnostics, if enabled

    def _hour_of_day(self):
        return int(self.cluster_data["Timestamp"].iloc[self._cursor].hour)

    def _make_obs(self, prev_action_mean=0.0, prev_deficit=0.0):
        row = self.cluster_data.iloc[self._cursor]
        D = np.array([row[f"Cluster{k}"] for k in range(self.cfg.N_CLUSTERS)], dtype=np.float32)
        D_tot = float(row["TotalDemand"])
        avail, deficit = self._scen_gen.sample(D_tot, self.scenario)
        self._last_avail, self._last_deficit = avail, deficit
        sh, ch = cyclic_hour(self._hour_of_day())
        obs = np.array([
            D[0] / (self._cluster_max[0] + 1e-9),
            D[1] / (self._cluster_max[1] + 1e-9),
            D[2] / (self._cluster_max[2] + 1e-9),
            D_tot / (self._demand_max + 1e-9),
            avail / (self._demand_max + 1e-9),
            deficit,
            (sh + 1) / 2,
            (ch + 1) / 2,
            prev_action_mean,
            prev_deficit,
        ], dtype=np.float32)
        return obs, D, D_tot, avail, deficit

    def _apply_curtailment(self, curtailment, D, D_tot, avail):
        curtailment = np.minimum(curtailment, self._max_cap)
        target_served = (1.0 - curtailment) * D
        requested_total = float(target_served.sum())
        oversubscribed = requested_total > avail and requested_total > 1e-6
        if oversubscribed:
            scale = avail / requested_total
            served = target_served * scale
        else:
            served = target_served
        curtailed = D - served
        if self.diagnostics:
            self.diag_log.append(dict(
                requested_total=requested_total, avail=float(avail),
                oversubscribed=bool(oversubscribed),
                slack=float(avail - requested_total),
            ))
        return served, curtailed, curtailment

    def _step_reward(self, served, curtailed, D, D_tot, action_mean, prev_action_mean):
        served_ratio = served.sum() / (D_tot + 1e-9)
        critical_pen = (self._crit_share * curtailed).sum() / (D_tot + 1e-9)
        curtail_pen = curtailed.sum() / (D_tot + 1e-9)
        repeat_pen = 1.0 if abs(action_mean - prev_action_mean) < 1e-6 else 0.0
        served_ratios = served / np.maximum(D, 1e-6)
        fairness = jain(served_ratios)
        r = (self.w["energy"] * served_ratio
             - self.w["critical"] * critical_pen
             - self.w["curtail"] * curtail_pen
             - self.w["repeat"] * repeat_pen
             + self.w["fairness"] * fairness)
        return float(r), dict(energy=self.w["energy"] * served_ratio,
                               critical=-self.w["critical"] * critical_pen,
                               curtail=-self.w["curtail"] * curtail_pen,
                               repeat=-self.w["repeat"] * repeat_pen,
                               fairness=self.w["fairness"] * fairness)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._scen_gen = PowerScenario(np.random.default_rng(seed))
        max_start = len(self.cluster_data) - self.episode_length - 1
        self._cursor = int(self._rng.integers(0, max_start))
        self._t = 0
        self._prev_action_mean = 0.0
        self._prev_deficit = 0.0
        self._history = []
        obs, *_ = self._make_obs()
        return obs, {}


class CampusEnvDiscrete(CampusEnvBase):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.action_space = spaces.Discrete(self.cfg.N_CURTAIL_LEVELS ** self.cfg.N_CLUSTERS)

    def decode_action(self, a):
        L = self.cfg.N_CURTAIL_LEVELS
        idx = [(a // (L ** k)) % L for k in range(self.cfg.N_CLUSTERS)]
        return CURTAIL_LEVELS[idx]

    def step(self, action):
        action = int(action)
        curt = self.decode_action(action)
        obs, D, D_tot, avail, deficit = self._make_obs(self._prev_action_mean, self._prev_deficit)
        served, curtailed, curt_capped = self._apply_curtailment(curt, D, D_tot, avail)
        action_mean = float(np.mean(curt_capped))
        r, br = self._step_reward(served, curtailed, D, D_tot, action_mean, self._prev_action_mean)
        self._history.append(dict(demand=D, served=served, curtailed=curtailed,
                                   available=avail, deficit=deficit, action=action, breakdown=br))
        self._prev_action_mean, self._prev_deficit = action_mean, deficit
        self._cursor += 1
        self._t += 1
        terminated = self._t >= self.episode_length
        info = dict(served=float(served.sum()), curtailed=float(curtailed.sum()),
                    breakdown=br, available=avail, deficit=deficit)
        return obs, r, terminated, False, info


class CampusEnvContinuous(CampusEnvBase):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.action_space = spaces.Box(low=0.0, high=0.5, shape=(self.cfg.N_CLUSTERS,), dtype=np.float32)

    def step(self, action):
        curt = np.clip(action.astype(np.float32), 0.0, 0.5)
        obs, D, D_tot, avail, deficit = self._make_obs(self._prev_action_mean, self._prev_deficit)
        served, curtailed, curt_capped = self._apply_curtailment(curt, D, D_tot, avail)
        action_mean = float(curt_capped.mean())
        r, br = self._step_reward(served, curtailed, D, D_tot, action_mean, self._prev_action_mean)
        self._history.append(dict(demand=D, served=served, curtailed=curtailed,
                                   available=avail, deficit=deficit, action=curt_capped, breakdown=br))
        self._prev_action_mean, self._prev_deficit = action_mean, deficit
        self._cursor += 1
        self._t += 1
        terminated = self._t >= self.episode_length
        info = dict(served=float(served.sum()), curtailed=float(curtailed.sum()),
                    breakdown=br, available=avail, deficit=deficit)
        return obs, r, terminated, False, info


class CampusEnvMasked(CampusEnvDiscrete):
    def action_masks(self):
        L = self.cfg.N_CURTAIL_LEVELS
        n = L ** self.cfg.N_CLUSTERS
        mask = np.ones(n, dtype=bool)
        for a in range(n):
            idx = [(a // (L ** k)) % L for k in range(self.cfg.N_CLUSTERS)]
            curt = CURTAIL_LEVELS[idx]
            if np.any(curt > self._max_cap + 1e-6):
                mask[a] = False
        return mask


# --------------------------------------------------------------------------- #
# Controllers
# --------------------------------------------------------------------------- #
class BaseController:
    name = "Base"

    def __init__(self, env):
        self.env = env

    def select_action(self, obs, info=None):
        raise NotImplementedError

    def evaluate(self, n_episodes=100, seed=0):
        rewards, served, curtail, critical, fair, runtime = [], [], [], [], [], []
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=seed + ep)
            r_tot = s_tot = c_tot = cr_tot = 0.0
            t0 = time.perf_counter()
            done = False
            while not done:
                a = self.select_action(obs)
                obs, r, done, _, info = self.env.step(a)
                r_tot += r
                s_tot += info["served"]
                c_tot += info["curtailed"]
                cr_tot += info["breakdown"]["critical"]
            runtime.append(time.perf_counter() - t0)
            served_by = np.sum([h["served"] for h in self.env._history], axis=0)
            demand_by = np.sum([h["demand"] for h in self.env._history], axis=0)
            ratios = served_by / np.maximum(demand_by, 1e-9)
            rewards.append(r_tot); served.append(s_tot); curtail.append(c_tot)
            critical.append(cr_tot); fair.append(jain(ratios))
        return dict(reward=np.array(rewards), served=np.array(served),
                    curtail=np.array(curtail), critical=np.array(critical),
                    fairness=np.array(fair), runtime=np.array(runtime))


class RandomCtrl(BaseController):
    name = "Random"
    def select_action(self, obs, info=None):
        return self.env.action_space.sample()


class RoundRobinCtrl(BaseController):
    name = "RoundRobin"
    def __init__(self, env):
        super().__init__(env); self._c = 0
    def select_action(self, obs, info=None):
        a = self._c % self.env.action_space.n
        self._c += 1
        return a


class StaticPriorityCtrl(BaseController):
    name = "StaticPriority"
    def select_action(self, obs, info=None):
        idx = [0, 2, 4]
        L = self.env.cfg.N_CURTAIL_LEVELS
        return int(idx[0] + idx[1] * L + idx[2] * L ** 2)


class LowestPriorityFirstCtrl(BaseController):
    name = "LowestPriorityFirst"
    def select_action(self, obs, info=None):
        deficit = obs[5]
        L = self.env.cfg.N_CURTAIL_LEVELS
        if deficit < 0.05:
            return int(0)
        lvl2 = int(min(L - 1, round(deficit * (L - 1))))
        lvl1 = int(min(L - 1, round(deficit * (L - 1) * 0.6)))
        return int(0 + lvl1 * L + lvl2 * L ** 2)


class GreedyFormulaCtrl(BaseController):
    """Original notebook Greedy (kept for the paper's before/after audit)."""
    name = "GreedyFormula"

    def select_action(self, obs, info=None):
        e = self.env
        cursor = e._cursor
        row = e.cluster_data.iloc[cursor]
        D = np.array([row[f"Cluster{k}"] for k in range(e.cfg.N_CLUSTERS)])
        D_tot = D.sum()
        avail = e._last_avail if hasattr(e, "_last_avail") else D_tot
        if avail >= D_tot:
            return 0
        deficit_kw = D_tot - avail
        L = e.cfg.N_CURTAIL_LEVELS
        idx = [0, 0, 0]
        remaining = deficit_kw
        for k in [2, 1, 0]:
            if remaining <= 0:
                break
            need_frac = min(0.5, remaining / max(D[k], 1e-6))
            lvl = int(round(need_frac * (L - 1) / 0.5))
            lvl = min(lvl, L - 1)
            idx[k] = lvl
            remaining -= CURTAIL_LEVELS[lvl] * D[k]
        return int(idx[0] + idx[1] * L + idx[2] * L ** 2)


class GreedyCtrl(BaseController):
    """
    Exact one-step-lookahead greedy: enumerate all 216 discrete actions,
    apply the environment's own curtailment/reward function to each
    (without mutating state), and pick the reward-maximising action.

    This is the operator-grade reference used in the camera-ready. It
    replaces GreedyFormulaCtrl, whose nearest-0.1 rounding had no
    consistent bias direction and, combined with the environment's
    auto-clamp on over-subscribed requests, made roughly half of its
    per-step decisions strictly dominated by simply asking for less
    curtailment (see paper/CHANGES.md).
    """
    name = "Greedy"

    def __init__(self, env):
        super().__init__(env)
        L = env.cfg.N_CURTAIL_LEVELS
        K = env.cfg.N_CLUSTERS
        self._all_actions = np.arange(L ** K)
        self._all_curt = np.stack([env.decode_action(a) for a in self._all_actions])  # (216, K)
        self._capped = np.minimum(self._all_curt, env._max_cap[None, :])              # (216, K)

    def select_action(self, obs, info=None):
        """Vectorised one-step lookahead over all 216 actions at once
        (equivalent to calling env._apply_curtailment/_step_reward in a
        Python loop, but ~100x faster since it never leaves numpy)."""
        e = self.env
        row = e.cluster_data.iloc[e._cursor]
        D = np.array([row[f"Cluster{k}"] for k in range(e.cfg.N_CLUSTERS)])
        D_tot = D.sum()
        avail = e._last_avail if hasattr(e, "_last_avail") else D_tot

        capped = self._capped                                    # (216, K)
        target_served = (1.0 - capped) * D[None, :]               # (216, K)
        requested_total = target_served.sum(axis=1)                # (216,)
        oversub = requested_total > avail
        scale = np.where(oversub, avail / np.maximum(requested_total, 1e-6), 1.0)
        served = target_served * scale[:, None]                    # (216, K)
        curtailed = D[None, :] - served                             # (216, K)
        action_mean = capped.mean(axis=1)                           # (216,)

        served_ratio = served.sum(axis=1) / (D_tot + 1e-9)
        critical_pen = (e._crit_share[None, :] * curtailed).sum(axis=1) / (D_tot + 1e-9)
        curtail_pen = curtailed.sum(axis=1) / (D_tot + 1e-9)
        repeat_pen = (np.abs(action_mean - e._prev_action_mean) < 1e-6).astype(float)
        served_ratios = served / np.maximum(D[None, :], 1e-6)
        s1 = served_ratios.sum(axis=1)
        s2 = np.square(served_ratios).sum(axis=1)
        K = served_ratios.shape[1]
        fairness = (s1 ** 2) / (K * s2 + 1e-9)

        r = (e.w["energy"] * served_ratio
             - e.w["critical"] * critical_pen
             - e.w["curtail"] * curtail_pen
             - e.w["repeat"] * repeat_pen
             + e.w["fairness"] * fairness)
        best_a = int(np.argmax(r))

        if e.diagnostics:
            e.diag_log.append(dict(
                requested_total=float(requested_total[best_a]), avail=float(avail),
                oversubscribed=bool(oversub[best_a]),
                slack=float(avail - requested_total[best_a]),
            ))
        return best_a

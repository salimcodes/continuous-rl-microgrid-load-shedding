"""Algorithm factories + training driver, equalised to CFG.TRAIN_TIMESTEPS
for every variant. Ported 1:1 from the notebook's hyperparameters (only the
step-count budget and the Greedy implementation used for BC/Hybrid changed).
Models are checkpointed to disk so a crashed run can resume without redoing
finished (algo, seed) pairs.
"""
import warnings
warnings.filterwarnings("ignore")
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from sb3_contrib import QRDQN, TRPO, RecurrentPPO, MaskablePPO
    SB3_CONTRIB = True
except Exception as e:
    print(f"sb3-contrib unavailable ({e})")
    SB3_CONTRIB = False

from campus_env import (CFG, load_data, CampusEnvDiscrete, CampusEnvContinuous,
                         CampusEnvMasked, GreedyFormulaCtrl)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

df, cluster_loads, priority_df, building_cols = load_data(CFG)


def set_seeds(seed):
    import random, os
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_env_fn(env_cls, seed, scenario="Random"):
    def _thunk():
        return Monitor(env_cls(cluster_loads, priority_df, scenario=scenario, seed=seed))
    return _thunk


class TrainingLogger(BaseCallback):
    def __init__(self):
        super().__init__(0)
        self.episode_rewards = []
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True


# --------------------------------------------------------------------------- #
# Factories (hyperparameters unchanged from the notebook)
# --------------------------------------------------------------------------- #
def ppo_factory(venv, seed):
    return PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=2048, batch_size=64,
               gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, verbose=0, seed=seed)

def ppo_large_factory(venv, seed):
    return PPO("MlpPolicy", venv, policy_kwargs=dict(net_arch=[256, 256]),
               learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99,
               gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, verbose=0, seed=seed)

def ppo_tuned_factory(venv, seed):
    return PPO("MlpPolicy", venv, learning_rate=1e-4, n_steps=4096, batch_size=128,
               n_epochs=20, gamma=0.99, gae_lambda=0.95, ent_coef=0.005, clip_range=0.2,
               verbose=0, seed=seed)

def a2c_factory(venv, seed):
    return A2C("MlpPolicy", venv, learning_rate=7e-4, gamma=0.99, ent_coef=0.01, verbose=0, seed=seed)

def dqn_factory(venv, seed):
    return DQN("MlpPolicy", venv, learning_rate=1e-4, buffer_size=100_000, batch_size=64,
               gamma=0.99, target_update_interval=1000, exploration_fraction=0.2,
               learning_starts=1000, verbose=0, seed=seed)

def sac_factory(venv, seed):
    return SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=100_000, batch_size=128,
               gamma=0.99, tau=0.005, learning_starts=1000, verbose=0, seed=seed)

def td3_factory(venv, seed):
    return TD3("MlpPolicy", venv, learning_rate=3e-4, buffer_size=100_000, batch_size=128,
               gamma=0.99, tau=0.005, policy_delay=2, learning_starts=1000, verbose=0, seed=seed)

if SB3_CONTRIB:
    def qrdqn_factory(venv, seed):
        return QRDQN("MlpPolicy", venv, learning_rate=5e-5, buffer_size=100_000, batch_size=64,
                     gamma=0.99, target_update_interval=1000, exploration_fraction=0.2,
                     learning_starts=1000, verbose=0, seed=seed)

    def recurrent_factory(venv, seed):
        return RecurrentPPO("MlpLstmPolicy", venv, learning_rate=3e-4, n_steps=2048, batch_size=64,
                             gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, verbose=0, seed=seed)

    def masked_factory(venv, seed):
        return MaskablePPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=2048, batch_size=64,
                            gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, verbose=0, seed=seed)

    def trpo_factory(venv, seed):
        return TRPO("MlpPolicy", venv, learning_rate=1e-3, n_steps=2048, batch_size=64,
                    gamma=0.99, gae_lambda=0.95, verbose=0, seed=seed)


# --------------------------------------------------------------------------- #
# Generic trainer with checkpointing
# --------------------------------------------------------------------------- #
ALGO_TABLE = {
    # name -> (factory, env_cls, ModelClass)
    "PPO": (ppo_factory, CampusEnvDiscrete, PPO),
    "PPO-Large": (ppo_large_factory, CampusEnvDiscrete, PPO),
    "PPO-Tuned": (ppo_tuned_factory, CampusEnvDiscrete, PPO),
    "A2C": (a2c_factory, CampusEnvDiscrete, A2C),
    "DQN": (dqn_factory, CampusEnvDiscrete, DQN),
    "SAC": (sac_factory, CampusEnvContinuous, SAC),
    "TD3": (td3_factory, CampusEnvContinuous, TD3),
}
if SB3_CONTRIB:
    ALGO_TABLE.update({
        "QR-DQN": (qrdqn_factory, CampusEnvDiscrete, QRDQN),
        "RecurrentPPO": (recurrent_factory, CampusEnvDiscrete, RecurrentPPO),
        "MaskablePPO": (masked_factory, CampusEnvMasked, MaskablePPO),
        "TRPO": (trpo_factory, CampusEnvDiscrete, TRPO),
    })

HEADLINE_MULTISEED = ["SAC", "TD3", "PPO-Tuned", "MaskablePPO", "PPO-Large"]
SEEDS_MULTI = [42, 43, 44, 45, 46]
SEEDS_SINGLE = [42]


def model_path(name, seed):
    return MODEL_DIR / f"{name}_seed{seed}.zip"


def train_one(name, seed, total_timesteps=None):
    total_timesteps = total_timesteps or CFG.TRAIN_TIMESTEPS
    path = model_path(name, seed)
    if path.exists():
        print(f"[skip] {name} seed={seed} already trained -> {path}")
        return
    factory, env_cls, ModelClass = ALGO_TABLE[name]
    set_seeds(seed)
    venv = DummyVecEnv([make_env_fn(env_cls, seed=seed)])
    logger = TrainingLogger()
    model = factory(venv, seed)
    print(f"[train] {name} seed={seed} for {total_timesteps:,} steps ...", flush=True)
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=logger, progress_bar=False)
    dt = time.perf_counter() - t0
    last20 = np.mean(logger.episode_rewards[-20:]) if logger.episode_rewards else float("nan")
    print(f"[done ] {name} seed={seed} in {dt:.1f}s  last20ep={last20:.2f}", flush=True)
    model.save(str(path))
    with open(LOG_DIR / f"{name}_seed{seed}.json", "w") as f:
        json.dump(dict(name=name, seed=seed, seconds=dt, total_timesteps=total_timesteps,
                        last20ep_reward=float(last20),
                        episode_rewards=logger.episode_rewards), f)


def train_curriculum(seed=42):
    name = "PPO-Curriculum"
    path = model_path(name, seed)
    if path.exists():
        print(f"[skip] {name} seed={seed} already trained -> {path}")
        return
    set_seeds(seed)
    logger = TrainingLogger()
    venv = DummyVecEnv([make_env_fn(CampusEnvDiscrete, seed=seed, scenario="Normal")])
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=2048, batch_size=64,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, verbose=0, seed=seed)
    third = CFG.TRAIN_TIMESTEPS // 3
    stages = [("Normal", third), ("Moderate", third), ("Random", CFG.TRAIN_TIMESTEPS - 2 * third)]
    t0 = time.perf_counter()
    for scen, budget in stages:
        print(f"[train] {name} stage={scen} budget={budget}", flush=True)
        venv = DummyVecEnv([make_env_fn(CampusEnvDiscrete, seed=seed, scenario=scen)])
        model.set_env(venv)
        model.learn(total_timesteps=budget, callback=logger, progress_bar=False)
    dt = time.perf_counter() - t0
    print(f"[done ] {name} seed={seed} in {dt:.1f}s", flush=True)
    model.save(str(path))
    with open(LOG_DIR / f"{name}_seed{seed}.json", "w") as f:
        json.dump(dict(name=name, seed=seed, seconds=dt, total_timesteps=CFG.TRAIN_TIMESTEPS,
                        episode_rewards=logger.episode_rewards), f)


def collect_greedy_dataset(n_episodes=25, seed=42):
    env = CampusEnvDiscrete(cluster_loads, priority_df, seed=seed)
    greedy = GreedyFormulaCtrl(env)
    X, y = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a = greedy.select_action(obs)
            X.append(obs.copy()); y.append(int(a))
            obs, _, done, _, _ = env.step(a)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_bc(seed=42):
    name = "PPO-BC"
    path = model_path(name, seed)
    if path.exists():
        print(f"[skip] {name} seed={seed} already trained -> {path}")
        return
    print("[train] PPO-BC: collecting Greedy(formula) expert trajectories ...", flush=True)
    Xe, ye = collect_greedy_dataset(n_episodes=25, seed=seed)
    set_seeds(seed)
    venv = DummyVecEnv([make_env_fn(CampusEnvDiscrete, seed=seed)])
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=2048, batch_size=64,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, verbose=0, seed=seed)
    device = model.policy.device
    Xt = torch.tensor(Xe, dtype=torch.float32, device=device)
    yt = torch.tensor(ye, dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print(f"[train] PPO-BC: cloning on {len(Xt)} transitions for 200 epochs ...", flush=True)
    for ep in range(200):
        idx = torch.randperm(len(Xt), device=device)[:1024]
        bx, by = Xt[idx], yt[idx]
        dist = model.policy.get_distribution(bx)
        loss = loss_fn(dist.distribution.logits, by)
        opt.zero_grad(); loss.backward(); opt.step()
    print("[train] PPO-BC: fine-tuning with PPO ...", flush=True)
    logger = TrainingLogger()
    t0 = time.perf_counter()
    model.learn(total_timesteps=CFG.TRAIN_TIMESTEPS, callback=logger, progress_bar=False)
    dt = time.perf_counter() - t0
    print(f"[done ] {name} seed={seed} in {dt:.1f}s", flush=True)
    model.save(str(path))
    with open(LOG_DIR / f"{name}_seed{seed}.json", "w") as f:
        json.dump(dict(name=name, seed=seed, seconds=dt, total_timesteps=CFG.TRAIN_TIMESTEPS,
                        episode_rewards=logger.episode_rewards), f)


def build_schedule():
    """Return list of (name, seed) pairs to train, in a sensible order:
    cheap on-policy algos first, expensive off-policy algos last, so partial
    progress is maximally useful if interrupted."""
    schedule = []
    cheap_single = ["PPO", "A2C"]
    if SB3_CONTRIB:
        cheap_single += ["TRPO", "RecurrentPPO"]
    for n in cheap_single:
        schedule.append((n, 42))
    # PPO-Curriculum and PPO-BC use bespoke training loops, handled separately
    for n in HEADLINE_MULTISEED:
        if n in ("PPO-Large", "MaskablePPO", "PPO-Tuned"):
            for s in SEEDS_MULTI:
                schedule.append((n, s))
    schedule.append(("DQN", 42))
    if SB3_CONTRIB:
        schedule.append(("QR-DQN", 42))
    for n in ["SAC", "TD3"]:
        for s in SEEDS_MULTI:
            schedule.append((n, s))
    return schedule


if __name__ == "__main__":
    import sys
    train_curriculum(42)
    train_bc(42)
    for name, seed in build_schedule():
        train_one(name, seed)
    print("ALL TRAINING DONE")

import time
import warnings
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from campus_env import CFG, load_data, CampusEnvDiscrete, CampusEnvContinuous

df, cluster_loads, priority_df, building_cols = load_data(CFG)

def make_env(cls):
    def _thunk():
        return Monitor(cls(cluster_loads, priority_df, seed=42))
    return _thunk

N = 4000

venv = DummyVecEnv([make_env(CampusEnvDiscrete)])
m = PPO("MlpPolicy", venv, verbose=0, seed=42)
t0 = time.perf_counter()
m.learn(total_timesteps=N, progress_bar=False)
dt = time.perf_counter() - t0
print(f"PPO discrete: {N} steps in {dt:.1f}s -> {dt/N*60000:.1f}s for 60k steps")

venv2 = DummyVecEnv([make_env(CampusEnvContinuous)])
m2 = SAC("MlpPolicy", venv2, verbose=0, seed=42, learning_starts=200)
t0 = time.perf_counter()
m2.learn(total_timesteps=N, progress_bar=False)
dt2 = time.perf_counter() - t0
print(f"SAC continuous: {N} steps in {dt2:.1f}s -> {dt2/N*60000:.1f}s for 60k steps")

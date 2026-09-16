"""End-to-end pipeline smoke test with tiny budgets, to catch bugs before
committing to the full multi-hour run. Uses a separate models/results dir.
"""
import shutil
from pathlib import Path

from campus_env import CFG
CFG.TRAIN_TIMESTEPS = 600
CFG.EVAL_EPISODES = 3

import train
train.MODEL_DIR = Path("models_smoke")
train.MODEL_DIR.mkdir(exist_ok=True)
train.LOG_DIR = Path("logs_smoke")
train.LOG_DIR.mkdir(exist_ok=True)
train.SEEDS_MULTI = [42, 43]  # just 2 seeds for the smoke test

train.train_curriculum(42)
train.train_bc(42)
for name in list(train.ALGO_TABLE.keys()):
    if name in train.HEADLINE_MULTISEED:
        for s in train.SEEDS_MULTI:
            train.train_one(name, s, total_timesteps=600)
    else:
        train.train_one(name, 42, total_timesteps=600)

print("\n=== TRAINING SMOKE TEST OK, now evaluating ===\n")

import evaluate
evaluate.MODEL_DIR = train.MODEL_DIR
evaluate.RESULTS_DIR = Path("results_smoke")
evaluate.RESULTS_DIR.mkdir(exist_ok=True)

def model_path_smoke(name, seed):
    return train.MODEL_DIR / f"{name}_seed{seed}.zip"
evaluate.model_path = model_path_smoke
evaluate.HEADLINE_MULTISEED = train.HEADLINE_MULTISEED
evaluate.SEEDS_MULTI = train.SEEDS_MULTI

evaluate.main()
print("\n=== SMOKE TEST FULLY PASSED ===")

"""Full camera-ready rerun: train all 14 RL variants at the equalised 60k-step
budget (5 seeds for the headline controllers), then evaluate everything and
write results/. Expected wall-clock: roughly 10-13 hours on this machine
(CPU-only, 4 cores) -- dominated by the off-policy algorithms (SAC, TD3,
DQN, QR-DQN), which do a gradient update every environment step.
"""
import time

t0 = time.perf_counter()
print("=== STAGE 1: TRAINING ===", flush=True)
import train
train.train_curriculum(42)
train.train_bc(42)
schedule = train.build_schedule()
print(f"Training schedule ({len(schedule)} runs): {schedule}", flush=True)
for i, (name, seed) in enumerate(schedule):
    print(f"--- schedule {i+1}/{len(schedule)} ---", flush=True)
    train.train_one(name, seed)

dt_train = time.perf_counter() - t0
print(f"=== TRAINING DONE in {dt_train/3600:.2f}h ===", flush=True)

print("=== STAGE 2: EVALUATION ===", flush=True)
t1 = time.perf_counter()
import evaluate
evaluate.main()
dt_eval = time.perf_counter() - t1
print(f"=== EVALUATION DONE in {dt_eval/60:.1f}min ===", flush=True)

print(f"=== FULL PIPELINE DONE in {(time.perf_counter()-t0)/3600:.2f}h total ===", flush=True)

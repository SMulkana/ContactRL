"""

Date: 2025-10-01

Evaluate a trained SAC policy on a custom Gymnasium ContactLag environment with
Lagrangian penalty on costs. The script runs deterministic rollouts, logs per-episode
Rewards/Costs plus env metrics (Force, Steps, Success, jerk, etc.), saves results to
Excel, and prints a concise summary (mean reward/cost, success rate, force violations).
Configure MODEL_PATH, EXCEL_OUT, ENV_NAME, N_EVAL_EPISODES, and SEED as needed.

"""
import os
import gymnasium as gym
import env
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# === CONFIGURATION ===
MODEL_PATH = "/path/to/folder/file_name"
EXCEL_OUT = "/path/to.folder/file_name.xlsx"
ENV_NAME = "EnvID"
N_EVAL_EPISODES = 100
SEED = 4

# === Lagrangian penalty wrapper ===
class LagrangePenalty(gym.Wrapper):
    def __init__(self, env, lam: float = 0.0):
        super().__init__(env)
        self.lam = lam

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = info.get("cost", 0.0)
        reward -= self.lam * cost
        return obs, reward, terminated, truncated, info

# === Environment setup ===
base_env = gym.make(ENV_NAME)
base_env.reset(seed=SEED)
penalty_env = LagrangePenalty(base_env, lam=0.0)

# === Load SAC model ===
model = SAC.load(MODEL_PATH)

# === Define metrics to extract ===
metric_names = [
    "Average_episode_reward",        # smoothed reward
    "interaction_forceB_counter",    # force measurement
    "TT",                            # total steps
    "reach_success",                 # success (0 or 1)
    "jerk_mag",                      # optional
    "rms_jerk",                      # motion smoothness
    "acceleration_ee"                # EE acceleration
]

# === Evaluation loop ===
episode_records = []
returns, costs = [], []

for ep in range(N_EVAL_EPISODES):
    obs, _ = penalty_env.reset(seed=SEED + ep)
    done = False
    ep_return, ep_cost = 0.0, 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = penalty_env.step(action)
        ep_return += reward
        ep_cost += info.get("cost", 0.0)
        if terminated or truncated:
            break

    # Record standard metrics
    record = {
        "Rewards": ep_return,
        "Costs": ep_cost
    }

    # Extract custom attributes if available
    for key in metric_names:
        val = getattr(penalty_env, key, None)
        if isinstance(val, (list, np.ndarray)):
            record[key] = val[-1] if len(val) > 0 else None
        else:
            record[key] = val

    episode_records.append(record)
    returns.append(ep_return)
    costs.append(ep_cost)

# === Save DataFrame to Excel ===
df = pd.DataFrame(episode_records)
df = df.rename(columns={
    "interaction_forceB_counter": "Force",
    "TT": "Steps",
    "reach_success": "Success"
})

os.makedirs(os.path.dirname(EXCEL_OUT), exist_ok=True)
df.to_excel(EXCEL_OUT, index=False)
print(f"Evaluation data saved to: {EXCEL_OUT}")

# === Summary ===
print(f"\n Evaluation Summary over {N_EVAL_EPISODES} episodes:")
print(f"→ Mean Reward: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
print(f"→ Mean Cost:   {np.mean(costs):.2f}")

if "Success" in df.columns and df["Success"].notna().any():
    success_rate = df["Success"].dropna().mean()
    print(f"→ Success Rate: {success_rate * 100:.2f}%")

if "Force" in df.columns:
    violations = df["Force"].dropna() > 50.0
    print(f"→ Safety Violations (>50N): {violations.sum()} episodes ({violations.mean() * 100:.2f}%)")

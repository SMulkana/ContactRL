"""
Lagrangian SAC Training with Dual Ascent for ContactLag Environment (Gymnasium + SB3)

This script trains a Soft Actor Critic agent on ContactLag env while enforcing a
safety budget via a simple Lagrangian penalty that’s adapted online:

- Reproducibility: seeds NumPy, Python, and PyTorch with a single master SEED.
- Penalty wrapper: `LagrangePenalty` subtracts λ·cost from the environment reward
  (reads per-step safety `cost` from `info["cost"]`).
- Dual ascent: `DualAscentCallback` updates λ after each rollout to push the
  episode cost toward a target `cost_limit` (λ ← max(0, λ + lr·(Σcost − d))).
- Agent: Stable-Baselines3 SAC with a progress-bar callback for live feedback.
- Logging: environment is wrapped with `Monitor`; after training, metrics exposed
  by the underlying env (timesteps, rewards, forces, etc.) are gathered and saved
  to an Excel file for analysis.

Update `root_dir`, `tag`, and the `cost_limit`/`lr` in `DualAscentCallback` to fit
your experiment. Assumes the environment populates `info["cost"]` and maintains
the listed attributes (e.g., `timestep_counter`, `interaction_forceB_counter`).

Date: 3 August 2025
"""

import gymnasium as gym, env
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from tqdm import tqdm
import numpy as np, torch, random, os, time, pandas as pd

SEED = 5                                # ← master seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)                 # ensures SB3/PyTorch reproducibility

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose); self.total_timesteps = total_timesteps
    def _on_training_start(self): self.pbar = tqdm(total=self.total_timesteps)
    def _on_step(self):           self.pbar.update(1); return True
    def _on_training_end(self):   self.pbar.close()

class LagrangePenalty(gym.Wrapper):
    def __init__(self, env, lam: float = 0.0):
        super().__init__(env); self.lam = lam
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = info.get("cost", 0.0)
        reward -= self.lam * cost
        return obs, reward, terminated, truncated, info

class DualAscentCallback(BaseCallback):
    def __init__(self, penalty_wrapper, cost_limit, lr=0.01, verbose=0):
        super().__init__(verbose)
        self.w, self.d, self.lr = penalty_wrapper, cost_limit, lr
    def _on_rollout_start(self): self.cost_sum = 0.0
    def _on_step(self):
        self.cost_sum += self.locals["infos"][0].get("cost", 0.0); return True
    def _on_rollout_end(self):
        err = self.cost_sum - self.d
        self.w.lam = max(0.0, self.w.lam + self.lr * err)

root_dir  = "/path/to/folder/"
os.makedirs(root_dir, exist_ok=True)
tag       = "file_name"

base_env    = gym.make("envID")
base_env.reset(seed=SEED)               # seed the Gymnasium env
penalty_env = LagrangePenalty(base_env, lam=0.0)
env         = Monitor(penalty_env, root_dir)


model = SAC("MlpPolicy", env, seed=SEED, verbose=1)

total_steps = 1000_000
progress_cb = ProgressBarCallback(total_timesteps=total_steps)
lag_cb      = DualAscentCallback(penalty_wrapper=penalty_env,
                                 cost_limit=1.0,   # per-episode budget
                                 lr=0.01,
                                 verbose=1)
callbacks   = CallbackList([progress_cb, lag_cb])

t0 = time.time()
model.learn(total_timesteps=total_steps, log_interval=100, callback=callbacks)
print(f"Training finished in {time.time()-t0:.1f} s")

model.save(os.path.join(root_dir, tag))

unwrapped = env
while hasattr(unwrapped, "env"):
    unwrapped = unwrapped.env

cols = {
    "Timesteps": getattr(unwrapped, "timestep_counter",             []),
    "Reward":    getattr(env,       "episodic_reward",              []),
    "Force":     getattr(env,       "interaction_forceB_counter",   []),
    "AvgReward": getattr(env,       "Average_episode_reward",       []),
    "Steps":     getattr(env,       "TT",                           []),
}

valid_cols = {k: v for k, v in cols.items() if len(v) > 0}

if len(valid_cols) == 0:
    print("⚠️  No arrays found — nothing to log.")
else:
    # find the common length (min) and trim all arrays to that length
    min_len = min(len(v) for v in valid_cols.values())
    trimmed = {k: v[:min_len] for k, v in valid_cols.items()}

    log_df = pd.DataFrame(trimmed)
    excel_path = os.path.join(root_dir, tag + "-Training.xlsx")
    log_df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"Logged metrics saved to {excel_path} (shape: {log_df.shape})")

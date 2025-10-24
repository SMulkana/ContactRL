
"""

Date: 2025-10-01

SAC Training Script for ContactRL environment (Gymnasium + SB3)

This script trains a Soft Actor Critic (SAC) agent on the custom environment. 
It emphasizes reproducibility, progress
visibility, and result logging:

- Reproducibility: seeds Python, NumPy, and PyTorch RNGs.
- Environment: wraps the env with `Monitor` for episode stats and logs.
- Algorithm: Stable-Baselines3 SAC with TensorBoard logging enabled.
- UX: custom `ProgressBarCallback` using `tqdm` to show real-time training progress.
- Persistence: saves the trained policy and exports key per-step/episode metrics
  (e.g., rewards, contact force, average episode reward, timing) to Excel for analysis.

Paths in this script point to local experiment folders (update as needed).

"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import psutil
from tqdm import tqdm   
import time
import pandas as pd
import sys
import random
import torch
SEED = 0

# 2) Seed the Python & library RNGs
os.environ["PYTHONHASHSEED"] = str(SEED)    # hash‐seed for consistency
random.seed(SEED)                           # Python built-ins
np.random.seed(SEED)                        # NumPy RNG
torch.manual_seed(SEED)                     # PyTorch CPU RNG

import gymnasium as gym
import env
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=2):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:  # Note the underscore
        self.pbar =  tqdm(total=self.total_timesteps, desc='Training Progress', file=sys.stdout)

    def _on_step(self) -> bool:  # Note the underscore
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:  # Note the underscore
        self.pbar.close()


log_dir = "/path/to/folder/"
os.makedirs(log_dir, exist_ok=True)
# Specify the folder for TensorBoard logs
tensorboard_log_dir = "/path/to/folder/"
# Create and wrap the environment
env = gym.make('envID')
env = Monitor(env, log_dir)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)
obs, _ = env.reset(seed=SEED)


# Specify to use the CPU
model_SAC = SAC("MlpPolicy", env, seed=SEED, verbose=2, tensorboard_log=tensorboard_log_dir)
# Total timesteps and callback
total_timesteps = 100_000
progress_bar_callback = ProgressBarCallback(total_timesteps=total_timesteps)

# Learn
model_SAC.learn(total_timesteps=total_timesteps, log_interval=1000, callback=progress_bar_callback)
start_time = time.time()
training_duration = time.time() - start_time
print(f"Training took {training_duration:.2f} seconds.")

save_path = "/path/to/folder/"
# Create save directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)
model_SAC.save(os.path.join(save_path, "file_name"))

#Logging metrics
a = env.timestep_counter
b = env.reward_list
c = env.contact_counter
d = env.episode_end_condition
e = env.episodic_reward
f = env.interaction_forceB_counter
g = env.distance_ee_objB
h = env.ee_increment
i = env.Reward_safe
k = env.Average_episode_reward
l = env.TT
m = env.Reward_smooth
n = env.Reward_proximity

max_entries = total_timesteps

data = {'Timesteps': a, 'Rewards': b, 'Contact Type': c,
        'Episode end Condition': d,'Episodic Reward': e, 'ForceB': f,
        'Distance ee to obj B': g, 'delta': h, 'Average Episode Rewards': k, 'Total time': l}


max_length = max(len(k), len(f), len(l), len(h))
k = k + [None] * (max_length - len(k))
f = f + [None] * (max_length - len(f))
l = l + [None] * (max_length - len(l))
h = h + [None] * (max_length - len(h))

df2= pd.DataFrame({'Rewards': k,   'ForceB': f, 'Time': l, 'delta': h})
df2.to_excel('path/to/folder/file_name.xlsx', index=False)

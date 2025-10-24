"""

Date: 2025-10-01

This script evaluates a trained Soft Actor-Critic (SAC) model using Stable-Baselines3 
on a custom Gymnasium ContactRL environment. It loads the saved policy, computes the 
average reward over multiple evaluation episodes, and extracts key performance metrics 
such as interaction force, reward progression, and steps to reach the goal. Finally, 
it visualizes these metrics across episodes using matplotlib for easier analysis of 
policy performance and stability.
"""
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC # Replace PPO with your algorithm
import gymnasium as gym
#from gymnasium.wrappers import Monitor
import env
import pandas as pd
import matplotlib.pyplot as plt
import csv


# Load trained modelp
model = SAC.load("/path/to/folder/file_name.zip")

# Create the environment
env = gym.make("EnvID")
#eval_env = Monitor(env)
# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=False)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

a = env.Average_episode_reward
b = env.interaction_forceB_counter
c = env.TT
d= env.reach_success
e= env.jerk_mag
f = env.rms_jerk
g = env.acceleration_ee

Force = df['Force'][df['Force'] != 0]
Reward = df['Rewards'][df['Rewards'] != 0]
Time = df['Steps'][df['Steps'] != 0]

Reward_list = Reward.tolist() 
ForceB_list = Force.tolist() 
Time_list = Time.tolist()  

A = len(Reward_list)
A_plot=list(range(1, A+1))

B= len(ForceB_list)
B_plot=list(range(1, B+1))

C= len(Time_list)
C_plot = list(range(1, C+1))


# Create a single figure with 3 horizontal subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

# Plot 1: Force vs Episodes
axs[0].plot(B_plot, ForceB_list, linestyle='-', color='b')
axs[0].set_xlabel('Episodes', fontsize=20)
axs[0].set_ylabel('Force (N)', fontsize=20)
axs[0].tick_params(axis='both', labelsize=20)
axs[0].grid(True)

# Plot 2: Reward vs Episodes
axs[1].plot(A_plot, Reward_list, linestyle='-', color='g')
axs[1].set_xlabel('Episodes', fontsize=20)
axs[2].set_ylabel('Steps to Reach (Timesteps)', fontsize=20)
axs[1].set_ylabel('Mean Cumulative Reward', fontsize=20)
axs[1].tick_params(axis='both', labelsize=20)
axs[1].grid(True)

# Plot 3: Steps vs Episodes
axs[2].plot(C_plot, Time_list, linestyle='-', color='r')
axs[2].set_xlabel('Episodes', fontsize=20)
axs[2].tick_params(axis='both', labelsize=20)
axs[2].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

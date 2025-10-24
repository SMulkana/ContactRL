"""

Date: 2025-10-01

Evaluate a trained CPO-style policy (PyTorch) on a custom Gymnasium ContactCPO environment.

- Loads a checkpoint into a lightweight policy class matching the training architecture.
- Runs N_EVAL_EPISODES with deterministic actions for reproducible returns and costs.
- Collects per-episode metrics exposed by the env (e.g., force counters, steps, success, jerk).
- Saves all episode records to an Excel file for later analysis.
- Plots Force, Reward, and Steps trajectories across episodes.

Configure paths, env name, and device in the CONFIG block before running.

"""
import torch
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import env

# === CONFIG ===
MODEL_PATH = "/path/to/folder/file_name"
EXCEL_OUT = "/path/to/folder/file_name.xlsx"
ENV_NAME = "EnvID"
N_EVAL_EPISODES = 100
SEED = 0
DEVICE = torch.device("cpu")  # change to "cuda" if available & desired

# === Policy class matching training architecture ===
class CPOPolicy:
    def __init__(self, obs_dim, act_dim, device):
        self.device = device
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, act_dim)
        ).to(self.device)
        # log std parameter: NOTE this is not saved in your current training script, so remains zero
        self.log_std = torch.nn.Parameter(torch.zeros(act_dim, device=self.device))

    def load(self, path):
        """
        Load a checkpoint saved by the training script.
        Falls back gracefully if `path` is just a plain state-dict.
        """
        ckpt = torch.load(path, map_location=self.device)

        # -- if training saved the full dict (recommended) --
        if isinstance(ckpt, dict) and "policy_net" in ckpt:
            self.policy_net.load_state_dict(ckpt["policy_net"])
            if "log_std" in ckpt:          # restore learned exploration scale
                self.log_std.data.copy_(ckpt["log_std"].to(self.device))
        else:
            # backward-compat: assume ckpt IS the policy state-dict itself
            self.policy_net.load_state_dict(ckpt)

    def get_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mu = self.policy_net(obs_t)  # (1, act_dim)
            if deterministic:
                action = mu.squeeze(0).cpu().numpy()
            else:
                std = torch.exp(self.log_std)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample().squeeze(0).cpu().numpy()
        return action

# === Environment setup ===
env = gym.make(ENV_NAME)
# seeding for reproducibility
env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

# instantiate policy
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy = CPOPolicy(obs_dim, act_dim, DEVICE)
policy.load(MODEL_PATH)

# === Evaluation loop ===
returns = []
costs = []
episode_records = []

# Metrics to try to extract per episode (fallbacks handled)
metric_names = [
    "Average_episode_reward",
    "interaction_forceB_counter",
    "TT",
    "reach_success",
    "jerk_mag",
    "rms_jerk",
    "acceleration_ee"
]

for ep in range(N_EVAL_EPISODES):
    obs, _ = env.reset(seed=SEED + ep)  # vary seed slightly
    done = False
    ep_return = 0.0
    ep_cost = 0.0
    # rollout until termination
    while True:
        action = policy.get_action(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(action)
        cost = info.get("cost", 0.0)
        ep_return += rew
        ep_cost += cost
        if term or trunc:
            break

    returns.append(ep_return)
    costs.append(ep_cost)

    # gather additional per-episode metrics if available
    record = {
        "Rewards": ep_return,
        "Costs": ep_cost
    }
    for name in metric_names:
        val = getattr(env, name, None)
        if val is None:
            record[name] = None
        else:
            try:
                if isinstance(val, (list, np.ndarray)):
                    record[name] = val[-1]
                else:
                    record[name] = val
            except Exception:
                record[name] = val
    episode_records.append(record)

# === Summary statistics ===
mean_reward = np.mean(returns)
std_reward = np.std(returns)
mean_cost = np.mean(costs)
print(f"Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")
print(f"Mean cost: {mean_cost:.3f}")

# success rate if available
if any(rec.get("reach_success") is not None for rec in episode_records):
    successes = [rec["reach_success"] for rec in episode_records if rec["reach_success"] is not None]
    success_rate = sum(successes) / len(successes)
    print(f"Success rate (reach_success): {success_rate:.3f}")

# === Save to Excel ===
df = pd.DataFrame(episode_records)
# Normalize column names for plotting
# e.g., rename interaction_forceB_counter (if present) to Force, TT to Steps
if "interaction_forceB_counter" in df.columns:
    df = df.rename(columns={"interaction_forceB_counter": "Force"})
if "TT" in df.columns:
    df = df.rename(columns={"TT": "Steps"})
if "reach_success" in df.columns:
    df = df.rename(columns={"reach_success": "success"})
# Export
os.makedirs(os.path.dirname(EXCEL_OUT), exist_ok=True)
df.to_excel(EXCEL_OUT, index=False)
print(f"Saved evaluation data to {EXCEL_OUT}")

# === Post-processing + plotting similar to your SB3 script ===
# Filter non-zero where appropriate
Force = df["Force"][df["Force"].notna() & (df["Force"] != 0)] if "Force" in df.columns else pd.Series([])
Reward = df["Rewards"][df["Rewards"].notna() & (df["Rewards"] != 0)]
Steps = df["Steps"][df["Steps"].notna() & (df["Steps"] != 0)]

Force_list = Force.tolist()
Reward_list = Reward.tolist()
Steps_list = Steps.tolist()

A = len(Reward_list)
A_plot = list(range(1, A + 1))
B = len(Force_list)
B_plot = list(range(1, B + 1))
C = len(Steps_list)
C_plot = list(range(1, C + 1))

# Create a single figure with 3 horizontal subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Force vs Episodes
if B > 0:
    axs[0].plot(B_plot, Force_list, linestyle='-')
    axs[0].set_xlabel('Episodes', fontsize=14)
    axs[0].set_ylabel('Force (N)', fontsize=14)
    axs[0].tick_params(axis='both', labelsize=12)
    axs[0].grid(True)
    axs[0].set_title("Force per Episode", fontsize=16)
else:
    axs[0].text(0.5, 0.5, "No Force Data", ha='center', va='center')
    axs[0].set_title("Force per Episode", fontsize=16)

# Plot 2: Reward vs Episodes
if A > 0:
    axs[1].plot(A_plot, Reward_list, linestyle='-')
    axs[1].set_xlabel('Episodes', fontsize=14)
    axs[1].set_ylabel('Return', fontsize=14)
    axs[1].tick_params(axis='both', labelsize=12)
    axs[1].grid(True)
    axs[1].set_title("Reward per Episode", fontsize=16)
else:
    axs[1].text(0.5, 0.5, "No Reward Data", ha='center', va='center')
    axs[1].set_title("Reward per Episode", fontsize=16)

# Plot 3: Steps vs Episodes
if C > 0:
    axs[2].plot(C_plot, Steps_list, linestyle='-')
    axs[2].set_xlabel('Episodes', fontsize=14)
    axs[2].set_ylabel('Steps to Reach (Timesteps)', fontsize=14)
    axs[2].tick_params(axis='both', labelsize=12)
    axs[2].grid(True)
    axs[2].set_title("Steps per Episode", fontsize=16)
else:
    axs[2].text(0.5, 0.5, "No Steps Data", ha='center', va='center')
    axs[2].set_title("Steps per Episode", fontsize=16)

plt.tight_layout()
plt.show()

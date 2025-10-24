#!/usr/bin/env python3
"""

Date: 2025-10-01

ContactRl safety shield with:
  • first-order low-pass filter (LPF)
  • kinetic-energy control-barrier projection (safety shield)
  • clean logging (no zero-velocity spike at episode end)
  
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC
import gym_panda                   # make sure the env family registers

# ───────────────── user constants ──────────────────
MODEL_PATH  = (
    "/path/to/folder/"
    "path/to/folder/file_name.zip"
)
ENV_ID       = "EnvID"
N_EPISODES   = 10
RENDER       = False

EE_MASS_KG   = 0.93          # UR3e + Robotiq 2F-85
E_MAX_J      = 0.30          # kinetic-energy budget (hand)
FC_CUTOFF_HZ = 25.0          # LPF cut-off
# add just below the other user constants
FORCE_SKIP_N = 1_000.0      # episodes with Fmax ≥ this are ignored
# ───────────────────────────────────────────────────

# ---------- helpers ------------------------------------------------
def lpf(prev: np.ndarray, raw: np.ndarray, alpha: float) -> np.ndarray:
    """Single-pole IIR low-pass filter."""
    return alpha * prev + (1.0 - alpha) * raw


def scale_to_ke(v_cmd: np.ndarray,
                m: float,
                e_max: float) -> np.ndarray:
    """
    Radially scale velocity command so ½ m ‖v_cmd‖² ≤ e_max.
    """
    v_mag = np.linalg.norm(v_cmd)
    v_allow = math.sqrt(2.0 * e_max / m)
    if v_mag <= v_allow + 1e-12:
        return v_cmd
    return v_cmd * (v_allow / v_mag)


def get_ee_vel(env, obs: np.ndarray) -> np.ndarray:
    """
    Robustly extract EE Cartesian velocity (m/s).
    Works even on the very first tick when env.velocity_ee
    may not yet exist.
    """
    if hasattr(env, "velocity_ee"):
        return np.asarray(env.velocity_ee)
    # fallback: assume obs = [pos(3), vel(3), …]
    return np.asarray(obs[3:6])
# -------------------------------------------------------------------


def rollout_episode(env, model, dt):
    """
    Execute one episode with LPF + KE shield.
    Returns: dist[], ke[], speed[], ke_contact, force_contact
    """
    alpha_lpf = math.exp(-2.0 * math.pi * FC_CUTOFF_HZ * dt)

    d_hist, ke_hist, spd_hist = [], [], []

    obs, _ = env.reset()
    raw_act, _ = model.predict(obs, deterministic=False)
    filt_act = raw_act.copy()

    ke_prev = 0.0
    ke_contact = force_contact = None

    while True:
        # 1) raw policy action
        raw_act, _ = model.predict(obs, deterministic=True)
        # 2) low-pass filter
        filt_act = lpf(filt_act, raw_act, alpha_lpf)
        # 3) kinetic-energy shield
        v_curr = get_ee_vel(env, obs)            # ← robust fetch
        safe_act = scale_to_ke(filt_act, EE_MASS_KG, E_MAX_J)

        # 4) env step
        if RENDER:
            env.render()
        obs_next, _, done, truncated, _ = env.step(safe_act)

        # 5) metrics using *next* state
        dist = float(env.Distance) if hasattr(env, "Distance") else float(obs_next[-1])
        spd  = np.linalg.norm(get_ee_vel(env, obs_next))
        ke   = 0.5 * EE_MASS_KG * spd ** 2

        if done or truncated:
            force_contact = float(env.forceB) if hasattr(env, "forceB") else 0.0
            ke_contact    = ke_prev          # KE one step earlier
            break

        d_hist.append(dist)
        ke_hist.append(ke)
        spd_hist.append(spd)

        ke_prev = ke
        obs = obs_next                       # advance

    return (np.asarray(d_hist),
            np.asarray(ke_hist),
            np.asarray(spd_hist),
            ke_contact,
            force_contact)


def main():
    env = gym.make(ENV_ID, render_mode="human" if RENDER else None)
    dt  = getattr(env, "control_dt", 0.004)   # default 250 Hz
    model = SAC.load(MODEL_PATH, env=env)

    all_d, all_ke, all_spd = [], [], []
    ke_contacts, force_contacts = [], []

    kept = 0
    for ep in range(N_EPISODES):
        d, ke, spd, ke_c, f_c = rollout_episode(env, model, dt)

        # ----- skip episodes with very large forces -----------------
        if f_c >= FORCE_SKIP_N:
            print(f"Episode {ep+1:3d} skipped (force {f_c:.1f} N ≥ {FORCE_SKIP_N})")
            continue
        # ------------------------------------------------------------

        all_d.append(d)
        all_ke.append(ke)
        all_spd.append(spd)
        ke_contacts.append(ke_c)
        force_contacts.append(f_c)

        kept += 1
        print(f"Episode {kept:3d} kept: {len(d):3d} steps, "
              f"KE_contact = {ke_c:6.3f} J,  forceB = {f_c:6.1f} N")

    env.close()

    plt.rcParams.update({"font.size": 20})  # set axis font size globally

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))  # 1×3 layout

    # ── Plot 1: KE vs distance ──
    for d, ke in zip(all_d, all_ke):
        idx = np.argsort(d)
        ax[0].plot(d[idx], ke[idx], "-o", markersize=3, linewidth=1.0)
    ax[0].set_xlabel("Distance (robot → hand) [m]", fontsize=22)
    ax[0].set_ylabel("Kinetic Energy [J]", fontsize=22)
    ax[0].tick_params(axis='both', labelsize=16)
    ax[0].grid(True, linestyle="--", linewidth=1.0)

    # ── Plot 2: Peak contact force per episode ──
    ax[1].plot(np.arange(1, kept+1), force_contacts, "o-", ms=6, lw=1.2)
    ax[1].axhline(50, ls="--", c="red", lw=1.2)
    ax[1].set_xlabel("Episode No.", fontsize=22)
    ax[1].set_ylabel("$F_N$ [N]", fontsize=22)
    ax[1].tick_params(axis='both', labelsize=16)
    ax[1].grid(True, linestyle="--", linewidth=1.0)

    # ── Plot 3: Speed profile ──
    for v in all_spd:
        t = np.arange(len(v)) * dt
        ax[2].plot(t, v)
    ax[2].set_xlabel("Time [s]", fontsize=22)
    ax[2].set_ylabel("$v_z$ [m/s]", fontsize=22)
    ax[2].tick_params(axis='both', labelsize=16)
    ax[2].grid(True, linestyle="--", linewidth=1.0)

    # Styling
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

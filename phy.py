"""
ContactRL physical implementation with moderated speed

- Streams commands via RTDEControl and reads state via RTDEReceive
- Policy proposes Cartesian deltas

"""

import time, math, socket, os, re
import numpy as np

import rtde_receive
import rtde_control
from stable_baselines3 import SAC

# ── Robot / control constants
ROBOT_HOST   = "192.168.3.10"
USE_POLICY   = True
MODEL_PATH   = ""
N_EPISODES   = 1
DRY_RUN      = False

# Global speed scaling
SPEED_SCALE   = 1.5

DT            = 0.008
ACCEL         = 1.0 * SPEED_SCALE

VEL_CART_MAX  = 0.12 * SPEED_SCALE * 1.3
TIMEOUT_CMD   = 0.25

LOOKAHEAD     = 0.10
GAIN          = 300

SLOW_DIST     = 0.06
GOAL_TOL      = 0.01
AXIS_TOL      = 0.01

MAX_ACC       = 0.8 * SPEED_SCALE * 1.3

EE_MASS_KG    = 0.93
E_MAX_J       = 0.20

FC_CUTOFF_HZ        = 40.0
USE_FORCE_LP        = True
FORCE_LP_CUTOFF_HZ  = 20.0

CONTACT_USE_Z_ONLY  = True
FORCE_CONTACT_ON_N  = 0.8
FORCE_CONTACT_OFF_N = 0.5

FORCE_ABORT_N       = 60.0

KP_POS        = 6.0 * SPEED_SCALE
POS_ERR_MAX   = 0.15

ACTION_DELTA_MAX_VEC = np.array([0.02, 0.02, 0.02], dtype=float)
HAND_CARE_DIR   = np.array([0.0, 0.0, -1.0], dtype=float)
V_REF_NO_POLICY = 0.03 * SPEED_SCALE
DEADBAND_EPS    = 1e-5

# Goal (base frame)
HAND_POS_M = np.array([-0.387, -0.128, 0.26], dtype=float)

# Gripper close script (send as-is over 30002) 
SECONDARY_PORT         = 30002
GRIPPER_CLOSE_SCRIPT   = "/path/to/your/folder"  # close-only script
CLOSE_ON_CONTACT       = False   # set True if you also want to close when contact stops motion
PAUSE_AFTER_STOP_SEC   = 0.12    # small pause after servoStop before sending script

# Helpers
def lpf(prev, raw, alpha): return alpha * prev + (1.0 - alpha) * raw

def scale_to_ke(v, m, e):
    n = np.linalg.norm(v); vmax = math.sqrt(2.0*e/m)
    return v if n <= vmax or n == 0 else v*(vmax/n)

def clamp_vel(v, vmax):
    n = np.linalg.norm(v)
    return v if n <= vmax or n == 0 else v*(vmax/n)

def rate_limit(prev, cur, max_dv):
    dv = cur - prev
    n  = np.linalg.norm(dv)
    return cur if n <= max_dv or n == 0 else prev + dv*(max_dv/n)

def deadband(v, eps): return v if np.linalg.norm(v) > eps else np.zeros_like(v)

def reachable(host, port=30004, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout): return True
    except OSError:
        return False

def build_observation(rr):
    tcp = np.array(rr.getActualTCPPose(), dtype=float)
    spd = np.array(rr.getActualTCPSpeed(), dtype=float)
    return np.concatenate([HAND_POS_M, tcp[:3], spd[:3]], axis=0)

def calibrate_force(rr, dur=1.0, dt=0.008):
    n = max(1, int(dur/dt)); acc = np.zeros(3,float)
    for _ in range(n):
        F = np.array(rr.getActualTCPForce()[:3], float); acc += F; time.sleep(dt)
    F0 = acc/n; print(f"[CAL] F0 = {np.round(F0,4)} N, |F0| = {np.linalg.norm(F0):.2f} N"); return F0

def print_final_summary(rr, reason: str):
    time.sleep(0.05)
    final_ee = np.array(rr.getActualTCPPose(), dtype=float)[:3]
    err_vec  = HAND_POS_M - final_ee
    err_norm = float(np.linalg.norm(err_vec))

def smoothstep01(x):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    return x*x*(3.0 - 2.0*x)

def sb3_predict_action(model, obs, deterministic=True):
    out = model.predict(obs, deterministic=deterministic)
    return (out[0] if isinstance(out, tuple) else out).ravel().astype(float)

# Secondary interface: send a URScript file verbatim 
def send_urscript_file(host: str, port: int, path: str, timeout: float = 3.0):
    with open(path, "rb") as f:
        payload = f.read()
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(payload)

# Rollout (no logging, no CSV) 
def rollout_episode(rc, rr, model, dt, F0):
    alpha_vel = math.exp(-2.0*math.pi*FC_CUTOFF_HZ*dt)
    filt_lin  = np.zeros(3,float)
    last_cmd  = np.zeros(3,float)

    f_lp      = 0.0
    contact_on = False

    t0 = time.perf_counter()
    state = "APPROACH"
    end_reason = "unknown"
    sent_close = False

    print("[STATE] APPROACH → stop on goal (radial or X/Z) OR contact; close gripper on GOAL only by sending your script.")

    try:
        while True:
            t_abs = time.perf_counter()

            obs  = build_observation(rr)
            ee_p = obs[3:6]
            ee_v = obs[6:9]

            d_vec = HAND_POS_M - ee_p
            ex = float(d_vec[0]); ez = float(d_vec[2])

            wrench = np.array(rr.getActualTCPForce(), dtype=float)
            F_raw  = wrench[:3]
            F_c    = F_raw - F0

            metric = abs(F_c[2]) if CONTACT_USE_Z_ONLY else float(np.linalg.norm(F_c))
            if USE_FORCE_LP:
                aF = math.exp(-2.0*math.pi*FORCE_LP_CUTOFF_HZ*dt)
                f_lp = aF*f_lp + (1.0-aF)*metric
                metric_used = f_lp
            else:
                metric_used = metric

            if (not contact_on) and (metric_used >= FORCE_CONTACT_ON_N):
                contact_on = True
                print(f"[CONTACT] metric rose to {metric_used:.2f} N ≥ {FORCE_CONTACT_ON_N} N — STOP.")
            elif contact_on and (metric_used <= FORCE_CONTACT_OFF_N):
                contact_on = False
                print(f"[CONTACT] metric fell to {metric_used:.2f} N ≤ {FORCE_CONTACT_OFF_N} N.")

            if metric_used >= FORCE_ABORT_N:
                print(f"[ABORT] |F|_used={metric_used:.1f} N ≥ {FORCE_ABORT_N}. Stopping.")
                soft_brake(rc, rr, dt)
                state = "HOLD"
                end_reason = "force abort threshold reached"

            # periodic status
            if int((time.perf_counter()-t0)/dt) % 10 == 1:
                axis_err = np.abs(d_vec)
                goal_err = np.linalg.norm(d_vec)
                print(f"[DIST] radial={goal_err*1000:.1f} mm | "
                      f"dx={axis_err[0]*1000:.1f} mm, dz={axis_err[2]*1000:.1f} mm | "
                      f"tcp={np.round(ee_p,4)} v={np.round(ee_v,4)} | contact={contact_on} ({metric_used:.2f} N)")

            if state == "APPROACH":
                axis_err = np.abs(d_vec)
                goal_err = np.linalg.norm(d_vec)
                axis_hit = np.array([axis_err[0] <= AXIS_TOL, False, axis_err[2] <= AXIS_TOL])

                if contact_on:
                    soft_brake(rc, rr, dt)
                    state = "HOLD"
                    end_reason = "contact reached"
                    if CLOSE_ON_CONTACT and not DRY_RUN and not sent_close:
                        _send_close_script_once()
                        sent_close = True

                elif (goal_err <= GOAL_TOL) or np.any(axis_hit):
                    if goal_err <= GOAL_TOL:
                        end_reason = f"radial within {GOAL_TOL:.3f} m"
                        print("goal reached (radial) → HOLD.")
                    else:
                        end_reason = f"axis within {AXIS_TOL:.3f} m"
                        print("goal reached (axis X/Z) → HOLD.")

                    soft_brake(rc, rr, dt)
                    state = "HOLD"

                    if not DRY_RUN and not sent_close:
                        _send_close_script_once()
                        sent_close = True

                else:
                    # normal approach control
                    if USE_POLICY:
                        a_vec = sb3_predict_action(model, obs, deterministic=True)
                        delta_p = np.tanh(a_vec[:3]) * ACTION_DELTA_MAX_VEC
                    else:
                        delta_p = HAND_CARE_DIR * 1e-6

                    if goal_err <= GOAL_TOL:
                        speed_scale = 0.0
                    else:
                        denom = max(1e-6, SLOW_DIST - GOAL_TOL)
                        x = (goal_err - GOAL_TOL) / denom
                        speed_scale = smoothstep01(x)
                    v_mag_target = VEL_CART_MAX * speed_scale

                    err_sat = np.clip(d_vec, -POS_ERR_MAX, POS_ERR_MAX)
                    v_goal  = KP_POS * err_sat
                    ng = np.linalg.norm(v_goal)
                    if ng > 1e-9 and ng > v_mag_target + 1e-12:
                        v_goal = v_goal * (v_mag_target / ng)

                    if np.linalg.norm(delta_p) > 1e-6:
                        dh = delta_p / np.linalg.norm(delta_p)
                        v_des = 0.9*v_goal + 0.1*(dh * np.linalg.norm(v_goal))
                    else:
                        v_des = v_goal

                    v_lpf = lpf(filt_lin, v_des, alpha_vel)

                    # safety shaping (no counters, just action)
                    safe_lin = scale_to_ke(v_lpf, EE_MASS_KG, E_MAX_J)
                    safe_lin = deadband(safe_lin, DEADBAND_EPS)
                    safe_lin = clamp_vel(safe_lin, VEL_CART_MAX)
                    safe_lin = rate_limit(last_cmd, safe_lin, MAX_ACC*dt)

                    filt_lin = v_lpf.copy()
                    last_cmd = safe_lin.copy()

                    cur_pose = np.array(rr.getActualTCPPose(), dtype=float)
                    target_pose = cur_pose.copy()
                    target_pose[:3] = cur_pose[:3] + safe_lin * dt

                    if not DRY_RUN:
                        ok = rc.servoL(target_pose.tolist(), ACCEL, VEL_CART_MAX, DT, LOOKAHEAD, GAIN)
                        if not ok:
                            print("[WARN] servoL returned False (controller busy?)")

            else:
                # HOLD: do nothing; waiting for user/program end
                pass

            delay = dt - (time.perf_counter() - t_abs)
            if delay > 0: time.sleep(delay)

    except KeyboardInterrupt:
        print("[USER] Interrupted. Ending loop.")
    finally:
        print_final_summary(rr, end_reason)

def _send_close_script_once():
    """Stop motion cleanly, wait a beat, then send your close script once."""
    time.sleep(PAUSE_AFTER_STOP_SEC)
    try:
        send_urscript_file(ROBOT_HOST, SECONDARY_PORT, GRIPPER_CLOSE_SCRIPT)
        print(f"[GRIPPER] Sent close script: {GRIPPER_CLOSE_SCRIPT}")
    except Exception as e:
        print(f"[GRIPPER] Failed to send close script: {e}")

def soft_brake(rc, rr, dt, brake_time=0.25, steps=None):
    if DRY_RUN:
        print("[DRY] soft_brake()"); return
    if steps is None: steps = max(1, int(brake_time/dt))
    pose = np.array(rr.getActualTCPPose(), float)
    for _ in range(steps):
        rc.servoL(pose.tolist(), ACCEL, 0.0, dt, LOOKAHEAD, GAIN)  # hold pose
        time.sleep(dt)
    try:
        rc.servoStop()
    except Exception:
        pass

# Main 
def main():
    if not reachable(ROBOT_HOST):
        print(f"[WARN] Cannot reach {ROBOT_HOST}:30004. Set DRY_RUN=True for offline testing.")

    try:
        rr = rtde_receive.RTDEReceiveInterface(ROBOT_HOST)
        rc = rtde_control.RTDEControlInterface(ROBOT_HOST)
    except Exception as e:
        print(f"[WARN] RTDE init failed: {e}"); return

    try:
        sc = rr.getSpeedScaling()
        print(f"[INFO] Pendant speed scaling: {sc*100:.1f}%")
    except Exception:
        pass

    try:
        rc.setPayload(EE_MASS_KG, [0, 0, 0])
        print(f"[INFO] Payload set to {EE_MASS_KG} kg.")
    except Exception as e:
        print(f"[WARN] setPayload failed: {e}")

    F0 = calibrate_force(rr, dur=1.0, dt=DT)

    model = None
    if USE_POLICY:
        try:
            model = SAC.load(MODEL_PATH)
            print("[INFO] SAC model loaded (obs=9D, action=3D Δposition).")
        except Exception as e:
            print(f"[ERROR] Load model failed: {e}"); return

    for ep in range(N_EPISODES):
        print(f"\n=== Episode {ep+1}/{N_EPISODES} ===")
        rollout_episode(rc, rr, model, DT, F0)

    if not DRY_RUN:
        soft_brake(rc, rr, DT)

if __name__ == "__main__":
    main()

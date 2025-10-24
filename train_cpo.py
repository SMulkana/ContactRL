"""

Date: 3 August 2025

CPO Training Script (Achiam-style) for ContactLag Environment (Gymnasium + SB3)

This script implements Constrained Policy Optimization (CPO) following Achiam et al.
to train a continuous-control policy in ContactCPO env under explicit safety
constraints. Performance (return) is improved while keeping the expected safety cost
below a target limit.

Core components:
- Policy update: natural-gradient step via conjugate gradients with KL trust region
  and backtracking line search.
- Critics: separate value and cost value networks; targets computed with GAE(γ, λ).
- Advantages: reward and cost advantages normalized per batch.
- Safety: per-step cost read from env (`info["cost"]`); constraint slack enforced in
  the CPO step; tracks discounted/undiscounted costs.
- Fallback: entropy-regularized policy gradient with step clipping if the CPO step
  violates KL or cost constraints.
- Reproducibility & UX: global seeding; tqdm progress bar; Excel export of key metrics.

Update paths and hyperparameters (e.g., `cost_limit`, `max_kl`, `steps_per_epoch`)
as needed for your setup.

"""
import os
import sys
import gymnasium as gym
import numpy as np
import torch
import random
import time
import pandas as pd
from tqdm import tqdm
import env


SEED = 5
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Logging
log_dir = "/path/to/folder/"
os.makedirs(log_dir, exist_ok=True)
excel_out_path = os.path.join(log_dir, "file_name.xlsx")

# Environment
env = gym.make('envID')
obs, _ = env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

total_timesteps = 1000_000
steps_per_epoch = 2000
n_epochs = total_timesteps // steps_per_epoch

class CPOAgent:
    def __init__(self, env, seed, cost_limit, epochs, steps_per_epoch, gamma=0.99, lam=0.95,
                 max_kl=0.01, cg_iters=10, cg_damping=0.1, vf_lr=1e-3, vf_iters=5,
                 train_v_iters=5):
        self.env = env
        self.seed = seed
        self.cost_limit = cost_limit
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.min_cpo_step_norm = 1e-8

        self.train_v_iters  = train_v_iters
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Policy network outputs mean
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, act_dim)
        )
        # Log std parameter
        self.log_std = torch.nn.Parameter(torch.zeros(act_dim))
        # Value network
        self.value_fn = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )
        # Cost value network
        self.cost_fn = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )
        # Optimizers for value and cost critic
        self.value_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=vf_lr)
        self.cost_optimizer = torch.optim.Adam(self.cost_fn.parameters(), lr=vf_lr)
        self.policy_optimizer = torch.optim.Adam(list(self.policy_net.parameters()) + [self.log_std], lr=3e-4)
        self.entropy_coef = 0.01  # tune as needed
        self.fallback_grad_clip = 1.0  # optional gradient clipping

    def _get_dist(self, obs):
        mu = self.policy_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

    def fisher_vector_product(self, obs, v):
        dist = self._get_dist(obs)
        # Old policy distribution
        mu_old = dist.mean.detach()
        std_old = dist.stddev.detach()
        dist_old = torch.distributions.Normal(mu_old, std_old)
        # Compute kl divergence
        dist_new = dist
        kl = torch.distributions.kl_divergence(dist_old, dist_new).mean()
        grads = torch.autograd.grad(kl, list(self.policy_net.parameters()) + [self.log_std], create_graph=True)
        flat_grad_kl = torch.cat([g.view(-1) for g in grads])
        kl_v = (flat_grad_kl * v).sum()
        grad_grad_kl = torch.autograd.grad(kl_v, list(self.policy_net.parameters()) + [self.log_std])
        flat_grad_grad_kl = torch.cat([g.contiguous().view(-1) for g in grad_grad_kl])
        return flat_grad_grad_kl + self.cg_damping * v

    def conjugate_gradients(self, f_Ax, b, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            Avp = f_Ax(p)
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_gae(self, rews, vals, last_val, gamma, lam):
        T = len(rews)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            v_next = vals[t+1] if t+1 < T else last_val
            delta  = rews[t] + gamma * v_next - vals[t]
            adv[t] = delta + gamma * lam * last_gae
            last_gae = adv[t]
        ret = adv + vals
        return adv, ret
    

    def compute_cpo_step(self, flat_g, flat_b, obs_tensor, Jc_old):
        params = list(self.policy_net.parameters()) + [self.log_std]
        with torch.no_grad():
            old_dist = self._get_dist(obs_tensor)  # frozen reference

        def Hx(v):
            dist_new = self._get_dist(obs_tensor)
            kl = torch.distributions.kl_divergence(old_dist, dist_new).mean()
            grads = torch.autograd.grad(kl, params, create_graph=True)
            flat_grad_kl = torch.cat([g.view(-1) for g in grads])
            kl_v = (flat_grad_kl * v).sum()
            grad_grad_kl = torch.autograd.grad(kl_v, params)
            flat_grad_grad_kl = torch.cat([g.contiguous().view(-1) for g in grad_grad_kl])
            return flat_grad_grad_kl + self.cg_damping * v

        s = self.conjugate_gradients(Hx, flat_g, nsteps=self.cg_iters)
        q = self.conjugate_gradients(Hx, flat_b, nsteps=self.cg_iters)

        gs = flat_g.dot(s)
        bs = flat_b.dot(s)
        bb = flat_b.dot(q)

        kl_epsilon = self.max_kl  # target: 0.5 * s^T H s <= kl_epsilon
        slack = self.cost_limit - Jc_old  # allowed first-order cost change

            # safe helper to scale s to satisfy KL constraint: sqrt(2*epsilon / (g^T s))
        def scale_reward_step(s_vec, g_dot_s):
            if g_dot_s <= 0:
                return torch.zeros_like(s_vec)
            # avoid division by extremely small number
            denom = torch.clamp(g_dot_s, min=1e-12)
            scale = torch.sqrt(2.0 * kl_epsilon / denom)
            # optional cap to avoid exploding step
            max_scale = 10.0
            if scale > max_scale:
                scale = max_scale
            return s_vec * scale

        # Feasibility / fallback if already violating cost
        if slack < 0:
            return scale_reward_step(-q, -flat_b.dot(q))

        # If reward-only step satisfies cost slack, take KL-scaled reward step
        if bs <= slack:
            return scale_reward_step(s, gs)

        # General case: solve quadratic for dual variable nu ≥ 0
        delta0 = 2.0 * kl_epsilon  # used in the quadratic terms as in Achiam
        A = (slack ** 2) * bb - delta0 * (bb ** 2)
        B = 2 * bs * (delta0 * bb - slack ** 2)
        C = (slack ** 2) * gs - delta0 * (bs ** 2)

        disc = B * B - 4 * A * C
        disc = torch.clamp(disc, min=0.0)
        sqrt_disc = torch.sqrt(disc)
        denom_nu = 2 * A + 1e-12
        nu1 = (-B + sqrt_disc) / denom_nu
        nu2 = (-B - sqrt_disc) / denom_nu

        candidates = torch.stack([nu1, nu2])
        positive = candidates[candidates > 0]
        nu = positive.max() if positive.numel() > 0 else torch.tensor(0.0, device=flat_g.device)

        denom = bs - nu * bb
        if denom <= 0:
            # fallback to scaled reward step if denominator invalid
            return scale_reward_step(s, gs)

        alpha = slack / (denom + 1e-12)
        return alpha * (s - nu * q)

    def train(self, update_fn=None):
        for epoch in range(self.epochs):
            # === Collect a full batch ===
            obs = self.env.reset(seed=self.seed)[0]
            obs_buf, act_buf, logp_buf, ret_buf, cost_ret_buf, done_buf = [], [], [], [], [], []
            for step in range(self.steps_per_epoch):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                dist = self._get_dist(obs_t)
                act = dist.sample().squeeze().detach().numpy()
                logp = dist.log_prob(torch.tensor(act, dtype=torch.float32)).sum().item()
                new_obs, rew, term, trunc, info = self.env.step(act)
                cost = info.get("cost", 0.0)
                done = term or trunc

                # store transition
                obs_buf.append(obs)
                act_buf.append(act)
                logp_buf.append(logp)
                ret_buf.append(rew)
                cost_ret_buf.append(cost)
                done_buf.append(done)

                # next obs (reset if episode ended)
                obs = new_obs if not (term or trunc) else self.env.reset(seed=self.seed)[0]

                if update_fn:
                    update_fn(1)

            # === After batch collection: compute advantages/returns ===
            # episode boundaries
            episode_ends = [i for i, d in enumerate(done_buf) if d]
            if not episode_ends or episode_ends[-1] != len(ret_buf) - 1:
                episode_ends.append(len(ret_buf) - 1)

            all_adv, all_ret = [], []
            all_cost_adv, all_cost_ret = [], []
            start_idx = 0
            for end_idx in episode_ends:
                rews = ret_buf[start_idx : end_idx+1]
                costs = cost_ret_buf[start_idx : end_idx+1]

                obs_segment = obs_buf[start_idx : end_idx+1]
                obs_tensor_seg = torch.tensor(obs_segment, dtype=torch.float32)
                with torch.no_grad():
                    vals = self.value_fn(obs_tensor_seg).detach().cpu().numpy().flatten()
                    cost_vals = self.cost_fn(obs_tensor_seg).detach().cpu().numpy().flatten()

                # bootstrap from last state in segment
                last_obs = obs_buf[end_idx]
                last_val = self.value_fn(torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0)).item()
                last_cost_val = self.cost_fn(torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0)).item()

                adv, ret = self.compute_gae(rews, vals, last_val, self.gamma, self.lam)
                cost_adv, cost_ret = self.compute_gae(costs, cost_vals, last_cost_val, self.gamma, self.lam)

                all_adv.extend(adv)
                all_ret.extend(ret)
                all_cost_adv.extend(cost_adv)
                all_cost_ret.extend(cost_ret)

                start_idx = end_idx + 1

            # convert to numpy and normalize
            adv_np = np.array(all_adv, dtype=np.float32)
            cost_adv_np = np.array(all_cost_adv, dtype=np.float32)
            ret_np = np.array(all_ret, dtype=np.float32)
            cost_ret_np = np.array(all_cost_ret, dtype=np.float32)

            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
            cost_adv_np = (cost_adv_np - cost_adv_np.mean()) / (cost_adv_np.std() + 1e-8)

            # to torch
            obs_tensor = torch.tensor(obs_buf, dtype=torch.float32)
            act_tensor = torch.tensor(act_buf, dtype=torch.float32)
            adv_tensor = torch.tensor(adv_np, dtype=torch.float32)
            cost_adv_tensor = torch.tensor(cost_adv_np, dtype=torch.float32)
            ret_tensor = torch.tensor(ret_np, dtype=torch.float32)
            cost_ret_tensor = torch.tensor(cost_ret_np, dtype=torch.float32)

            # === Critic updates ===
            for _ in range(self.train_v_iters):
                vpred = self.value_fn(obs_tensor).squeeze()
                loss = ((vpred - ret_tensor)**2).mean()
                self.value_optimizer.zero_grad(); loss.backward(); self.value_optimizer.step()

                cpred = self.cost_fn(obs_tensor).squeeze()
                closs = ((cpred - cost_ret_tensor)**2).mean()
                self.cost_optimizer.zero_grad(); closs.backward(); self.cost_optimizer.step()

            # === Policy gradient and CPO step ===
            params = list(self.policy_net.parameters()) + [self.log_std]
            dist = self._get_dist(obs_tensor)
            logp = dist.log_prob(act_tensor).sum(dim=-1)

            g = torch.autograd.grad((logp * adv_tensor).mean(), params, retain_graph=True)
            b = torch.autograd.grad((logp * cost_adv_tensor).mean(), params)
            flat_g = torch.cat([gi.view(-1) for gi in g]).detach()
            flat_b = torch.cat([bi.view(-1) for bi in b]).detach()

            # Slack / expected cost
            Jc_old = cost_ret_tensor.mean().item()
            slack = self.cost_limit - Jc_old

            # Logging for diagnostics (optional but recommended)
            print(f"[CPO] Epoch {epoch+1} Jc_old={Jc_old:.4f} slack={slack:.4f}")

            # Compute constrained step
            step = self.compute_cpo_step(flat_g, flat_b, obs_tensor, Jc_old)
    
            with torch.no_grad():
                # snapshot of the old policy distribution – *do not* update after this
                old_dist_snapshot = self._get_dist(obs_tensor)

            def fvp_old(v):
                """
                Fisher-vector product that ALWAYS measures KL against `old_dist_snapshot`,
                so it stays valid even while we temporarily move the parameters during
                back-tracking.
                """
                dist_new = self._get_dist(obs_tensor)        # uses current (possibly perturbed) params
                kl = torch.distributions.kl_divergence(old_dist_snapshot, dist_new).mean()
                grads = torch.autograd.grad(kl, params, create_graph=True)
                flat_grad_kl = torch.cat([g.view(-1) for g in grads])
                kl_v = (flat_grad_kl * v).sum()
                grad_grad_kl = torch.autograd.grad(kl_v, params)
                flat_grad_grad_kl = torch.cat([g.contiguous().view(-1)
                                            for g in grad_grad_kl])
                return flat_grad_grad_kl + self.cg_damping * v
            # Sanity check shapes
            expected_size = sum(p.numel() for p in params)
            assert flat_g.numel() == expected_size, "flat_g size mismatch"
            assert flat_b.numel() == expected_size, "flat_b size mismatch"
            assert step.numel() == expected_size, "CPO step size mismatch"

            # === Line search ===
            with torch.no_grad():
                old_dist = self._get_dist(obs_tensor)

            old_params = [p.data.clone() for p in params]
            alphas = [1.0, 0.5, 0.25, 0.125, 0.0625]
            chosen_step = None
            for alpha in alphas:
                full_step = alpha * step

                # approximate filters
                #fvp = self.fisher_vector_product(obs_tensor, full_step)
                kl_quadratic = 0.5 * full_step.dot(fvp_old(full_step))
                cost_sur = flat_b.dot(full_step)
                if kl_quadratic > self.max_kl or cost_sur > slack:
                    continue

                # tentative apply
                offset = 0
                for p in params:
                    numel = p.numel()
                    p.data += full_step[offset:offset+numel].view_as(p)
                    offset += numel

                # exact KL
                with torch.no_grad():
                    new_dist = self._get_dist(obs_tensor)
                    kl_exact = torch.distributions.kl_divergence(old_dist, new_dist).mean()

                if kl_exact <= self.max_kl and cost_sur <= slack:
                    chosen_step = full_step
                    break

                # revert
                for p, old in zip(params, old_params):
                    p.data.copy_(old)

            if chosen_step is None:
                chosen_step = alphas[-1] * step  # fallback

            # ensure baseline params before final apply
            for p, old in zip(params, old_params):
                p.data.copy_(old)

        # === Evaluate candidate CPO step exactly ===
        with torch.no_grad():
            # restore baseline
            for p, old in zip(params, old_params):
                p.data.copy_(old)

            # apply candidate step temporarily
            offset = 0
            for p in params:
                numel = p.numel()
                p.data += chosen_step[offset:offset+numel].view_as(p)
                offset += numel

            # exact KL and cost surrogate for chosen step
            new_dist = self._get_dist(obs_tensor)
            kl_exact_chosen = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            cost_sur_chosen = flat_b.dot(chosen_step)

            # revert to baseline before deciding
            for p, old in zip(params, old_params):
                p.data.copy_(old)

        # decide if CPO step failed
        cpo_failed = (
            kl_exact_chosen > self.max_kl or
            cost_sur_chosen > slack or
            chosen_step.norm() < self.min_cpo_step_norm
        )

        if cpo_failed:
            # --- fallback: constrained entropy-regularized policy gradient ---
            # Build fallback loss (reward + entropy) and get its gradient direction
            dist = self._get_dist(obs_tensor)
            logp = dist.log_prob(act_tensor).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            pg_loss = - (logp * adv_tensor).mean() - self.entropy_coef * entropy.mean()

            policy_params = list(self.policy_net.parameters()) + [self.log_std]
            # Compute gradient of fallback loss manually
            for p in policy_params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            pg_loss.backward(retain_graph=True)

            # Flatten the gradient to get the direction
            flat_pg = torch.cat([p.grad.view(-1) for p in policy_params]).detach()  # ∇(fallback loss)
            # Proposed step: gradient descent step (simple SGD-style)
            fallback_lr = 3e-4  # or make this a hyperparam (could reuse same as policy_optimizer)
            step_vec = -fallback_lr * flat_pg  # s = -lr * grad

            # First-order cost change: b^T s
            cost_sur_fallback = flat_b.dot(step_vec)
            # Only consider positive cost increases (i.e., violations)
            cost_increase = torch.clamp(cost_sur_fallback, min=0.0)

            # If it would violate slack, shrink the step
            if slack > 0 and cost_increase > slack:
                scale = (slack / (cost_increase + 1e-12)).clamp(max=1.0)
                step_vec = step_vec * scale

            # NEW: final safety clip on the whole vector magnitude
            step_vec = step_vec.clamp_(
                -self.fallback_grad_clip,
                    self.fallback_grad_clip
            )

            # Apply the fallback step manually (bypassing optimizer to control magnitude)
            offset = 0
            for p in policy_params:
                numel = p.numel()
                p.data += step_vec[offset:offset+numel].view_as(p)
                offset += numel

            print(f"[Fallback] Epoch {epoch+1}: CPO step rejected (KL={kl_exact_chosen:.6f}, cost_sur={cost_sur_chosen:.6f}), used entropy-PG fallback.")
        else:
            # --- apply accepted CPO step ---
            for p, old in zip(params, old_params):
                p.data.copy_(old)  # baseline

            offset = 0
            for p in params:
                numel = p.numel()
                p.data += chosen_step[offset:offset+numel].view_as(p)
                offset += numel

        # === Logging summary (always) ===
        avg_return = ret_np.mean()
        avg_cost = cost_ret_np.mean()
        violation = max(0.0, avg_cost - self.cost_limit)


    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            # optionally you can include critics if you want to restore full training state:
            "value_fn": self.value_fn.state_dict(),
            "cost_fn": self.cost_fn.state_dict(),
        }, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        # restore log_std (ensure device match)
        self.log_std.data.copy_(checkpoint["log_std"].to(self.log_std.device))
        # if you saved critics too, uncomment:
        self.value_fn.load_state_dict(checkpoint["value_fn"])
        self.cost_fn.load_state_dict(checkpoint["cost_fn"])

# Agent
agent = CPOAgent(
    env=env,
    seed=SEED,
    cost_limit=25,
    epochs=n_epochs,
    steps_per_epoch=steps_per_epoch,
    gamma=0.99,
    lam=0.95,
    max_kl=0.01,
    cg_iters=10,
    cg_damping=0.01,
    vf_lr=1e-3,
    vf_iters=5,
    train_v_iters=80,
)

# Train with progress bar
class ProgressBar:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc='Training Progress', file=sys.stdout)
    def update(self, step):
        self.pbar.update(step)
    def close(self):
        self.pbar.close()

pbar = ProgressBar(total_timesteps)
start_time = time.time()
agent.train(update_fn=pbar.update)
training_duration = time.time() - start_time
pbar.close()
print(f"Training took {training_duration:.2f} seconds.")
# Save model (if applicable)
agent.save(os.path.join(log_dir, "file_name"))

# Extract and save logged data to Excel
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

max_length = max(len(k), len(f), len(l), len(h))
k = k + [None] * (max_length - len(k))
f = f + [None] * (max_length - len(f))
l = l + [None] * (max_length - len(l))
h = h + [None] * (max_length - len(h))

df2 = pd.DataFrame({'Rewards': k, 'ForceB': f, 'Time': l, 'delta': h})
df2.to_excel(excel_out_path, index=False)

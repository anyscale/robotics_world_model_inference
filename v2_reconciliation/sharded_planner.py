"""
sharded_planner.py — data-parallel CEM across G GPUs (the on-thesis Anyscale lever: more GPUs on ONE
plan -> lower latency -> smaller staleness Δ).

One CEM plan, sample batch split across G ShardWorker actors. Each worker samples its shard of candidate
action trajectories, rolls them through its own copy of the V-JEPA predictor (the expensive part), and
returns only tiny (action_traj, score) arrays. A coordinator gathers all shards, takes the GLOBAL top-k,
refits the sampling distribution, and broadcasts the new mean/std — so the heavy rollouts parallelize and
only small vectors cross the wire. Semantics match the upstream single-process CEM.
"""
import numpy as np
import ray
import torch
import torch.nn.functional as F

from vjepa_core import WorldModelCore

MAXNORM = 0.05


@ray.remote(num_gpus=1)
class ShardWorker:
    def __init__(self):
        self.core = WorldModelCore(device="cuda")
        self.z = self.pose = self.z_goal = None
        print("[ShardWorker] ready", flush=True)

    def build_problem(self, ctx_img, goal_img, pose_np, goal_steps=8, a_star=None):
        """Encode context + roll a K-step-reachable goal (same as world.py) -> numpy z0/p0/z_goal."""
        z0 = self.core.encode(ctx_img)
        p = self.core.t(pose_np).reshape(1, 1, 7)
        a = self.core.t(a_star if a_star else [0.04, 0, 0, 0, 0, 0, 0]).reshape(1, 1, 7)
        z = z0
        for _ in range(goal_steps):
            z, p = self.core.step(z, p, a)
        return {"z0": z0.float().cpu().numpy(),
                "p0": self.core.t(pose_np).reshape(1, 1, 7).float().cpu().numpy(),
                "z_goal": z.float().cpu().numpy()}

    def set_context(self, z_np, pose_np, zg_np):
        """Cache the plan's context/goal latents on-GPU so they aren't re-sent every CEM step."""
        self.z = self.core.t(z_np); self.pose = self.core.t(pose_np); self.z_goal = self.core.t(zg_np)
        return True

    @torch.no_grad()
    def roll_and_score(self, mean_np, std_np, n, rollout, seed):
        """Sample n action trajectories ~N(mean,std), roll through the predictor, score energy to goal.
        Returns (action_trajs [n, rollout, 7], scores [n]). Sampling matches upstream cem()."""
        g = torch.Generator(device="cuda").manual_seed(int(seed))
        mean = self.core.t(mean_np); std = self.core.t(std_np)          # [rollout, 4] (xyz3 + grip1)
        frame_traj = self.z.repeat(n, 1, 1, 1)
        pose_traj = self.pose.repeat(n, 1, 1)
        goal = self.z_goal.repeat(n, 1, 1, 1)
        action_traj = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for h in range(rollout):
                s = torch.randn(n, 4, generator=g, device="cuda") * std[h] + mean[h]
                s[:, :3] = torch.clip(s[:, :3], -MAXNORM, MAXNORM)
                s[:, -1:] = torch.clip(s[:, -1:], -0.75, 0.75)
                act = torch.cat([s[:, :3], torch.zeros(n, 3, device="cuda"), s[:, -1:]], -1)[:, None]
                action_traj = act if action_traj is None else torch.cat([action_traj, act], 1)
                nf, npose = self.core._step(frame_traj, action_traj, pose_traj)
                frame_traj = torch.cat([frame_traj, nf], 1)
                pose_traj = torch.cat([pose_traj, npose], 1)
            scores = torch.mean(torch.abs(frame_traj[:, -1].flatten(1) - goal.flatten(1)), dim=-1)
        return action_traj.float().cpu().numpy(), scores.float().cpu().numpy()


def plan_sharded(workers, z_np, pose_np, zg_np, total_samples=128, cem_steps=6, rollout=2, momentum=0.15):
    """Coordinator: one CEM plan, sample batch sharded across `workers`. Returns action [1, rollout, 7]."""
    G = len(workers)
    per = total_samples // G
    topk = max(4, total_samples // 10)
    ray.get([w.set_context.remote(z_np, pose_np, zg_np) for w in workers])

    mean = np.zeros((rollout, 4), dtype=np.float32)
    std = np.concatenate([np.ones((rollout, 3), np.float32) * MAXNORM, np.ones((rollout, 1), np.float32)], -1)

    for step in range(cem_steps):
        shards = ray.get([workers[i].roll_and_score.remote(mean, std, per, rollout, step * G + i)
                          for i in range(G)])
        actions = np.concatenate([a for a, _ in shards], axis=0)     # [total, rollout, 7]
        scores = np.concatenate([s for _, s in shards], axis=0)      # [total]
        idx = np.argsort(scores)[:topk]
        sel = actions[idx]                                           # [topk, rollout, 7]
        sel4 = np.concatenate([sel[..., :3], sel[..., 6:7]], axis=-1)  # [topk, rollout, 4]
        mean = sel4.mean(0) * (1 - momentum) + mean * momentum
        std = sel4.std(0) * (1 - momentum) + std * momentum

    action = np.concatenate([mean[:, :3], np.zeros((rollout, 3), np.float32),
                             np.where(np.abs(mean[:, 3:4]) < 0.25, 0.0, mean[:, 3:4])], axis=-1)
    return action[None]  # [1, rollout, 7]

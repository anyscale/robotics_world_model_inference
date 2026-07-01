"""
world.py — the World oracle: a stateful Ray actor standing in for the robot's physical reality.
It holds the TRUE (latent, pose) and advances it with the V-JEPA predictor given the action the
robot actually executes (plus any unmodeled disturbance). The cloud slow-brain and the robot never
see this internal state except via snapshot()/step() — exactly like a real robot + sensors.
"""
import numpy as np
import ray

from vjepa_core import WorldModelCore


@ray.remote(num_gpus=1)
class World:
    def __init__(self):
        self.core = WorldModelCore(device="cuda")
        self.z = self.pose = self.z_goal = self.p_goal = None
        print("[World] oracle ready", flush=True)

    def build_problem(self, ctx_img, goal_img, pose_np, goal_steps=8, a_star=None):
        """Encode context, build a K-step-reachable goal by rolling the predictor; set initial state."""
        z0 = self.core.encode(ctx_img)
        pose = self.core.t(pose_np).reshape(1, 1, 7)
        if a_star is None:
            a_star = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        a = self.core.t(a_star).reshape(1, 1, 7)
        z, p = z0, pose
        for _ in range(goal_steps):
            z, p = self.core.step(z, p, a)
        self.z, self.pose, self.z_goal, self.p_goal = z0, pose, z, p
        return {"z0": z0.float().cpu().numpy(), "p0": pose.float().cpu().numpy(),
                "z_goal": z.float().cpu().numpy(), "p_goal": p.float().cpu().numpy()}

    def reset(self, z0_np, pose0_np):
        self.z, self.pose = self.core.t(z0_np), self.core.t(pose0_np)
        return True

    def step(self, action_np, drift_np=None):
        """Advance the true state by the executed action (+ optional unmodeled disturbance)."""
        a = self.core.t(action_np).reshape(1, 1, 7)
        if drift_np is not None:
            a = a + self.core.t(drift_np).reshape(1, 1, 7)
        self.z, self.pose = self.core.step(self.z, self.pose, a)
        return {"pose": self.pose.float().cpu().numpy(),
                "energy": self.core.energy(self.z, self.z_goal),
                "pose_err": float(np.linalg.norm((self.pose[..., :3] - self.p_goal[..., :3]).cpu().numpy())),
                # pooled latent summary the on-robot reflex consumes (its "perception")
                "z_pool": self.z[:, -1].mean(1).flatten().float().cpu().numpy()}

    def snapshot(self):
        """What the robot ships to the cloud: current latent + pose (kept on GPU until asked)."""
        return {"z": self.z.float().cpu().numpy(), "pose": self.pose.float().cpu().numpy()}

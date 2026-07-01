"""reflex_policy.py — the distilled FAST REFLEX at runtime (the on-robot fast brain).

Loads reflex_policy.npz (a tiny MLP + its feature transform, distilled from the cloud's world-model
search by distill_reflex.py) and maps a latent state summary + goal latent to an action in ONE forward
pass. Pure numpy — no search, no GPU for the policy itself: given the latent the robot already
perceives (via its snapshot), the reflex is a few small matmuls.

It is deliberately one-shot and reactive (no lookahead), so it is weaker than the cloud's multi-step
CEM search — the reactive-vs-deliberative split. It is conditioned on the LATENT, not a goal pose, so
it cannot cheat by beelining to a known target.
"""
import numpy as np


class ReflexPolicy:
    def __init__(self, path):
        d = np.load(path)
        self.mu, self.sd, self.P = d["mu"], d["sd"], d["P"]
        self.w0, self.b0, self.w2, self.b2 = d["w0"], d["b0"], d["w2"], d["b2"]

    @staticmethod
    def pool(z):
        """[1,1,tokens,D] (numpy) -> mean over tokens -> [D]."""
        z = np.asarray(z)
        return z[:, -1].mean(1).flatten()

    def action_from_pooled(self, pooled_now, pooled_goal):
        """pooled_now/pooled_goal: [D] each. Returns a 7-DoF action (numpy)."""
        f = (np.concatenate([pooled_now, pooled_goal]) - self.mu) / self.sd
        x = f @ self.P.T                                  # -> [10] PCA feature
        h = np.maximum(0.0, x @ self.w0.T + self.b0)      # ReLU hidden
        return (h @ self.w2.T + self.b2).astype(np.float32)  # [7]

    def action(self, z_now, z_goal):
        """Full-latent convenience: pool then predict."""
        return self.action_from_pooled(self.pool(z_now), self.pool(z_goal))

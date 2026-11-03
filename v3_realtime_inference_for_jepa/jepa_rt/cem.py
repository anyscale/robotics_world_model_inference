"""Cross-Entropy Method over action sequences, split into the three pieces a distributed planner needs.

    dist = CEMDistribution(horizon=2)
    for _ in range(rounds):                      # sequential: each round refits on the previous round's winners
        cands = dist.sample(n=800)               # [800, H, 7]   cheap, on the planner (CPU)
        energy = score(cands)                    # [800]         expensive: the world model -- fan this out
        dist.refit(cands, energy)                # keep the best 80, move the Gaussian towards them
    action = dist.action()                       # first action of the plan

The sampling, clipping, momentum and final-action rules follow Meta's reference `cem()` in
facebookresearch/vjepa2 `notebooks/utils/mpc_utils.py`; only the *score* call is pulled out so it can run anywhere.
Actions are 7-D end-effector deltas (dx, dy, dz, droll, dpitch, dyaw, dgripper); like the reference, CEM searches
over xyz + gripper and keeps rotation deltas at zero.
"""
import numpy as np


class CEMDistribution:
    def __init__(self, horizon=2, maxnorm=0.05, momentum_mean=0.15, momentum_std=0.15,
                 momentum_mean_gripper=0.15, momentum_std_gripper=0.15, topk_frac=0.1, seed=None):
        self.H, self.maxnorm, self.topk_frac = horizon, maxnorm, topk_frac
        self.mm, self.ms, self.mmg, self.msg = momentum_mean, momentum_std, momentum_mean_gripper, momentum_std_gripper
        self.rng = np.random.default_rng(seed)
        # Gaussian over [H, 4] = (dx, dy, dz, dgripper) per step: zero mean, std = maxnorm for xyz, 1 for gripper
        self.mean = np.zeros((horizon, 4), np.float32)
        self.std = np.concatenate([np.full((horizon, 3), maxnorm), np.ones((horizon, 1))], 1).astype(np.float32)

    def sample(self, n):
        """n candidate action sequences [n, H, 7] (float32)."""
        a = self.rng.standard_normal((n, self.H, 4)).astype(np.float32) * self.std + self.mean
        a[..., :3] = np.clip(a[..., :3], -self.maxnorm, self.maxnorm)
        a[..., 3:] = np.clip(a[..., 3:], -0.75, 0.75)
        out = np.zeros((n, self.H, 7), np.float32)
        out[..., :3], out[..., 6:] = a[..., :3], a[..., 3:]
        return out

    def refit(self, cands, energy, topk=None):
        """Move the Gaussian towards the lowest-energy candidates (momentum-smoothed, as in the reference)."""
        k = topk or max(1, int(round(len(cands) * self.topk_frac)))
        elite = cands[np.argsort(energy)[:k]]
        e = np.concatenate([elite[..., :3], elite[..., 6:]], -1)      # [k, H, 4]
        m, s = e.mean(0), e.std(0, ddof=1)                            # torch .std() is the unbiased estimator
        self.mean[:, :3] = m[:, :3] * (1 - self.mm) + self.mean[:, :3] * self.mm
        self.mean[:, 3:] = m[:, 3:] * (1 - self.mmg) + self.mean[:, 3:] * self.mmg
        self.std[:, :3] = s[:, :3] * (1 - self.ms) + self.std[:, :3] * self.ms
        self.std[:, 3:] = s[:, 3:] * (1 - self.msg) + self.std[:, 3:] * self.msg

    def plan(self):
        """The whole planned sequence [H, 7]: the mean, gripper deltas below 0.25 rounded to 0 (as the reference)."""
        p = np.zeros((self.H, 7), np.float32)
        p[:, :3] = self.mean[:, :3]
        g = self.mean[:, 3]
        p[:, 6] = np.where(np.abs(g) < 0.25, 0.0, g)
        return p

    def action(self):
        """The action to execute now: the first step of the plan [7]."""
        return self.plan()[0]

    def shift(self, std_reset=None):
        """Warm start for the next control step: drop the executed first step, repeat the last one.

        The kept steps carry their (already-shrunk) std; the new tail step restarts broad so the search can still
        explore. `std_reset` scales the whole std back up a little (None = leave it).
        """
        self.mean = np.concatenate([self.mean[1:], self.mean[-1:]], 0)
        tail = np.concatenate([np.full((1, 3), self.maxnorm), np.ones((1, 1))], 1).astype(np.float32)
        self.std = np.concatenate([self.std[1:], tail], 0)
        if std_reset is not None:
            self.std = np.maximum(self.std, std_reset * np.concatenate(
                [np.full((self.H, 3), self.maxnorm), np.ones((self.H, 1))], 1).astype(np.float32))

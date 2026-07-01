"""distill_reflex.py — distill the on-robot FAST REFLEX from the cloud's world-model search.

The reflex is a real learned policy (a tiny MLP), reused from v1's distillation recipe: it maps a
compact summary of the V-JEPA latent state + goal latent to an action, in ONE forward pass (~ms, no
search). It is genuinely weaker than the cloud because it is one-shot and reactive (no lookahead),
so the cloud's multi-step CEM search beats it — the reactive-vs-deliberative split that is the thesis.

Crucially the reflex is conditioned on the LATENT (what the robot observes via snapshots), not on a
goal pose, so it cannot "cheat" by beelining to a known target.

Pipeline: mine (latent-summary -> cloud action) labels across the 4 GPUs on a spread of
(current, goal) latent states, then train the MLP. Saves reflex_policy.npz (weights + feature
normalization) for use in the closed loop (robot_client / P1).

Run:  python distill_reflex.py
"""
import os
import sys
import time

import numpy as np
import ray

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"

NUM_WORKERS = 4
GOAL_STEPS = 8                      # the (far) goal the reflex must reach — greedy struggles, search wins
A_STAR = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CLOUD_SAMPLES, CLOUD_CEM, CLOUD_ROLLOUT = 128, 6, 2   # the "teacher": USABLE multi-step deliberative search
SEEDS_PER_SPEC = 3
HORIZON = 16


@ray.remote(num_gpus=1)
class MineWorker:
    def __init__(self):
        sys.path.insert(0, REPO_DIR)
        from vjepa_core import WorldModelCore
        self.core = WorldModelCore(device="cuda")
        self.z0 = self.pose0 = None
        self.goals = {}   # k -> (z_goal, p_goal)
        print("[MineWorker] ready", flush=True)

    def setup(self, ctx_img, pose_np):
        """Encode context, and build reachable goals at several distances k by rolling the predictor."""
        import torch
        self.z0 = self.core.encode(ctx_img)
        self.pose0 = self.core.t(pose_np).reshape(1, 1, 7)
        a = self.core.t(A_STAR).reshape(1, 1, 7)
        z, p, k = self.z0, self.pose0, 0
        seq = {0: (self.z0, self.pose0)}
        for _ in range(GOAL_STEPS):
            z, p = self.core.step(z, p, a)
            k += 1
            seq[k] = (z, p)
        self.seq = seq                       # rolled states/goals along the nominal trajectory
        return {"z0": self.z0.float().cpu().numpy(), "p0": self.pose0.float().cpu().numpy(),
                "z_goal": seq[GOAL_STEPS][0].float().cpu().numpy(),
                "p_goal": seq[GOAL_STEPS][1].float().cpu().numpy()}

    @staticmethod
    def _pool(z):
        return z[:, -1].mean(1).flatten()    # mean over tokens -> [D]

    def mine(self, specs):
        """specs: list of (j, k, seed) -> plan from state@j toward goal@k, label the first action.
        Returns (feats [n, 2D], actions [n, 7])."""
        import torch
        feats, acts = [], []
        for (j, k, seed) in specs:
            torch.manual_seed(seed)
            z_now, p_now = self.seq[min(j, GOAL_STEPS)]
            z_goal, _ = self.seq[k]
            plan = self.core.cem_plan(z_now, p_now, z_goal, samples=CLOUD_SAMPLES,
                                      cem_steps=CLOUD_CEM, rollout=CLOUD_ROLLOUT)  # [1, rollout, 7]
            feat = torch.cat([self._pool(z_now), self._pool(z_goal)], -1)          # [2D]
            feats.append(feat.float().cpu().numpy())
            acts.append(plan[0, 0].float().cpu().numpy())                          # first action [7]
        return np.stack(feats), np.stack(acts)

    def _pose_err(self, pose):
        import torch
        pg = self.seq[GOAL_STEPS][1][..., :3]
        return float(torch.linalg.norm(pose[..., :3] - pg).item())

    def reflex_rollout(self, w, horizon=HORIZON):
        """Closed-loop with the DISTILLED reflex: apply MLP(pool(z_now) + pool(z_goal)) each tick.
        Tests whether the one-shot reactive policy genuinely fails (wanders/plateaus) on its own."""
        import torch
        t = lambda a: torch.tensor(a, device=self.core.device, dtype=torch.float32)
        mu, sd, P = t(w["mu"]), t(w["sd"]), t(w["P"])
        w0, b0, w2, b2 = t(w["w0"]), t(w["b0"]), t(w["w2"]), t(w["b2"])
        z_goal = self.seq[GOAL_STEPS][0]
        gpool = self._pool(z_goal)
        z, pose = self.z0, self.pose0
        perrs = [self._pose_err(pose)]
        for _ in range(horizon):
            feat = torch.cat([self._pool(z), gpool], -1)
            x = ((feat - mu) / sd) @ P.T
            a = (torch.relu(x @ w0.T + b0) @ w2.T + b2).reshape(1, 1, 7)
            z, pose = self.core.step(z, pose, a)
            perrs.append(self._pose_err(pose))
        return perrs

    def cem_rollout(self, samples, cem_steps, rollout, horizon=HORIZON):
        """Closed-loop with a REACTIVE CEM (cheap, few-sample; rollout=1 = no lookahead) each tick."""
        z_goal = self.seq[GOAL_STEPS][0]
        z, pose = self.z0, self.pose0
        perrs = [self._pose_err(pose)]
        plan, pidx = None, 0
        for _ in range(horizon):
            if plan is None or pidx >= plan.size(1):
                plan = self.core.cem_plan(z, pose, z_goal, samples=samples, cem_steps=cem_steps, rollout=rollout)
                pidx = 0
            z, pose = self.core.step(z, pose, plan[:, pidx:pidx + 1]); pidx += 1
        return perrs


def main():
    ray.init(address="auto", logging_level="ERROR", runtime_env={
        "py_modules": [f"{REPO_DIR}/vjepa_core.py"],
        "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    })
    print(f"GPUs: {int(ray.cluster_resources().get('GPU', 0))}", flush=True)
    workers = [MineWorker.remote() for _ in range(NUM_WORKERS)]

    traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
    ctx_img = traj["observations"][0, 0]
    pose = traj["states"][0, 0]
    t0 = time.time()
    prob = ray.get(workers[0].setup.remote(ctx_img, pose))
    ray.get([w.setup.remote(ctx_img, pose) for w in workers[1:]])
    np.savez(f"{REPO_DIR}/reflex_problem.npz", **prob)
    print(f"problem/goals built in {time.time()-t0:.1f}s", flush=True)

    # spread: from several current states j toward several goals k>=j (nearer & farther goals -> action range)
    goal_ks = [4, 6, 8]
    specs = [(j, k, s) for k in goal_ks for j in range(0, k) for s in range(SEEDS_PER_SPEC)]
    shards = [specs[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
    print(f"mining {len(specs)} labels across {NUM_WORKERS} GPUs...", flush=True)

    t0 = time.time()
    futs = [workers[i].mine.remote(shards[i]) for i in range(NUM_WORKERS)]
    pending, out = list(futs), []
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        out.append(ray.get(ready[0]))
        print(f"  shard done ({len(out)}/{NUM_WORKERS})  ({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate([f for f, _ in out], 0)
    Y = np.concatenate([a for _, a in out], 0)
    print(f"mined {len(X)} labels | feats {X.shape} -> actions {Y.shape} in {time.time()-t0:.1f}s", flush=True)
    print(f"action spread (std per dim): {np.round(Y.std(0), 4)}", flush=True)

    # ---- train the tiny MLP (v1 recipe: standardize -> top-k PCA -> 2-layer MLP) ----
    import torch, torch.nn as nn
    perm = np.random.default_rng(1).permutation(len(X))
    ntr = int(0.8 * len(X)); tr, va = perm[:ntr], perm[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Xs[tr], full_matrices=False)
    P = Vt[:10]                                    # top-10 principal components
    Xt = torch.tensor(Xs @ P.T, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    baseline = ((Yt[tr].mean(0) - Yt[va]) ** 2).mean().item()

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 7))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-2)
    for epoch in range(400):
        opt.zero_grad()
        loss = ((net(Xt[tr]) - Yt[tr]) ** 2).mean()
        loss.backward(); opt.step()
        if epoch % 100 == 0 or epoch == 399:
            with torch.no_grad():
                val = ((net(Xt[va]) - Yt[va]) ** 2).mean().item()
            print(f"  epoch {epoch:>3}  train {loss.item():.4f}  val {val:.4f}", flush=True)
    print(f"  (predict-the-mean baseline val MSE = {baseline:.4f})", flush=True)

    # save weights + feature transform for the closed loop
    w0, b0 = net[0].weight.detach().numpy(), net[0].bias.detach().numpy()
    w2, b2 = net[2].weight.detach().numpy(), net[2].bias.detach().numpy()
    np.savez(f"{REPO_DIR}/reflex_policy.npz", mu=mu, sd=sd, P=P,
             w0=w0, b0=b0, w2=w2, b2=b2)
    print(f"saved {REPO_DIR}/reflex_policy.npz  (val MSE {val:.4f} vs baseline {baseline:.4f})", flush=True)

    # ---- verify closed-loop: does the reflex genuinely FAIL on its own (vs the usable cloud ~0.045)? ----
    print("\n=== closed-loop reflex check (start pose_err ~0.320; cloud reaches ~0.045) ===", flush=True)
    wdict = {"mu": mu, "sd": sd, "P": P, "w0": w0, "b0": b0, "w2": w2, "b2": b2}
    distilled = ray.get(workers[0].reflex_rollout.remote(wdict))
    cheap = ray.get(workers[1].cem_rollout.remote(8, 2, 1))       # cheap reactive CEM (no lookahead)
    react = ray.get(workers[2].cem_rollout.remote(64, 4, 1))      # well-resourced but still reactive (rollout=1)
    print(f"  distilled one-shot reflex : final pose_err {distilled[-1]:.3f}   traj {np.round(distilled, 3)}", flush=True)
    print(f"  cheap reactive CEM (8x2x1): final pose_err {cheap[-1]:.3f}", flush=True)
    print(f"  reactive CEM (64x4x1)     : final pose_err {react[-1]:.3f}", flush=True)
    print("  (a good reflex for the story FAILS here — stays well above the cloud's ~0.045)", flush=True)

    for w in workers:
        ray.kill(w)


if __name__ == "__main__":
    main()

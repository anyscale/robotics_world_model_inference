"""verify_reflex.py — quick closed-loop check of the distilled reflex WITH proper action clipping
(actions bounded to the valid range, like any real controller). Loads the saved policy + problem,
runs the reflex on its own, and reports the pose-error trajectory. A good reflex for the story stays
mediocre-but-stable (~0.2) — clearly worse than the cloud (~0.045) but not exploding.
Run: python verify_reflex.py
"""
import sys
import numpy as np
import ray

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
MAXNORM = 0.05
HORIZON = 16


@ray.remote(num_gpus=1)
class Verifier:
    def __init__(self):
        sys.path.insert(0, REPO_DIR)
        from vjepa_core import WorldModelCore
        self.core = WorldModelCore(device="cuda")
        print("[Verifier] ready", flush=True)

    def run(self, prob, w, maxnorm=MAXNORM, horizon=HORIZON):
        import torch
        t = lambda a: torch.tensor(np.asarray(a), device=self.core.device, dtype=torch.float32)
        z = t(prob["z0"]); pose = t(prob["p0"]); z_goal = t(prob["z_goal"])
        pg = t(prob["p_goal"])[..., :3]
        mu, sd, P = t(w["mu"]), t(w["sd"]), t(w["P"])
        w0, b0, w2, b2 = t(w["w0"]), t(w["b0"]), t(w["w2"]), t(w["b2"])

        def pool(zz):
            return zz[:, -1].mean(1).flatten()

        def pe():
            return float(torch.linalg.norm(pose[..., :3] - pg).item())

        gpool = pool(z_goal)
        perrs = [pe()]
        for _ in range(horizon):
            feat = torch.cat([pool(z), gpool], -1)
            x = ((feat - mu) / sd) @ P.T
            a = (torch.relu(x @ w0.T + b0) @ w2.T + b2)          # [7]
            a = a.clone()
            a[:3] = torch.clip(a[:3], -maxnorm, maxnorm)         # bound the action, like a real controller
            a[3:6] = 0.0
            a[6] = torch.clip(a[6], -maxnorm, maxnorm)
            z, pose = self.core.step(z, pose, a.reshape(1, 1, 7))
            perrs.append(pe())
        return perrs


def main():
    ray.init(address="auto", logging_level="ERROR", runtime_env={
        "py_modules": [f"{REPO_DIR}/vjepa_core.py"],
        "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    })
    prob = dict(np.load(f"{REPO_DIR}/reflex_problem.npz"))
    w = dict(np.load(f"{REPO_DIR}/reflex_policy.npz"))
    v = Verifier.remote()
    perrs = ray.get(v.run.remote(prob, w))
    print(f"start pose_err {perrs[0]:.3f} -> final {perrs[-1]:.3f}  (cloud reaches ~0.045)", flush=True)
    print(f"trajectory (clipped distilled reflex): {np.round(perrs, 3)}", flush=True)
    ray.kill(v)


if __name__ == "__main__":
    main()

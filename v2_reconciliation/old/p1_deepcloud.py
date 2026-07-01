"""p1_deepcloud.py — linchpin test: does a deeper multi-step CEM plan drive energy near 0?
Compares rollout depth (greedy 1 vs multi-step) for the +x goal, fanned across workers."""
import numpy as np
import ray

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"

ray.init(address="auto", logging_level="ERROR", runtime_env={
    "py_modules": [f"{REPO_DIR}/p1_world.py"],
    "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
})
from p1_world import SpikeWorker

# (rollout, samples) — shrink samples as the horizon (and memory) grows on the 24GB L4.
CONFIGS = [(1, 128), (2, 96), (3, 64), (4, 48)]
workers = [SpikeWorker.remote() for _ in range(min(4, len(CONFIGS)))]

traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
ctx_img, goal_img = traj["observations"][0, 0], traj["observations"][0, 1]
pose = traj["states"][0, 0]
a_star = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

prob = ray.get(workers[0].build_problem.remote(ctx_img, goal_img, pose, goal_steps=8, a_star=a_star))
z0, p0, zg = prob["z0"], prob["p0"], prob["z_goal"]
print(f"goal is 8 steps of +x away; start energy ~0.50, oracle reaches 0.0\n")

futs = [workers[i % len(workers)].deep_cem_test.remote(z0, p0, zg, rollout=r, samples=s, cem_steps=10)
        for i, (r, s) in enumerate(CONFIGS)]
for r in ray.get(futs):
    if r.get("error"):
        print(f"rollout={r['rollout']:>2} samples={r['samples']:>3}  -> {r['error']}\n")
        continue
    print(f"rollout={r['rollout']:>2} samples={r['samples']:>3}  final_energy={r['energy'][-1]:.4f}  "
          f"peak={r['peak_gb']:.1f}GB  trace={np.round(r['energy'], 3)}")
    print(f"           plan xyz (should trend +x):\n{np.round(r['plan_xyz'], 3)}\n")

for w in workers:
    ray.kill(w)

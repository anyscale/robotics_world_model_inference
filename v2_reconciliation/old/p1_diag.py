"""p1_diag.py — substrate diagnostic. Does latent energy track pose, and does CEM track the oracle?"""
import numpy as np
import ray

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"

ray.init(address="auto", logging_level="ERROR", runtime_env={
    "py_modules": [f"{REPO_DIR}/p1_world.py"],
    "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache"},
})
from p1_world import SpikeWorker

w = SpikeWorker.remote()
traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
ctx_img, goal_img = traj["observations"][0, 0], traj["observations"][0, 1]
pose = traj["states"][0, 0]

for a_star, tag in [([0.04, 0, 0, 0, 0, 0, 0], "+x"), ([0, 0.04, 0, 0, 0, 0, 0], "+y"),
                    ([0, 0, 0.04, 0, 0, 0, 0], "+z")]:
    prob = ray.get(w.build_problem.remote(ctx_img, goal_img, pose, goal_steps=8, a_star=a_star))
    d = ray.get(w.diagnose.remote(prob["z0"], prob["p0"], prob["z_goal"], a_star=a_star, steps=8))
    print(f"\n=== goal built with {tag} action ===")
    print(f"  oracle replay energy: {np.round(d['e_oracle'], 4)}")
    print(f"  greedy CEM   energy: {np.round(d['e_cem'], 4)}")
    print(f"  CEM xyz picks (should point {tag}):\n{np.round(d['cem_xyz'], 3)}")

ray.kill(w)

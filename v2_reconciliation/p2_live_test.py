"""p2_live_test.py — quick end-to-end check of the live loop: distilled reflex (fast brain) + sharded
cloud over real HTTP + vector_add reconciliation. Confirms robot_client runs and the robot makes
progress. Run: python p2_live_test.py
"""
import sys
import time

import numpy as np
import requests
import ray
from ray import serve

REPO = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"
URL = "http://127.0.0.1:8000"


def main():
    sys.path.insert(0, REPO)
    ray.init(address="auto", logging_level="ERROR", log_to_driver=False, runtime_env={
        "py_modules": [f"{REPO}/vjepa_core.py", f"{REPO}/slow_brain_service.py", f"{REPO}/world.py",
                       f"{REPO}/sharded_planner.py", f"{REPO}/robot_client.py", f"{REPO}/reflex_policy.py"],
        "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    })
    from slow_brain_service import build_sharded_app
    from world import World
    from robot_client import run_episode
    from reflex_policy import ReflexPolicy

    serve.run(build_sharded_app(n_shards=3, total_samples=128, cem_steps=6, rollout=2),
              name="sharded", route_prefix="/")
    print("health:", requests.get(URL + "/health", timeout=180).json(), flush=True)

    world = World.remote()
    traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
    ctx_img, goal_img = traj["observations"][0, 0], traj["observations"][0, 1]
    prob = ray.get(world.build_problem.remote(ctx_img, goal_img, traj["states"][0, 0], goal_steps=8))
    z_goal = prob["z_goal"]
    reflex = ReflexPolicy(f"{REPO}/reflex_policy.npz")
    DRIFT = [0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]

    ray.get(world.reset.remote(prob["z0"], prob["p0"]))
    print("running live vector_add episode (reflex + sharded cloud)...", flush=True)
    res = run_episode(world, URL, z_goal, strategy="vector_add", reflex=reflex,
                      control_hz=2.0, horizon=20, drift=DRIFT)
    print(f"LIVE vector_add: final pose_err {res['final_pose_err']:.3f}  "
          f"measured Δ≈{res['mean_delta']:.0f} ticks  latency≈{res['mean_latency_ms']:.0f}ms", flush=True)
    print(f"pose_err trajectory: {np.round(res['pose_err'], 3)}", flush=True)

    serve.shutdown(); ray.kill(world)
    print("done.", flush=True)


if __name__ == "__main__":
    main()

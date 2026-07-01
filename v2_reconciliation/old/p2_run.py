"""
p2_run.py — the P2 architecture demo: deploy the slow brain on Ray Serve, bring up the World oracle,
and run the robot's reconciliation loop against the REAL served endpoint. Compares naive vs vector_add
at a moderate control rate (~3 Hz) where the real ~2.8 s cloud latency lands at a small, clean Δ.
Saves p2_reconciliation.png and prints the measured Δ / latency.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import requests
from ray import serve

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"
sys.path.insert(0, REPO_DIR)

CONTROL_HZ = 3.0          # deliberate-task rate -> 2.8s cloud latency ~= Δ 8 ticks
CLOUD_EVERY = 6
HORIZON = 28
DRIFT = [0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]


def main():
    ray.init(address="auto", logging_level="ERROR", runtime_env={
        "py_modules": [f"{REPO_DIR}/vjepa_core.py", f"{REPO_DIR}/slow_brain_service.py",
                       f"{REPO_DIR}/world.py", f"{REPO_DIR}/robot_client.py"],
        "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    })
    from slow_brain_service import build_app
    from world import World
    from robot_client import run_episode

    print("deploying SlowBrain on Ray Serve...", flush=True)
    serve.run(build_app(samples=32, cem_steps=4, rollout=2), name="slow_brain", route_prefix="/")
    URL = "http://127.0.0.1:8000"
    print("health:", requests.get(URL + "/health", timeout=120).json(), flush=True)

    print("World oracle + problem...", flush=True)
    world = World.remote()
    traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
    prob = ray.get(world.build_problem.remote(traj["observations"][0, 0], traj["observations"][0, 1],
                                              traj["states"][0, 0], goal_steps=8))
    z_goal = prob["z_goal"]

    results = {}
    for strategy in ["naive", "vector_add"]:
        ray.get(world.reset.remote(prob["z0"], prob["p0"]))
        print(f"\n--- episode: {strategy} (control {CONTROL_HZ}Hz, one request in flight) ---", flush=True)
        t0 = time.time()
        results[strategy] = run_episode(world, URL, z_goal, strategy=strategy, control_hz=CONTROL_HZ,
                                        cloud_every_ticks=CLOUD_EVERY, horizon=HORIZON, drift=DRIFT,
                                        samples=32, cem_steps=4, rollout=2, pool=None)  # fresh pool/episode
        r = results[strategy]
        print(f"    -> final pose_err={r['final_pose_err']:.4f}  mean Δ={r['mean_delta']:.1f} ticks  "
              f"replies={len(r['deltas'])}  mean latency={r['mean_latency_ms']:.0f}ms  ({time.time()-t0:.1f}s wall)", flush=True)

    # plot pose-error over the episode for both strategies
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.2))
    plt.plot(results["naive"]["pose_err"], "s--", color="C3", lw=2, label="naive (apply stale)")
    plt.plot(results["vector_add"]["pose_err"], "o-", color="C0", lw=2, label="vector-add reconciled")
    d = results["vector_add"]["mean_delta"]; lat = results["vector_add"]["mean_latency_ms"]
    plt.xlabel("control tick"); plt.ylabel("pose error to goal (↓)")
    plt.title(f"Reconciliation over real Ray Serve (Δ≈{d:.0f} ticks from {lat:.0f}ms real latency)")
    plt.legend(); plt.tight_layout()
    out = f"{REPO_DIR}/p2_reconciliation.png"
    plt.savefig(out, dpi=110)
    print(f"\nsaved {out}", flush=True)
    print(f"VERDICT: naive final={results['naive']['final_pose_err']:.4f}  "
          f"vector_add final={results['vector_add']['final_pose_err']:.4f}  "
          f"(vector_add better by {results['naive']['final_pose_err']-results['vector_add']['final_pose_err']:+.4f})", flush=True)

    serve.shutdown()
    ray.kill(world)


if __name__ == "__main__":
    main()

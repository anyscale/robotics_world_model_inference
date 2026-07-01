"""p2_sharded_serve_test.py — validate the SHARDED slow brain served over HTTP:
  (1) latency of one plan (should be ~4-6 s across 3 shards, vs ~18 s single-GPU),
  (2) usability: a cloud-only closed-loop MPC through the endpoint drives pose error toward the goal
      (confirms sharding didn't change plan quality — semantics match the single-process CEM).
GPU budget: 3 shards + 1 World oracle = 4 L4s. Run: python p2_sharded_serve_test.py
"""
import pickle
import time

import numpy as np
import requests
import ray
from ray import serve

REPO = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"
URL = "http://127.0.0.1:8000"


def post(req, timeout=120):
    r = requests.post(URL + "/plan", data=pickle.dumps(req),
                      headers={"Content-Type": "application/octet-stream"}, timeout=timeout)
    return pickle.loads(r.content)


def main():
    ray.init(address="auto", logging_level="ERROR", log_to_driver=True, runtime_env={
        "py_modules": [f"{REPO}/vjepa_core.py", f"{REPO}/slow_brain_service.py",
                       f"{REPO}/world.py", f"{REPO}/sharded_planner.py"],
        "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    })
    from slow_brain_service import build_sharded_app
    from world import World

    print("deploying sharded slow brain (3 shards, 128x6x2)...", flush=True)
    serve.run(build_sharded_app(n_shards=3, total_samples=128, cem_steps=6, rollout=2),
              name="sharded", route_prefix="/")
    print("health:", requests.get(URL + "/health", timeout=180).json(), flush=True)

    world = World.remote()
    traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
    ctx_img, goal_img = traj["observations"][0, 0], traj["observations"][0, 1]
    prob = ray.get(world.build_problem.remote(ctx_img, goal_img, traj["states"][0, 0], goal_steps=8))
    z_goal = prob["z_goal"]
    start_err = float(np.linalg.norm((prob["p0"][..., :3] - prob["p_goal"][..., :3])))
    print(f"problem built, start pose_err {start_err:.3f}", flush=True)

    # (1) one plan latency
    ray.get(world.reset.remote(prob["z0"], prob["p0"]))
    snap = ray.get(world.snapshot.remote())
    t0 = time.time()
    out = post({"z": snap["z"], "pose": snap["pose"], "z_goal": z_goal, "seq_id": 0})
    rtt = (time.time() - t0) * 1000
    print(f"SHARDED plan: chunk {out['chunk'].shape} | server {out['compute_ms']:.0f} ms | "
          f"round-trip {rtt:.0f} ms | shards {out['n_shards']}", flush=True)

    # (2) usability: cloud-only closed-loop MPC through the endpoint
    ray.get(world.reset.remote(prob["z0"], prob["p0"]))
    perrs, res = [start_err], None
    for t in range(0, 16, 2):
        snap = ray.get(world.snapshot.remote())
        out = post({"z": snap["z"], "pose": snap["pose"], "z_goal": z_goal, "seq_id": t})
        chunk = out["chunk"]                       # [rollout, 7]
        for k in range(chunk.shape[0]):
            res = ray.get(world.step.remote(chunk[k:k + 1].reshape(1, 1, 7)))
        perrs.append(res["pose_err"])
        print(f"  tick {t:>2}: pose_err {res['pose_err']:.3f}", flush=True)
    print(f"SHARDED cloud closed-loop: start {start_err:.3f} -> final {perrs[-1]:.3f}  "
          f"(usable if well below ~0.6 reflex)", flush=True)
    print(f"trajectory: {np.round(perrs, 3)}", flush=True)

    serve.shutdown()
    ray.kill(world)
    print("done.", flush=True)


if __name__ == "__main__":
    main()

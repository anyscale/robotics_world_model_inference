"""p2_serve_test.py — validate the serving path: deploy SlowBrain on Ray Serve, bring up the World
oracle, build the problem, and POST a few real snapshots over HTTP. Confirms the round-trip works and
reports REAL latency (client round-trip vs. server compute)."""
import pickle
import sys
import time

import numpy as np
import ray
import requests
from ray import serve

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"
sys.path.insert(0, REPO_DIR)

ray.init(address="auto", logging_level="ERROR", runtime_env={
    "py_modules": [f"{REPO_DIR}/vjepa_core.py", f"{REPO_DIR}/slow_brain_service.py", f"{REPO_DIR}/world.py"],
    "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
})
from slow_brain_service import build_app
from world import World

print("deploying SlowBrain on Ray Serve...", flush=True)
serve.run(build_app(samples=32, cem_steps=4, rollout=2), name="slow_brain", route_prefix="/")
URL = "http://127.0.0.1:8000"
print("health:", requests.get(URL + "/health", timeout=120).json(), flush=True)

print("bringing up World oracle + building problem...", flush=True)
world = World.remote()
traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
prob = ray.get(world.build_problem.remote(traj["observations"][0, 0], traj["observations"][0, 1],
                                          traj["states"][0, 0], goal_steps=8))
z_goal = prob["z_goal"]

print("\nposting 3 real snapshots over HTTP:", flush=True)
ray.get(world.reset.remote(prob["z0"], prob["p0"]))
for i in range(3):
    snap = ray.get(world.snapshot.remote())
    req = {"z": snap["z"], "pose": snap["pose"], "z_goal": z_goal, "seq_id": i}
    t0 = time.time()
    r = requests.post(URL + "/plan", data=pickle.dumps(req),
                      headers={"Content-Type": "application/octet-stream"}, timeout=300)
    rtt = (time.time() - t0) * 1000
    out = pickle.loads(r.content)
    print(f"  req {i}: chunk{out['chunk'].shape} xyz0={np.round(out['chunk'][0,:3],3)} "
          f"server_compute={out['compute_ms']:.0f}ms  round_trip={rtt:.0f}ms", flush=True)
    # advance the world one step with the cloud's first action, so the next snapshot differs
    ray.get(world.step.remote(out["chunk"][0]))

print("\nserving path OK.", flush=True)
serve.shutdown()
ray.kill(world)

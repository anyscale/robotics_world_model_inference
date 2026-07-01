"""p2_shard_test.py — prove the data-parallel CEM lever: measure ONE plan's latency as we shard its
fixed sample budget across G = 1, 2, 4 L4s. Expect latency to drop ~linearly (more GPUs -> faster plan
-> smaller staleness Δ). Live-logged."""
import sys
import time

import numpy as np
import ray

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"
sys.path.insert(0, REPO_DIR)

TOTAL_SAMPLES, CEM_STEPS, ROLLOUT = 128, 6, 2
REPS = 3  # timed repetitions per G (after a warmup)

ray.init(address="auto", logging_level="ERROR", runtime_env={
    "py_modules": [f"{REPO_DIR}/vjepa_core.py", f"{REPO_DIR}/sharded_planner.py"],
    "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
})
from sharded_planner import ShardWorker, plan_sharded

n_gpus = int(ray.cluster_resources().get("GPU", 0))
print(f"GPUs: {n_gpus} | total_samples={TOTAL_SAMPLES} cem_steps={CEM_STEPS} rollout={ROLLOUT}", flush=True)
workers = [ShardWorker.remote() for _ in range(4)]

traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
print("building problem on worker[0]...", flush=True)
prob = ray.get(workers[0].build_problem.remote(traj["observations"][0, 0], traj["observations"][0, 1],
                                               traj["states"][0, 0], goal_steps=8))
z0, p0, zg = prob["z0"], prob["p0"], prob["z_goal"]

results = {}
for G in [1, 2, 4]:
    ws = workers[:G]
    # warmup (also loads/JITs on each shard)
    plan_sharded(ws, z0, p0, zg, total_samples=TOTAL_SAMPLES, cem_steps=CEM_STEPS, rollout=ROLLOUT)
    times = []
    for r in range(REPS):
        t0 = time.time()
        action = plan_sharded(ws, z0, p0, zg, total_samples=TOTAL_SAMPLES, cem_steps=CEM_STEPS, rollout=ROLLOUT)
        dt = (time.time() - t0) * 1000
        times.append(dt)
        print(f"  G={G} rep{r}: {dt:.0f}ms  (samples/GPU={TOTAL_SAMPLES//G})", flush=True)
    results[G] = float(np.mean(times))
    print(f"G={G}: mean plan latency {results[G]:.0f}ms", flush=True)

print("\n=== DATA-PARALLEL CEM LATENCY vs #GPUs ===", flush=True)
base = results[1]
for G in [1, 2, 4]:
    print(f"  {G} GPU(s): {results[G]:>6.0f}ms   speedup {base/results[G]:.2f}x   (ideal {G}x)", flush=True)

# plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
Gs = [1, 2, 4]
plt.figure(figsize=(6.4, 4.2))
plt.plot(Gs, [base / results[G] for G in Gs], "o-", lw=2, color="C0", label="measured speedup")
plt.plot(Gs, Gs, "k--", lw=1.2, label="ideal linear")
plt.xlabel("# GPUs sharding one CEM plan"); plt.ylabel("plan-latency speedup vs 1 GPU")
plt.title(f"Data-parallel CEM: more GPUs → faster plan → smaller Δ\n({TOTAL_SAMPLES} samples, {CEM_STEPS} CEM steps)")
plt.legend(); plt.tight_layout()
out = f"{REPO_DIR}/images/p2_shard_scaling.png"
plt.savefig(out, dpi=110)
print(f"saved {out}", flush=True)

for w in workers:
    ray.kill(w)

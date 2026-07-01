"""cloud_quality.py — establish that the cloud produces a USABLE answer, and that it scales with
test-time compute. Closed-loop cloud MPC (replan from the true state, execute, repeat) on the +x goal,
sweeping the CEM search budget from cheap to expensive. Reports final pose error per config.

If a big-budget cloud drives pose error ~0 while a cheap one plateaus, we have both a usable cloud
AND the test-time-compute gap that justifies serving it. Run: python cloud_quality.py
"""
import sys
import time

import numpy as np
import ray

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"

GOAL_STEPS = 8
A_STAR = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HORIZON = 16
# (samples, cem_steps, rollout) from cheap -> expensive test-time compute
CONFIGS = [
    (8, 2, 1), (32, 4, 1), (128, 6, 1),
    (32, 4, 2), (128, 6, 2), (256, 8, 2),
    (128, 6, 4), (256, 8, 4),
]


def main():
    ray.init(address="auto", logging_level="ERROR", runtime_env={
        "py_modules": [f"{REPO_DIR}/old/p1_world.py"],
        "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    })
    sys.path.insert(0, f"{REPO_DIR}/old")
    from p1_world import SpikeWorker
    print(f"GPUs: {int(ray.cluster_resources().get('GPU', 0))} | goal {GOAL_STEPS} steps of +x | horizon {HORIZON}", flush=True)

    workers = [SpikeWorker.remote() for _ in range(4)]
    traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
    ctx_img, goal_img = traj["observations"][0, 0], traj["observations"][0, 1]
    pose = traj["states"][0, 0]
    prob = ray.get(workers[0].build_problem.remote(ctx_img, goal_img, pose, goal_steps=GOAL_STEPS, a_star=A_STAR))
    ray.get([w.build_problem.remote(ctx_img, goal_img, pose, goal_steps=GOAL_STEPS, a_star=A_STAR) for w in workers[1:]])
    start_err = float(np.linalg.norm((prob["p0"][..., :3] - prob["p_goal"][..., :3])))
    print(f"start pose error (distance to goal) = {start_err:.3f}\n", flush=True)

    t0 = time.time()
    ref = {}
    futs = []
    for i, (s, c, r) in enumerate(CONFIGS):
        f = workers[i % 4].cloud_mpc.remote(prob["z0"], prob["p0"], prob["z_goal"], prob["p_goal"],
                                            samples=s, cem_steps=c, rollout=r, horizon=HORIZON, seed=0)
        ref[f] = (s, c, r); futs.append(f)

    results = {}
    pending = list(futs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        r = ray.get(ready[0]); cfg = ref[ready[0]]
        results[cfg] = r
        compute = cfg[0] * cfg[1] * cfg[2]
        print(f"  {str(cfg):>14}  compute~{compute:>5}  final_pose_err={r['final_pose_err']:.3f}  "
              f"final_energy={r['final_energy']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== cloud usability vs test-time compute (start err {start_err:.3f}, lower=better) ===", flush=True)
    for cfg in CONFIGS:
        if cfg in results:
            r = results[cfg]
            print(f"  samples={cfg[0]:>3} steps={cfg[1]} rollout={cfg[2]}  ->  pose_err {r['final_pose_err']:.3f}", flush=True)
    best = min(results.items(), key=lambda kv: kv[1]["final_pose_err"])
    print(f"\nBEST cloud config: {best[0]}  -> pose_err {best[1]['final_pose_err']:.3f} "
          f"(from start {start_err:.3f})", flush=True)
    print(f"trajectory: {np.round(best[1]['pose_err'], 3)}", flush=True)
    for w in workers:
        ray.kill(w)


if __name__ == "__main__":
    main()

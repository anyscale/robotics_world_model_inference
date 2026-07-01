"""find_hard_goal.py — the linchpin for Option B: find a goal direction where a REACTIVE (greedy,
1-step) controller genuinely fails but the cloud's DELIBERATIVE (multi-step CEM) search succeeds.

For each candidate goal direction we build a goal 8 steps away, then compare, in latent energy:
  - oracle    : replay the goal-building action (lower bound; should -> ~0)
  - greedy    : 1-step CEM replanned every tick (the reactive reflex's ceiling)
  - deep(r)   : a single multi-step CEM plan, rollout r, executed open-loop (the cloud)
A big (greedy_final - deep_final) gap = reactive fails, deliberative wins = a task where the cloud matters.

Run: python find_hard_goal.py
"""
import sys
import time

import numpy as np
import ray

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"

DIRS = {
    "+x":       [0.04, 0.00, 0.00, 0, 0, 0, 0],
    "+y":       [0.00, 0.04, 0.00, 0, 0, 0, 0],
    "+z":       [0.00, 0.00, 0.04, 0, 0, 0, 0],
    "xy-diag":  [0.03, 0.03, 0.00, 0, 0, 0, 0],
    "xz-diag":  [0.03, 0.00, 0.03, 0, 0, 0, 0],
    "-y":       [0.00, -0.04, 0.00, 0, 0, 0, 0],
}
GOAL_STEPS = 8


def main():
    ray.init(address="auto", logging_level="ERROR", runtime_env={
        "py_modules": [f"{REPO_DIR}/old/p1_world.py"],
        "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    })
    sys.path.insert(0, f"{REPO_DIR}/old")
    from p1_world import SpikeWorker
    print(f"GPUs: {int(ray.cluster_resources().get('GPU', 0))}", flush=True)

    names = list(DIRS)
    workers = [SpikeWorker.remote() for _ in range(min(4, len(names)))]
    traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
    ctx_img, goal_img = traj["observations"][0, 0], traj["observations"][0, 1]
    pose = traj["states"][0, 0]

    t0 = time.time()

    def run_one(w, name):
        a_star = DIRS[name]
        prob = ray.get(w.build_problem.remote(ctx_img, goal_img, pose, goal_steps=GOAL_STEPS, a_star=a_star))
        diag = ray.get(w.diagnose.remote(prob["z0"], prob["p0"], prob["z_goal"],
                                         a_star=a_star, steps=GOAL_STEPS, cloud_samples=32, cloud_steps=4))
        deep2 = ray.get(w.deep_cem_test.remote(prob["z0"], prob["p0"], prob["z_goal"],
                                               rollout=2, samples=96, cem_steps=6))
        deep4 = ray.get(w.deep_cem_test.remote(prob["z0"], prob["p0"], prob["z_goal"],
                                               rollout=4, samples=48, cem_steps=6))
        return {"name": name, "e_oracle": diag["e_oracle"], "e_greedy": diag["e_cem"],
                "deep2": deep2.get("energy"), "deep4": deep4.get("energy"),
                "cem_xyz": diag["cem_xyz"]}

    # one direction per worker, round-robin
    futs = [workers[i % len(workers)].build_problem.remote for i in range(0)]  # noqa (placeholder)
    results = []
    inflight = {}
    free = list(workers)
    todo = list(names)
    # simple scheduler: keep all workers busy
    import threading
    def submit(w, name):
        return (name, w)
    # do it sequentially per worker via ray futures
    tasks = []
    for i, name in enumerate(names):
        w = workers[i % len(workers)]
        tasks.append((name, w))

    # fire each as a remote chain isn't trivial; just run serially per direction (clear logging)
    for name in names:
        w = workers[names.index(name) % len(workers)]
        r = run_one(w, name)
        results.append(r)
        og, gg = r["e_oracle"][-1], r["e_greedy"][-1]
        d2 = r["deep2"][-1] if r["deep2"] else float("nan")
        d4 = r["deep4"][-1] if r["deep4"] else float("nan")
        print(f"  [{name:>8}] oracle_final={og:.3f}  greedy_final={gg:.3f}  "
              f"deep2_final={d2:.3f}  deep4_final={d4:.3f}  gap(greedy-deep4)={gg-d4:+.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("\n=== ranking by (greedy_final - deep4_final) gap [bigger = cloud matters more] ===", flush=True)
    ranked = sorted(results, key=lambda r: -((r["e_greedy"][-1]) - (r["deep4"][-1] if r["deep4"] else 1e9)))
    for r in ranked:
        gg = r["e_greedy"][-1]; d4 = r["deep4"][-1] if r["deep4"] else float("nan")
        print(f"  {r['name']:>8}: greedy {gg:.3f} vs deep4 {d4:.3f}  gap {gg-d4:+.3f}", flush=True)
    best = ranked[0]
    print(f"\nBEST HARD GOAL: {best['name']}  (greedy fails at {best['e_greedy'][-1]:.3f}, "
          f"cloud reaches {best['deep4'][-1] if best['deep4'] else float('nan'):.3f})", flush=True)
    print(f"greedy cem_xyz picks (first 3 ticks) for {best['name']}:\n{np.round(best['cem_xyz'][:3], 3)}", flush=True)

    for w in workers:
        ray.kill(w)


if __name__ == "__main__":
    main()

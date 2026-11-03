"""Closed-loop episodes: does warm-starting CEM (fewer rounds per step) keep control quality?

The world model stands in for the robot: each executed action is rolled forward through the predictor to produce
the next observation latent. The goal is made reachable by rolling a known action sequence forward from the start,
so we can measure the physical error (end-effector distance to the goal position, in cm) as well as latent energy.

    python scripts/bench_closed_loop.py --gpus 8 --steps 12 --seeds 3
Writes results/04_closed_loop.json (updated after every episode).
"""
import argparse, json, os, sys, time
import numpy as np
import ray
from ray import serve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jepa_rt import EXAMPLE_TRAJ, RUNTIME_ENV, ensure_assets  # noqa: E402
from jepa_rt.serve_app import build_app  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", type=int, default=8)
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--goal-action", type=float, nargs=3, default=[0.03, 0.02, 0.0])
ap.add_argument("--goal-steps", type=int, default=6)
ap.add_argument("--strategies", nargs="+", default=["cold10", "warm3", "warm2", "cold3"])
ap.add_argument("--out", default=os.path.join(ROOT, "results/04_closed_loop.json"))
args = ap.parse_args()

traj = np.load(EXAMPLE_TRAJ)
ensure_assets()
ray.init(address="auto", runtime_env=RUNTIME_ENV, logging_level="ERROR", log_to_driver=False)
serve.run(build_app(n_gpus=args.gpus), name="jepa_planner", route_prefix="/jepa")
planner = serve.get_app_handle("jepa_planner")
wm = serve.get_deployment_handle("WorldModel", app_name="jepa_planner")

# -- the task: start at the first frame of Meta's example trajectory; goal = where a known action sequence leads
z0 = wm.encode.remote(traj["observations"][0][0]).result()
p0 = traj["states"][0][0].astype(np.float32)
a_star = np.zeros((args.goal_steps, 7), np.float32); a_star[:, :3] = args.goal_action
z_goal, p_goal = wm.simulate.remote(z0, p0, a_star).result()
print(f"goal is {np.linalg.norm(p_goal[:3] - p0[:3]) * 100:.1f} cm away", flush=True)

results = json.load(open(args.out)) if os.path.exists(args.out) else {}
results["task"] = {"goal_action": args.goal_action, "goal_steps": args.goal_steps,
                   "start_dist_cm": float(np.linalg.norm(p_goal[:3] - p0[:3]) * 100), "gpus": args.gpus}
for strat in args.strategies:
    warm_start, rounds = strat.startswith("warm"), int(strat[4:])
    for seed in range(args.seeds):
        z, p, warm = z0, p0, None
        dist, energy, plan_ms = [float(np.linalg.norm(p_goal[:3] - p[:3]) * 100)], [], []
        for t in range(args.steps):
            # warm-start strategies: the first step is a full cold search, later steps reuse the previous plan
            r = 10 if (warm_start and warm is None) else rounds
            out = planner.plan.remote(None, p, None, z=z, z_goal=z_goal, warm=warm if warm_start else None,
                                      rounds=r, seed=1000 * seed + t).result(timeout_s=600)
            warm = out["warm"]
            z, p = wm.simulate.remote(z, p, out["action"][None]).result()     # execute the first action
            dist.append(float(np.linalg.norm(p_goal[:3] - p[:3]) * 100))
            energy.append(out["elite_energy"][-1]); plan_ms.append(out["timing_ms"]["total"])
        results.setdefault(strat, {})[str(seed)] = {"dist_cm": dist, "elite_energy": energy, "plan_ms": plan_ms}
        print(f"{strat} seed {seed}: dist {dist[0]:.1f} -> {dist[-1]:.1f} cm (min {min(dist):.1f}), "
              f"plan ms median {np.median(plan_ms[1:]):.0f}", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
serve.delete("jepa_planner")

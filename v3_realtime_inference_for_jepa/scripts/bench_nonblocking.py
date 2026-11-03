"""Blocking vs non-blocking control, in wall-clock time.

The simulated robot takes `--exec-s` seconds to execute each action, and execution is noisy (the arm does not land
exactly where commanded), so a plan made from the *predicted* next state is made from slightly wrong information.

  blocking:     plan(state_t) -> execute a_t (robot moves) -> plan(state_t+1) -> ...      step = plan + exec
  non-blocking: while the robot executes a_t, plan a_t+1 from the world model's prediction of where a_t will
                land (warm-started from the current plan); then correct on the next step.  step = max(plan, exec)

    python scripts/bench_nonblocking.py --gpus 8 --rounds 3 --exec-s 1.0
Writes results/05_nonblocking.json.
"""
import argparse, asyncio, json, os, sys, time
import numpy as np
import ray
from ray import serve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jepa_rt import EXAMPLE_TRAJ, RUNTIME_ENV, ensure_assets  # noqa: E402
from jepa_rt.serve_app import build_app  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", type=int, default=8)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--exec-s", type=float, default=1.0)
ap.add_argument("--noise-m", type=float, default=0.004, help="std of execution error per axis (m)")
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out", default=os.path.join(ROOT, "results/05_nonblocking.json"))
args = ap.parse_args()

traj = np.load(EXAMPLE_TRAJ)
ensure_assets()
ray.init(address="auto", runtime_env=RUNTIME_ENV, logging_level="ERROR", log_to_driver=False)
serve.run(build_app(n_gpus=args.gpus), name="jepa_planner", route_prefix="/jepa")
planner = serve.get_app_handle("jepa_planner")
wm = serve.get_deployment_handle("WorldModel", app_name="jepa_planner")

z0 = wm.encode.remote(traj["observations"][0][0]).result()
p0 = traj["states"][0][0].astype(np.float32)
a_star = np.zeros((6, 7), np.float32); a_star[:, :3] = [0.03, 0.02, 0.0]          # same task as bench_closed_loop
z_goal, p_goal = wm.simulate.remote(z0, p0, a_star).result()
dist = lambda p: float(np.linalg.norm(p_goal[:3] - p[:3]) * 100)   # noqa: E731


async def execute(z, p, a, rng):
    """The robot: takes exec_s to move; lands with execution error. Returns the true next (latent, pose)."""
    noisy = a.copy(); noisy[:3] += rng.normal(0, args.noise_m, 3)
    t = time.perf_counter()
    nxt = await wm.simulate.remote(z, p, noisy[None])
    await asyncio.sleep(max(0.0, args.exec_s - (time.perf_counter() - t)))
    return nxt


async def episode(mode, seed):
    rng = np.random.default_rng(seed)
    z, p, t0 = z0, p0, time.perf_counter()
    out = await planner.plan.remote(None, p, None, z=z, z_goal=z_goal, rounds=10, seed=seed * 1000)  # first: cold
    log = [(0.0, dist(p))]
    for t in range(args.steps):
        a, warm = out["action"], out["warm"]
        if mode == "blocking":
            z, p = await execute(z, p, a, rng)                      # robot moves, planner idle
            out = await planner.plan.remote(None, p, None, z=z, z_goal=z_goal, warm=warm, rounds=args.rounds,
                                             seed=seed * 1000 + t + 1)
        else:
            z_pred, p_pred = await wm.simulate.remote(z, p, a[None])   # where a_t should land (~20 ms)
            moving = asyncio.ensure_future(execute(z, p, a, rng))      # robot starts moving ...
            plan_next = planner.plan.remote(None, p_pred, None, z=z_pred, z_goal=z_goal, warm=warm,
                                            rounds=args.rounds, seed=seed * 1000 + t + 1)   # ... and we plan ahead
            (z, p), out = await asyncio.gather(moving, plan_next)
        log.append((time.perf_counter() - t0, dist(p)))
    return log


async def main():
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res["config"] = vars(args) | {"start_dist_cm": dist(p0)}
    for mode in ("blocking", "nonblocking"):
        for seed in range(args.seeds):
            log = await episode(mode, seed)
            res.setdefault(mode, {})[str(seed)] = log
            steps_s = np.diff([t for t, _ in log])
            print(f"{mode} seed {seed}: {dist(p0):.1f} -> {log[-1][1]:.1f} cm in {log[-1][0]:.1f} s "
                  f"(step median {np.median(steps_s[1:]):.2f} s)", flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
    serve.delete("jepa_planner")

asyncio.run(main())

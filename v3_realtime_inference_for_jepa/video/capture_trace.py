"""Capture everything the video shows, from one live blocking and one live non-blocking episode on the cluster.

Same task and settings as scripts/bench_nonblocking.py (goal 21.6 cm away, 1 s per move, 4 mm execution noise,
first plan 10 rounds then 3 warm-started rounds). Logs, on one clock (the robot's):
  * every move: start/end time, gripper xyz before/after;
  * every plan: when it was requested and returned, and per CEM round its start/end, the 800 candidates' imagined
    end positions + energies, and each GPU shard's start/end.

    python video/capture_trace.py          -> video/trace.json   (~3 min on 8 GPUs)
"""
import asyncio, json, os, sys, time
import numpy as np
import ray
from ray import serve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jepa_rt import EXAMPLE_TRAJ, RUNTIME_ENV, ensure_assets  # noqa: E402
from jepa_rt.serve_app import build_app  # noqa: E402

STEPS, EXEC_S, NOISE_M, ROUNDS, SEED = 12, 1.0, 0.004, 3, 0
ensure_assets()
traj = np.load(EXAMPLE_TRAJ)
ray.init(address="auto", runtime_env=RUNTIME_ENV, logging_level="ERROR", log_to_driver=False)
serve.run(build_app(n_gpus=8), name="jepa_planner", route_prefix="/jepa")
planner = serve.get_app_handle("jepa_planner")
wm = serve.get_deployment_handle("WorldModel", app_name="jepa_planner")

z0 = wm.encode.remote(traj["observations"][0][0]).result()
p0 = traj["states"][0][0].astype(np.float32)
a_star = np.zeros((6, 7), np.float32); a_star[:, :3] = [0.03, 0.02, 0.0]
z_goal, p_goal = wm.simulate.remote(z0, p0, a_star).result()
r4 = lambda a: np.round(np.asarray(a, float), 5).tolist()   # noqa: E731


def plan_record(out, t_call, t_back):
    return {"t_call": t_call, "t_back": t_back, "action": r4(out["action"]), "plan": r4(out["plan"]),
            "rounds": [{"t_start": t_call + r["t_start_ms"] / 1e3, "t_end": t_call + r["t_end_ms"] / 1e3,
                        "cand_end_xyz": r4(r["cand_end_xyz"]), "energy": r4(r["energy"]),
                        "shards": [[t_call + a / 1e3, t_call + b / 1e3] for a, b in r["shards_ms"]]}
                       for r in out["rounds_detail"]]}


async def episode(mode):
    rng = np.random.default_rng(SEED)
    log = {"moves": [], "plans": []}
    t0 = time.perf_counter(); now = lambda: time.perf_counter() - t0   # noqa: E731

    async def plan(z, p, warm, rounds, seed):
        tc = now()
        out = await planner.plan.remote(None, p, None, z=z, z_goal=z_goal, warm=warm, rounds=rounds, seed=seed,
                                        detail=True)
        log["plans"].append(plan_record(out, tc, now()))
        return out

    async def move(z, p, a):
        noisy = a.copy(); noisy[:3] += rng.normal(0, NOISE_M, 3)
        ts = now()
        nz, np_ = await wm.simulate.remote(z, p, noisy[None])
        await asyncio.sleep(max(0.0, EXEC_S - (now() - ts)))
        log["moves"].append({"t_start": ts, "t_end": now(), "xyz_from": r4(p[:3]), "xyz_to": r4(np_[:3])})
        return nz, np_

    z, p = z0, p0
    out = await plan(z, p, None, 10, SEED * 1000)
    for t in range(STEPS):
        a, warm = out["action"], out["warm"]
        if mode == "blocking":
            z, p = await move(z, p, a)
            out = await plan(z, p, warm, ROUNDS, SEED * 1000 + t + 1)
        else:
            z_pred, p_pred = await wm.simulate.remote(z, p, a[None])
            (z, p), out = await asyncio.gather(move(z, p, a), plan(z_pred, p_pred, warm, ROUNDS, SEED * 1000 + t + 1))
    log["t_total"] = now()
    return log


async def main():
    res = {"config": {"steps": STEPS, "exec_s": EXEC_S, "noise_m": NOISE_M, "rounds": ROUNDS, "gpus": 8,
                      "samples": 800, "first_plan_rounds": 10},
           "xyz_start": r4(p0[:3]), "xyz_goal": r4(p_goal[:3]),
           "frame_start": "franka_example_traj.npz observations[0][0]"}
    for mode in ("blocking", "nonblocking"):
        res[mode] = await episode(mode)
        d = np.linalg.norm(np.array(res[mode]["moves"][-1]["xyz_to"]) - np.array(res["xyz_goal"])) * 100
        print(f"{mode}: {res[mode]['t_total']:.1f} s, final distance {d:.1f} cm, "
              f"{sum(len(p['rounds']) for p in res[mode]['plans'])} CEM rounds", flush=True)
    json.dump(res, open(os.path.join(ROOT, "video", "trace.json"), "w"))
    serve.delete("jepa_planner")

asyncio.run(main())

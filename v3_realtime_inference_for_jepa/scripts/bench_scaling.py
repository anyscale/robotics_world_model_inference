"""Deploy the Serve app at several GPU counts and measure plan latency (cold search: 800 x 10 rounds).

    python scripts/bench_scaling.py --gpus 8 4 2 1 --plans 5
Writes results/03_scaling.json (updated after each GPU count, so a partial run still leaves data).
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
ap.add_argument("--gpus", type=int, nargs="+", default=[8, 4, 2, 1])
ap.add_argument("--plans", type=int, default=5)
ap.add_argument("--out", default=os.path.join(ROOT, "results/03_scaling.json"))
args = ap.parse_args()

traj = np.load(EXAMPLE_TRAJ)
frame, goal, pose = traj["observations"][0][0], traj["observations"][0][-1], traj["states"][0][0]

ensure_assets()
ray.init(address="auto", runtime_env=RUNTIME_ENV, logging_level="ERROR", log_to_driver=False)
results = json.load(open(args.out)) if os.path.exists(args.out) else {}
for W in args.gpus:
    t = time.time()
    serve.run(build_app(n_gpus=W), name="jepa_planner", route_prefix="/jepa")
    deploy_s = time.time() - t
    h = serve.get_app_handle("jepa_planner")
    h.plan.remote(frame, pose, goal, seed=0).result(timeout_s=600)            # first call: goal encode etc.
    runs = [h.plan.remote(frame, pose, goal, seed=s).result(timeout_s=600) for s in range(1, args.plans + 1)]
    tot = [r["timing_ms"]["total"] for r in runs]
    results[str(W)] = {
        "deploy_s": deploy_s, "plan_ms_median": float(np.median(tot)), "plan_ms_all": tot,
        "encode_ms_median": float(np.median([r["timing_ms"]["encode"] for r in runs])),
        "round_ms_median": float(np.median([np.median(r["timing_ms"]["rounds"]) for r in runs])),
        "gpu_ms_median": float(np.median([np.median(r["timing_ms"]["gpu"]) for r in runs])),
        "final_elite_energy": [r["elite_energy"][-1] for r in runs],
        "actions": [np.round(r["action"], 4).tolist() for r in runs],
    }
    print(W, "GPUs:", {k: v for k, v in results[str(W)].items() if k not in ("plan_ms_all", "actions")}, flush=True)
    json.dump(results, open(args.out, "w"), indent=1)
    serve.delete("jepa_planner")

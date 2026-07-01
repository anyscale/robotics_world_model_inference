"""
p1_run.py — driver for the P1 feasibility spike (chunk-based MPC loop). Runs on the CPU head; fans
episodes across the 4 L4 workers. Sweeps cloud latency Δ and seed-averages to answer: does vector-add
reconciliation beat naive-apply-stale as the cloud goes stale?  Produces p1_feasibility.png + a table.

Usage:  python p1_run.py [--quick]
"""
import sys
import time

import numpy as np
import ray

REPO_DIR = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
TORCH_HOME = "/mnt/cluster_storage/torch_hub"
VJEPA_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"

QUICK = "--quick" in sys.argv
NUM_WORKERS = 4
DELTAS = [0, 4, 8] if QUICK else [0, 2, 4, 6, 8, 10]
SEEDS = [0] if QUICK else [0, 1, 2, 3]
HORIZON = 16 if QUICK else 20
CLOUD_PERIOD = 4
ROLLOUT = 2
GOAL_STEPS = 8
CLOUD_SAMPLES, CLOUD_STEPS = (32, 4) if QUICK else (32, 4)
REFLEX_SAMPLES, REFLEX_STEPS, REFLEX_GAIN = 4, 1, 0.5
ENV_DRIFT = [0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]  # exogenous off-axis push (+y) per tick


def main():
    ray.init(address="auto", logging_level="ERROR", runtime_env={
        "py_modules": [f"{REPO_DIR}/p1_world.py"],
        "env_vars": {"TORCH_HOME": TORCH_HOME, "HF_HOME": "/mnt/cluster_storage/hf_cache",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    })
    from p1_world import SpikeWorker

    print(f"GPUs: {int(ray.cluster_resources().get('GPU', 0))} | deltas: {DELTAS} | seeds: {SEEDS} | "
          f"horizon: {HORIZON} | chunk rollout: {ROLLOUT} | cloud {CLOUD_SAMPLES}x{CLOUD_STEPS}")
    workers = [SpikeWorker.remote() for _ in range(NUM_WORKERS)]

    traj = np.load(f"{VJEPA_REPO}/notebooks/franka_example_traj.npz")
    ctx_img, goal_img = traj["observations"][0, 0], traj["observations"][0, 1]
    pose = traj["states"][0, 0]

    t0 = time.time()
    prob = ray.get(workers[0].build_problem.remote(ctx_img, goal_img, pose, goal_steps=GOAL_STEPS))
    z0, p0, z_goal = prob["z0"], prob["p0"], prob["z_goal"]
    print(f"problem built in {time.time()-t0:.1f}s\n")

    cfg = dict(cloud_period=CLOUD_PERIOD, horizon=HORIZON, rollout=ROLLOUT,
               reflex_samples=REFLEX_SAMPLES, reflex_steps=REFLEX_STEPS,
               cloud_samples=CLOUD_SAMPLES, cloud_steps=CLOUD_STEPS, reflex_gain=REFLEX_GAIN,
               env_drift=ENV_DRIFT, pg_np=prob["p_goal"])

    # (strategy, delta, seed) jobs. reflex ignores delta -> only delta 0.
    jobs = ([("reflex", 0, s) for s in SEEDS]
            + [("naive", d, s) for d in DELTAS for s in SEEDS]
            + [("vector_add", d, s) for d in DELTAS for s in SEEDS])

    t0 = time.time()
    ref2job, futs = {}, []
    for i, (st, d, s) in enumerate(jobs):
        f = workers[i % NUM_WORKERS].run_episode.remote(z0, p0, z_goal, strategy=st, delta=d, seed=s, **cfg)
        ref2job[f] = (st, d, s); futs.append(f)

    # collect with LIVE per-episode progress (ray.wait), so the run is never silent
    results, pairs, pending, done = [], [], list(futs), 0
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        r = ray.get(ready[0]); job = ref2job[ready[0]]
        results.append(r); pairs.append(job); done += 1
        st, d, s = job
        print(f"  [{done:>2}/{len(jobs)}] {st:>10} Δ={d:<2} seed={s}  pose_err={r['final_pose_err']:.4f}  "
              f"energy={r['final_energy']:.4f}  ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"ran {len(results)} episodes in {time.time()-t0:.1f}s\n")

    # aggregate over seeds, per (strategy, delta): pose error (PRIMARY) + energy (secondary)
    agg = {}
    for (st, d, s), r in zip(pairs, results):
        a = agg.setdefault((st, d), {"pe": [], "en": [], "pe_trace": [], "en_trace": []})
        a["pe"].append(r["final_pose_err"]); a["en"].append(r["final_energy"])
        a["pe_trace"].append(r["pose_err"]); a["en_trace"].append(r["energy"])

    def m(metric, st, d): return float(np.mean(agg[(st, d)][metric]))
    def sd(metric, st, d): return float(np.std(agg[(st, d)][metric]))
    reflex_PE, reflex_E = m("pe", "reflex", 0), m("en", "reflex", 0)

    print(f"{'delta':>6} | {'naive pose_err':>16} | {'vector_add pose_err':>20} | {'reflex':>8}")
    print("-" * 62)
    for d in DELTAS:
        print(f"{d:>6} | {m('pe','naive',d):>8.4f}±{sd('pe','naive',d):<6.4f} | "
              f"{m('pe','vector_add',d):>11.4f}±{sd('pe','vector_add',d):<6.4f} | {reflex_PE:>8.4f}")

    # plot: LEFT = pose-tracking error vs Δ (exact, primary) ; RIGHT = latent energy vs Δ (secondary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, base, ylab, title in [
        (ax1, "pe", reflex_PE, "final pose error to goal (↓)", "Pose error vs. cloud latency (exact)"),
        (ax2, "en", reflex_E, "final energy to goal (↓)", "Latent energy vs. cloud latency (world model)")]:
        nv = [m(metric, "naive", d) for d in DELTAS]; nv_s = [sd(metric, "naive", d) for d in DELTAS]
        va = [m(metric, "vector_add", d) for d in DELTAS]; va_s = [sd(metric, "vector_add", d) for d in DELTAS]
        ax.axhline(base, color="gray", ls=":", lw=2, label="reflex-only")
        ax.errorbar(DELTAS, nv, yerr=nv_s, fmt="s--", color="C3", lw=2, capsize=3, label="naive (stale)")
        ax.errorbar(DELTAS, va, yerr=va_s, fmt="o-", color="C0", lw=2, capsize=3, label="vector-add reconciled")
        ax.set_xlabel("cloud latency  (ticks stale, Δ)"); ax.set_ylabel(ylab)
        ax.set_title(title); ax.legend()
    fig.tight_layout()
    out = f"{REPO_DIR}/images/p1_feasibility.png"
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")

    nv = [m("pe", "naive", d) for d in DELTAS]; va = [m("pe", "vector_add", d) for d in DELTAS]
    grows = nv[-1] - nv[0]; va_flat = np.std(va)
    print(f"\nVERDICT (pose error, exact):")
    print(f"  naive grows {grows:+.4f} from Δ={DELTAS[0]}→{DELTAS[-1]}  (mean {np.mean(nv):.4f})")
    print(f"  vector_add stays {np.mean(va):.4f}  (std across Δ = {va_flat:.4f})")
    print(f"  vector_add beats naive at Δ={DELTAS[-1]} by {nv[-1]-va[-1]:+.4f}; reflex={reflex_PE:.4f}")
    clean = grows > 0.02 and (nv[-1] - va[-1]) > 0.02 and va_flat < grows / 2
    print(f"  => CLEAN SEPARATION (pose)? {'YES' if clean else 'NO — push drift / seeds'}")

    for w in workers:
        ray.kill(w)


if __name__ == "__main__":
    main()

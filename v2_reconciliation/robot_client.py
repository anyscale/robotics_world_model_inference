"""
robot_client.py — the on-robot "fast brain" + reconciliation, talking to the cloud slow brain over
real HTTP (Ray Serve). The GPS model: the cloud returns a corrected DESTINATION; the robot keeps acting
and steers toward it (bounded steps) until the next reply. Staleness Δ is REAL — it's however many
control ticks elapse during the actual cloud round-trip (async, non-blocking).

  - naive:      anchor the cloud's destination at the robot's CURRENT (drifted) pose  -> carries the drift
  - vector_add: anchor it at the retained SNAPSHOT pose                               -> drift cancelled

Pose math is plain numpy here (CEM actions only move xyz + gripper; rotation dims are unused), so the
fast brain needs no GPU — true to "reflex on the edge."
"""
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import requests

MAXNORM = 0.05


def post_plan(url, snap, z_goal, seq, samples=32, cem_steps=4, rollout=2, timeout=300):
    """Blocking HTTP call (run in a worker thread). Returns (chunk, server_ms, round_trip_ms)."""
    req = {"z": snap["z"], "pose": snap["pose"], "z_goal": z_goal, "seq_id": seq,
           "samples": samples, "cem_steps": cem_steps, "rollout": rollout}
    t0 = time.time()
    r = requests.post(url + "/plan", data=pickle.dumps(req),
                      headers={"Content-Type": "application/octet-stream"}, timeout=timeout)
    rtt = (time.time() - t0) * 1000.0
    out = pickle.loads(r.content)
    return out["chunk"], out["compute_ms"], rtt


def _destination(anchor_pose, chunk):
    """Cloud's corrected destination = anchor + sum of chunk actions (xyz add, gripper add+clip)."""
    dest = anchor_pose.copy().reshape(7)
    dest[:3] = dest[:3] + chunk[:, :3].sum(axis=0)
    dest[6] = np.clip(dest[6] + chunk[:, 6].sum(), 0.0, 1.0)
    return dest





def run_episode(world, url, z_goal, *, strategy, control_hz=3.0, cloud_every_ticks=6,
                horizon=24, drift=None, samples=32, cem_steps=4, rollout=2, pool=None, log=print):
    """Run one closed-loop episode against the real Serve endpoint. Returns traces + measured Δ/latency."""
    tick_dt = 1.0 / control_hz
    own_pool = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=4)

    p0 = ray.get(world.snapshot.remote())["pose"]
    pose = p0.copy()
    destination = None
    pending = []                       # {req_tick, snap_pose, future}
    perrs, deltas, latencies = [], [], []

    for t in range(horizon):
        tick_start = time.time()

        # ONE request in flight (robot asks, keeps acting while it waits, asks again on reply) —
        # this bounds queueing so Δ reflects a single real round-trip, not a growing backlog.
        if not pending:
            snap = ray.get(world.snapshot.remote())
            fut = pool.submit(post_plan, url, snap, z_goal, t, samples, cem_steps, rollout)
            pending.append({"req_tick": t, "snap_pose": snap["pose"], "fut": fut})

        # adopt any replies that have come back -> set the (reconciled) destination
        still = []
        for pc in pending:
            if pc["fut"].done():
                chunk, server_ms, rtt = pc["fut"].result()
                delta = t - pc["req_tick"]            # REAL staleness in control ticks
                anchor = pose if strategy == "naive" else pc["snap_pose"]
                destination = _destination(anchor, chunk)
                deltas.append(delta); latencies.append(rtt)
                log(f"    [{strategy:>10}] tick {t:>2}: cloud reply (req@{pc['req_tick']}) "
                    f"Δ={delta} ticks, latency={rtt:.0f}ms, pose_err={perrs[-1] if perrs else float('nan'):.3f}")
            else:
                still.append(pc)
        pending = still

        # FAST BRAIN: steer toward the current destination (hold if none yet)
        action = _step_toward(pose, destination) if destination is not None else np.zeros((1, 1, 7), np.float32)
        res = ray.get(world.step.remote(action, drift))
        pose = res["pose"].reshape(7)
        perrs.append(res["pose_err"])

        # maintain the control rate (so Δ reflects real cloud latency / real time)
        time.sleep(max(0.0, tick_dt - (time.time() - tick_start)))

    # drain any in-flight request so it can't bleed into the next episode's timing
    for pc in pending:
        try:
            pc["fut"].result(timeout=60)
        except Exception:
            pass
    if own_pool:
        pool.shutdown(wait=False)
    return {"strategy": strategy, "pose_err": perrs, "final_pose_err": perrs[-1],
            "deltas": deltas, "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "mean_latency_ms": float(np.mean(latencies)) if latencies else float("nan")}

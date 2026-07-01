"""
robot_client.py — the on-robot FAST BRAIN (a distilled reactive policy) + reconciliation, talking to
the cloud slow brain over real HTTP (Ray Serve).

The fast brain is the distilled reflex (reflex_policy.py): a one-shot MLP on the perceived latent, run
every control tick in numpy — no GPU, no search. On its own it's reactive and can't hold the goal.
The cloud returns a smarter, but STALE, corrected DESTINATION; the robot steers toward it (bounded
steps) until the next reply, and reconciles the staleness:

  - naive:      anchor the cloud's destination at the robot's CURRENT (drifted) pose  -> carries the drift
  - vector_add: anchor it at the retained SNAPSHOT pose                               -> drift cancelled

Staleness Δ is REAL — the control ticks that elapse during the actual cloud round-trip (async).
"""
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import requests

MAXNORM = 0.05


def post_plan(url, snap, z_goal, seq, timeout=300):
    """Blocking HTTP call (run in a worker thread). Returns (chunk, server_ms, round_trip_ms).
    Sends only the snapshot; the served deployment governs the CEM/shard config."""
    req = {"z": snap["z"], "pose": snap["pose"], "z_goal": z_goal, "seq_id": seq}
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


def _step_toward(pose, dest):
    """Bounded action that steers the current pose toward the reconciled destination."""
    p = pose.reshape(7)
    a = np.zeros(7, dtype=np.float32)
    a[:3] = np.clip(dest[:3] - p[:3], -MAXNORM, MAXNORM)
    a[6] = np.clip(dest[6] - p[6], -MAXNORM, MAXNORM)
    return a.reshape(1, 1, 7)


def _reflex_action(reflex, z_pool_now, z_pool_goal):
    """The distilled fast brain: one-shot action from the perceived latent, clipped to the valid range."""
    a = reflex.action_from_pooled(z_pool_now, z_pool_goal).astype(np.float32)
    a[:3] = np.clip(a[:3], -MAXNORM, MAXNORM)
    a[3:6] = 0.0
    a[6] = np.clip(a[6], -MAXNORM, MAXNORM)
    return a.reshape(1, 1, 7)



def run_episode(world, url, z_goal, *, strategy, reflex, control_hz=3.0,
                horizon=24, drift=None, pool=None, log=print):
    """Run one closed-loop episode against the real Serve endpoint. The distilled `reflex` is the fast
    brain (acts every tick until a cloud reply lands); then the robot steers to the reconciled
    destination. Returns traces + measured Δ/latency."""
    tick_dt = 1.0 / control_hz
    own_pool = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=4)

    z_pool_goal = reflex.pool(z_goal)
    snap0 = ray.get(world.snapshot.remote())
    p0 = snap0["pose"]
    pose = p0.copy()
    z_pool_now = reflex.pool(snap0["z"])   # the reflex's "perception" of the current latent
    destination = None
    pending = []                       # {req_tick, snap_pose, future}
    perrs, deltas, latencies, reply_ticks = [], [], [], []

    for t in range(horizon):
        tick_start = time.time()

        # ONE request in flight (robot asks, keeps acting while it waits, asks again on reply) —
        # this bounds queueing so Δ reflects a single real round-trip, not a growing backlog.
        # strategy="reflex" never calls the cloud at all — a pure reflex-only baseline run.
        if strategy != "reflex" and not pending:
            snap = ray.get(world.snapshot.remote())
            fut = pool.submit(post_plan, url, snap, z_goal, t)
            pending.append({"req_tick": t, "snap_pose": snap["pose"], "fut": fut})

        # adopt any replies that have come back -> set the (reconciled) destination
        still = []
        for pc in pending:
            if pc["fut"].done():
                chunk, server_ms, rtt = pc["fut"].result()
                delta = t - pc["req_tick"]            # REAL staleness in control ticks
                anchor = pose if strategy == "naive" else pc["snap_pose"]
                destination = _destination(anchor, chunk)
                deltas.append(delta); latencies.append(rtt); reply_ticks.append(t)
                log(f"    [{strategy:>10}] tick {t:>2}: cloud reply (req@{pc['req_tick']}) "
                    f"Δ={delta} ticks, latency={rtt:.0f}ms, pose_err={perrs[-1] if perrs else float('nan'):.3f}")
            else:
                still.append(pc)
        pending = still

        # FAST BRAIN: run the distilled reflex every tick; once a cloud reply lands, steer toward the
        # reconciled destination instead (the smarter, corrected target).
        if destination is not None:
            action = _step_toward(pose, destination)
        else:
            action = _reflex_action(reflex, z_pool_now, z_pool_goal)
        res = ray.get(world.step.remote(action, drift))
        pose = res["pose"].reshape(7)
        z_pool_now = res["z_pool"]
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
            "deltas": deltas, "reply_ticks": reply_ticks,
            "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "mean_latency_ms": float(np.mean(latencies)) if latencies else float("nan")}

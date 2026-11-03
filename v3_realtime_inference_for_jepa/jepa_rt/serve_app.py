"""Real-time V-JEPA 2-AC planning as two composed Ray Serve deployments.

    WorldModel  -- the GPU tier. One replica per GPU, weights resident between calls, one shard at a time.
    Planner     -- the driver. Runs the CEM rounds, fans each round's candidates out across the WorldModel
                   replicas, gathers the energies, refits. Serves the endpoint the robot calls once per step.

    from jepa_rt.serve_app import build_app
    serve.run(build_app(n_gpus=8), name="jepa_planner", route_prefix="/")

A robot (or the notebook) then POSTs its camera frame + pose + goal image to /plan and gets back the next action.

Two details that matter for latency:
  * The observation/goal latents (2 x 1.4 MB) are put in Ray's object store ONCE per plan (`ray.put`) and each
    replica reads them from shared memory and keeps them on its GPU. Every CEM round then only ships the candidate
    actions (100 x 2 x 7 floats per shard) out and the energies back.
  * Planner state lives with the caller: a reply carries the CEM distribution (`warm`), and the robot sends it back
    with its next request to warm-start the search. The Planner itself stays stateless, so it can scale out freely.
"""
import asyncio
import hashlib
import io
import os
import time
import uuid

import numpy as np
import ray
from fastapi import FastAPI, Request
from fastapi.responses import Response
from ray import serve
from ray.serve.config import RequestRouterConfig
from ray.serve.handle import DeploymentHandle

from jepa_rt.cem import CEMDistribution

MAX_CHUNK = 400   # candidates per GPU batch: keeps peak memory ~31 GB on a 40 GB A100


# ------------------------------------------------------------------------------------------------- the GPU tier
@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=1,          # one shard at a time, so a W-way fan-out lands one shard per GPU
    # Round-robin routing: the W shards of a round go to W consecutive replicas, i.e. one per GPU. (The default
    # power-of-two-choices router occasionally put two shards on one replica, doubling that round's latency.)
    request_router_config=RequestRouterConfig(
        request_router_class="ray.serve.experimental.round_robin_router:RoundRobinRouter"),
    health_check_timeout_s=120,
)
class WorldModel:
    def __init__(self, warmup_shard_sizes=(100,), horizon=2):
        # Serve starts replica processes with TORCHDYNAMO_DISABLE=1, which silently turns torch.compile into a
        # no-op (we measured 372 ms instead of 272 ms per shard). Opt back in before torch is imported.
        os.environ.pop("TORCHDYNAMO_DISABLE", None)
        from jepa_rt.model import WorldModelCore      # imported here: only GPU replicas need torch + the model
        self.core = WorldModelCore()                 # load encoder + predictor once; they stay resident
        self._ctx_key, self._ctx = None, None
        # Compile/warm the encoder and the predictor (for the shard sizes this replica will see) before taking traffic.
        self.core.encode(np.zeros((256, 256, 3), np.uint8))
        z = np.zeros((1, 256, 1408), np.float32)
        for n in sorted(set(warmup_shard_sizes)):
            self.core.score(z, np.zeros(7, np.float32), z, np.zeros((n, horizon, 7), np.float32),
                            chunk=min(n, MAX_CHUNK))
        self.core.rollout(z, np.zeros(7, np.float32), np.zeros((1, 7), np.float32))   # simulate() shape

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Camera frame (H, W, 3 uint8) -> latent [1, 256, 1408] float32."""
        return self.core.encode(frame).cpu().numpy()

    def simulate(self, z: np.ndarray, pose: np.ndarray, actions: np.ndarray):
        """Imagined outcome of executing `actions` [H, 7] from (z, pose) -> (latent, pose). Experiments only: lets the
        world model stand in for the robot + camera so we can run closed-loop episodes without hardware."""
        zf, pf = self.core.rollout(z, pose, actions)
        return zf.cpu().numpy(), pf.cpu().numpy()

    async def score(self, ctx, actions: np.ndarray):
        """Energies of one shard of candidates. `ctx` is an ObjectRef to {key, z, pose, z_goal}: fetched from the
        object store (shared memory on this node) and moved to the GPU once per plan, then reused every round."""
        import torch
        t_recv = time.time()
        if isinstance(ctx, ray.ObjectRef):
            ctx = await ctx
        if ctx["key"] != self._ctx_key:
            dev = self.core.device
            self._ctx = {k: torch.as_tensor(ctx[k], device=dev) for k in ("z", "pose", "z_goal")}
            self._ctx_key = ctx["key"]
        c = self._ctx
        t = time.perf_counter()
        e = self.core.score(c["z"], c["pose"], c["z_goal"], actions, chunk=min(len(actions), MAX_CHUNK))
        e = e.cpu().numpy()                             # .cpu() synchronises, so the timing is the real GPU time
        return e, {"gpu_ms": (time.perf_counter() - t) * 1e3, "t_recv": t_recv, "t_done": time.time(),
                   "replica": serve.get_replica_context().replica_tag}


# ------------------------------------------------------------------------------------------------- the driver
app = FastAPI()


@serve.deployment(max_ongoing_requests=32)
@serve.ingress(app)
class Planner:
    def __init__(self, world_model: DeploymentHandle, n_shards=8, samples=800, rounds=10, topk=80, horizon=2):
        self.wm = world_model
        self.n_shards, self.samples, self.rounds, self.topk, self.horizon = n_shards, samples, rounds, topk, horizon
        self._goals = {}                                # goal image hash -> latent (goals rarely change)

    async def _goal_latent(self, goal_frame):
        key = hashlib.sha1(np.ascontiguousarray(goal_frame).tobytes()).hexdigest()
        if key not in self._goals:
            self._goals[key] = await self.wm.encode.remote(goal_frame)
        return self._goals[key]

    async def plan(self, frame, pose, goal_frame, warm=None, rounds=None, samples=None, seed=None, z=None, z_goal=None,
                   detail=False):
        """One control step: encode, run CEM with each round fanned out over the GPU tier, return the action.

        warm:   the `warm` dict from this robot's previous reply -> warm-started search (shifted previous plan).
        rounds: CEM rounds for this call (default: the deployment's; warm-started calls typically need fewer).
        z, z_goal: already-encoded latents, instead of frame / goal_frame (simulated episodes).
        detail: also return every round's candidates (imagined end positions), energies and per-GPU timing
                (for visualizing the search; off by default).
        """
        t0 = time.perf_counter(); t0_wall = time.time()
        rounds, samples = rounds or self.rounds, samples or self.samples
        if z is None or z_goal is None:
            z, z_goal = await asyncio.gather(self.wm.encode.remote(frame), self._goal_latent(goal_frame))
        ctx = ray.put({"key": uuid.uuid4().hex, "z": z, "pose": np.asarray(pose, np.float32), "z_goal": z_goal})
        t_enc = time.perf_counter()

        dist = CEMDistribution(horizon=self.horizon, seed=seed, topk_frac=self.topk / self.samples)
        if warm is not None:                            # start from the previous plan, shifted by one step
            dist.mean, dist.std = np.array(warm["mean"], np.float32), np.array(warm["std"], np.float32)
            dist.shift(std_reset=warm.get("std_reset"))
        round_ms, gpu_ms, elite, trace, rounds_detail = [], [], [], [], []
        for _ in range(rounds):                         # sequential: each round refits on the last one's winners
            tr = time.perf_counter(); tr_wall = time.time()
            cands = dist.sample(samples)
            shards = np.array_split(cands, self.n_shards)
            results = await asyncio.gather(*[self.wm.score.remote(ctx, s) for s in shards])   # fan out ... gather
            energy = np.concatenate([e for e, _ in results])
            dist.refit(cands, energy)
            elite.append(float(np.sort(energy)[:self.topk].mean()))   # search progress: mean elite energy
            round_ms.append((time.perf_counter() - tr) * 1e3)
            gpu_ms.append(max(m["gpu_ms"] for _, m in results))
            trace.append([(m["replica"][-6:], round((m["t_recv"] - tr_wall) * 1e3), round((m["t_done"] - tr_wall) * 1e3),
                           round(m["gpu_ms"])) for _, m in results])
            if detail:   # times in ms from the start of this plan call; end xyz = pose + summed xyz deltas
                end_xyz = np.asarray(pose, np.float32)[:3] + cands[:, :, :3].sum(1)
                rounds_detail.append({
                    "t_start_ms": (tr - t0) * 1e3, "t_end_ms": (time.perf_counter() - t0) * 1e3,
                    "cand_end_xyz": end_xyz.astype(np.float32), "energy": energy.astype(np.float32),
                    "shards_ms": [((m["t_recv"] - t0_wall) * 1e3, (m["t_done"] - t0_wall) * 1e3) for _, m in results]})
        return {
            "action": dist.action(), "plan": dist.plan(),
            "warm": {"mean": dist.mean.copy(), "std": dist.std.copy()}, "elite_energy": elite, "trace": trace,
            "timing_ms": {"total": (time.perf_counter() - t0) * 1e3, "encode": (t_enc - t0) * 1e3,
                          "rounds": round_ms, "gpu": gpu_ms},
            **({"rounds_detail": rounds_detail} if detail else {}),
        }

    # HTTP endpoint for the robot: body = .npz with frame, pose, goal_frame (+ optional warm_mean, warm_std).
    @app.post("/plan")
    async def plan_http(self, request: Request):
        req = np.load(io.BytesIO(await request.body()), allow_pickle=False)
        warm = {"mean": req["warm_mean"], "std": req["warm_std"]} if "warm_mean" in req else None
        rounds = int(req["rounds"]) if "rounds" in req else None
        out = await self.plan(req["frame"], req["pose"], req["goal_frame"], warm=warm, rounds=rounds)
        buf = io.BytesIO()
        np.savez(buf, action=out["action"], plan=out["plan"], warm_mean=out["warm"]["mean"],
                 warm_std=out["warm"]["std"], total_ms=out["timing_ms"]["total"])
        return Response(content=buf.getvalue(), media_type="application/octet-stream")

    @app.get("/health")
    def health(self):
        return {"ok": True, "gpus": self.n_shards, "samples": self.samples, "rounds": self.rounds}


def build_app(n_gpus=8, samples=800, rounds=10, topk=80, horizon=2):
    """The whole service. `n_gpus` is the only knob for scale: it sets both the replica count and the fan-out."""
    shard_sizes = [len(s) for s in np.array_split(np.zeros(samples), n_gpus)]
    world_model = WorldModel.options(num_replicas=n_gpus).bind(warmup_shard_sizes=tuple(shard_sizes),
                                                               horizon=horizon)
    return Planner.bind(world_model, n_shards=n_gpus, samples=samples, rounds=rounds, topk=topk, horizon=horizon)

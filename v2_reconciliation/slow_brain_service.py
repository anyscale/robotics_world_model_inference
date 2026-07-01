"""
slow_brain_service.py — the cloud "slow brain" as a Ray Serve deployment (the Anyscale story).

Mirrors the isaac-lab-gr00t-ray-serve pattern: one @serve.deployment, model loaded once per replica,
FastAPI ingress with a pickled-bytes POST endpoint. A robot client POSTs a snapshot (latent + pose +
goal latent); the service runs CEM in V-JEPA latent space and returns an action CHUNK plus the real
server-side compute time. Scale out with .options(num_replicas=N) (fleet throughput) — and the per-plan
data-parallel sharding lever lands in slow_brain_service_sharded.py.
"""
import pickle
import time

import torch
from fastapi import FastAPI, Request, Response
from ray import serve

from vjepa_core import WorldModelCore

app = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 1}, max_ongoing_requests=8)
@serve.ingress(app)
class SlowBrain:
    def __init__(self, samples=32, cem_steps=4, rollout=2):
        self.core = WorldModelCore(device="cuda")
        self.samples, self.cem_steps, self.rollout = samples, cem_steps, rollout
        self.calls = 0
        # warm the kernels so the first real request isn't paying compile/alloc cost
        z = self.core.encode((torch.zeros(256, 256, 3).numpy() * 0).astype("uint8"))
        self.core.cem_plan(z, self.core.t([[[0] * 7]]), z, samples=8, cem_steps=1, rollout=1)
        print(f"[SlowBrain] ready (samples={samples} cem_steps={cem_steps} rollout={rollout})", flush=True)

    @app.get("/health")
    def health(self):
        return {"ok": True, "calls": self.calls}

    @app.post("/plan")
    async def plan_http(self, request: Request):
        body = await request.body()
        req = pickle.loads(body)
        out = self._plan(req)
        return Response(content=pickle.dumps(out), media_type="application/octet-stream")

    @torch.no_grad()
    def _plan(self, req):
        """req: {z, pose, z_goal, [samples, cem_steps, rollout, seq_id]} -> {chunk, compute_ms, seq_id}."""
        z = self.core.t(req["z"]); pose = self.core.t(req["pose"]); z_goal = self.core.t(req["z_goal"])
        samples = req.get("samples", self.samples)
        cem_steps = req.get("cem_steps", self.cem_steps)
        rollout = req.get("rollout", self.rollout)
        torch.cuda.synchronize(); t0 = time.time()
        chunk = self.core.cem_plan(z, pose, z_goal, samples=samples, cem_steps=cem_steps, rollout=rollout)
        torch.cuda.synchronize()
        self.calls += 1
        return {"chunk": chunk[0].float().cpu().numpy(),  # [rollout, 7]
                "compute_ms": (time.time() - t0) * 1000.0,
                "seq_id": req.get("seq_id")}


def build_app(samples=32, cem_steps=4, rollout=2, num_replicas=1):
    return SlowBrain.options(num_replicas=num_replicas).bind(
        samples=samples, cem_steps=cem_steps, rollout=rollout)

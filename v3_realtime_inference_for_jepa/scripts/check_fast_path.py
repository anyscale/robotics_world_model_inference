"""On one GPU: (1) check the fast `score` path against the reference, (2) time Meta's unmodified CEM (the baseline).

    python scripts/check_fast_path.py
"""
import json
import os
import sys
import time

import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jepa_rt import RUNTIME_ENV  # noqa: E402


@ray.remote(num_gpus=1)
def run():
    import torch
    from jepa_rt.model import WorldModelCore
    from jepa_rt.cem import CEMDistribution
    t0 = time.time()
    core = WorldModelCore()
    out = {"load_s": time.time() - t0, "gpu": torch.cuda.get_device_name()}
    traj = np.load(os.path.join(os.environ["TORCH_HOME"], "hub/facebookresearch_vjepa2_main/notebooks/franka_example_traj.npz"))
    obs, states = traj["observations"][0], traj["states"][0]
    z, zg = core.encode(obs[0]), core.encode(obs[-1])
    pose = states[0]
    # -- (1) fast vs reference on the same candidates
    cands = CEMDistribution(horizon=2, seed=0).sample(128)
    e_ref = core.score(z, pose, zg, cands, fast=False).cpu().numpy()
    e_fast = core.score(z, pose, zg, cands, fast=True).cpu().numpy()
    k = 13
    out["check"] = {
        "max_abs_diff": float(np.abs(e_ref - e_fast).max()), "energy_range": [float(e_ref.min()), float(e_ref.max())],
        "pearson": float(np.corrcoef(e_ref, e_fast)[0, 1]),
        "topk_overlap": len(set(np.argsort(e_ref)[:k]) & set(np.argsort(e_fast)[:k])) / k,
    }
    # -- (2) Meta's reference cem(), unmodified, fp32: 800 samples x 10 rounds, keep 80, horizon 2
    from utils.mpc_utils import cem, compute_new_pose
    import torch.nn.functional as F
    pred, tpf = core.predictor, core.tpf

    def step(reps, actions, poses):
        B, T, N, D = reps.size()
        nxt = pred(reps.flatten(1, 2), actions, poses)[:, -tpf:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),)).view(B, 1, N, D)
        return nxt, compute_new_pose(poses[:, -1:], actions[:, -1:])

    zz, zzg = z[:, None], zg[:, None]
    pp = torch.as_tensor(pose, device="cuda", dtype=torch.float32).reshape(1, 1, 7)
    times = []
    for rep in range(3):
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            act = cem(context_frame=zz, context_pose=pp, goal_frame=zzg, world_model=step, rollout=2, samples=800,
                      topk=80, cem_steps=10, momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05)
        torch.cuda.synchronize(); times.append(time.time() - t)
    out["reference_cem_s"] = times
    out["reference_action"] = act[0, 0].tolist()
    out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(json.dumps(out), flush=True)
    # -- (3) fast score timing: one round of 800 on one GPU, and a 1/8 shard (100)
    for n, chunk in ((800, 400), (800, 200), (100, None)):
        c = CEMDistribution(horizon=2, seed=1).sample(n)
        torch.cuda.reset_peak_memory_stats()
        core.score(z, pose, zg, c, chunk=chunk); torch.cuda.synchronize()
        ts = []
        for _ in range(5):
            t = time.time(); core.score(z, pose, zg, c, chunk=chunk); torch.cuda.synchronize(); ts.append(time.time() - t)
        out[f"fast_score_{n}_chunk{chunk}_s"] = float(np.median(ts))
        out[f"fast_score_{n}_chunk{chunk}_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


if __name__ == "__main__":
    ray.init(address="auto", runtime_env=RUNTIME_ENV, logging_level="ERROR")
    print(json.dumps(ray.get(run.remote()), indent=1))

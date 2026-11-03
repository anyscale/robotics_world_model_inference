"""1 GPU: does torch.compile speed up the fast score path? (shard sizes 100 and 400)"""
import json, os, sys, time
import numpy as np
import ray
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jepa_rt import RUNTIME_ENV, ensure_assets  # noqa: E402


@ray.remote(num_gpus=1)
def run():
    import torch
    from jepa_rt.model import WorldModelCore
    from jepa_rt.cem import CEMDistribution
    core = WorldModelCore()
    from jepa_rt import EXAMPLE_TRAJ
    traj = np.load(EXAMPLE_TRAJ)
    z, zg, pose = core.encode(traj["observations"][0][0]), core.encode(traj["observations"][0][-1]), traj["states"][0][0]
    out = {}

    def bench(tag, n, chunk=None, reps=5):
        c = CEMDistribution(horizon=2, seed=1).sample(n)
        t = time.time(); e0 = core.score(z, pose, zg, c, chunk=chunk); torch.cuda.synchronize(); first = time.time() - t
        ts = []
        for _ in range(reps):
            t = time.time(); e = core.score(z, pose, zg, c, chunk=chunk); torch.cuda.synchronize(); ts.append(time.time() - t)
        out[f"{tag}_{n}"] = {"median_s": float(np.median(ts)), "first_call_s": first}
        return e.cpu().numpy()

    ref100 = bench("eager", 100); bench("eager", 400)
    core._block_step = torch.compile(core._block_step, dynamic=False)
    c100 = bench("compiled", 100); bench("compiled", 400)
    out["compiled_vs_eager_max_abs_diff_100"] = float(np.abs(ref100 - c100).max())
    return out


if __name__ == "__main__":
    ensure_assets()
    ray.init(address="auto", runtime_env=RUNTIME_ENV, logging_level="ERROR")
    print(json.dumps(ray.get(run.remote(), timeout=1500), indent=1))

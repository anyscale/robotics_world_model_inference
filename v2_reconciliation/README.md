# Reconciling the Stale Cloud — test-time compute for Physical AI, on Anyscale

**Fast reflexes on the robot; a slow world-model brain on Anyscale; and the hard part — reconciling the
cloud's always-late advice with the robot's real-time motion.**

Today's robot policies (VLAs) are dazzling **reflexes**: observation → action in one shot. A reflex can't
*deliberate*. A **world model** — a learned model of *what happens next given an action* — lets a robot
**imagine and search** for a good action (test-time compute for control, the robotics analogue of the
"think longer, answer better" unlock behind o1-style reasoning). That search is heavy, so it belongs in
the cloud.

But the moment the thinking lives in the cloud, its answer is **always stale**: computed for where the
robot *was* when it asked, delivered seconds later, after the fast reflex has moved on. Naively applying
stale advice is *worse* than ignoring it. The unlock is **reconciliation**:

> Fuse the live reflex with the delayed cloud correction so the robot ends up where the cloud's smarter
> plan intended — starting from where it actually is *now*. It's the driving-with-GPS move: keep driving,
> and steer toward the destination the (slightly behind) GPS implies.

**This is a Ray + Anyscale workload end to end.** The slow brain is a **Ray Serve** service; scaling it
(more GPUs → faster plans → *less staleness to reconcile*) is a config change, not a rewrite.

![architecture](images/architecture.png)

---

## What's here

| File | Role |
|---|---|
| `world_model_reconciliation.ipynb` | **Start here.** The end-to-end narrative: deploy the slow brain on Ray Serve → a real cloud plan + real latency → live reconciliation → the results → data-parallel scaling. Runs in ~90 s. |
| `vjepa_core.py` | Ray-free core: load V-JEPA 2-AC, encode → latent, roll latents forward (dynamics), CEM-plan a chunk, score energy. |
| `slow_brain_service.py` | The cloud slow brain as a **Ray Serve** `@serve.deployment` (HTTP `/plan`, model loaded once per GPU replica). |
| `world.py` | The `World` oracle — a stateful Ray actor standing in for the robot's physical reality (holds the true latent + pose). |
| `robot_client.py` | The on-robot **fast brain + reconciler**: async HTTP client, GPS-model reconciliation (`naive` vs `vector_add`). Numpy pose math, no GPU. |
| `sharded_planner.py` | **Data-parallel CEM** across GPUs — one plan's sample batch sharded over `ShardWorker` actors. |
| `p1_*.py` | Feasibility spike (reconciliation works). `p2_*.py` | Serve / reconciliation / sharding validation drivers. |
| `PLAN.md` | Full build plan, decisions log, and live status (start with the §0 HANDOFF). |

## Substrate

[**V-JEPA 2-AC**](https://github.com/facebookresearch/vjepa2) (`vjepa2_ac_vit_giant`, MIT) — Meta's open,
action-conditioned video world model (ViT-g encoder + action-conditioned predictor), loaded from
`torch.hub`. Planning is the **Cross-Entropy Method** in latent space: sample actions, roll them forward
through the predictor, keep the ones whose imagined latent lands closest to a goal latent. Pure PyTorch —
runs on the standard Anyscale base image plus a few prebuilt wheels.

## The core mechanic (vector addition)

The cloud's action `a_cloud` was right for the **snapshot pose** `p0`. By the time it lands, the robot has
drifted to `p_now`.
- **Naive** — apply `a_cloud` from `p_now`. It carries the drift; error grows with latency.
- **Vector-add reconciliation** — the cloud implied a **destination** `p0 + a_cloud`. Steer from `p_now`
  toward it: `correction = (p0 + a_cloud) − p_now`. The drift cancels. (Requires *retaining* `p0` until
  the matching reply arrives.)

## What it shows (measured on a 4× L4 dev cluster)

| | result |
|---|---|
| Serving latency | ~2.8 s / plan (compute-bound; HTTP + serialization ~30 ms) |
| Reconciliation | as cloud latency Δ grows, **naive** degrades toward the reflex-only baseline while **vector-add reconciled** stays low (pose-error, seed-averaged — see `images/p1_feasibility.png`) |
| Data-parallel CEM | one plan sharded across GPUs: **18.4 s → 4.3 s (4.28×) on 4 L4s** (`images/p2_shard_scaling.png`) — faster plan → smaller Δ |

> A 4× L4 dev cluster makes the latency gap deliberately stark; the numbers are illustrative. On
> production accelerators (H100 / H200 / B200) the **same** Ray orchestration drops plan latency toward
> sub-second — the curves shift down and to the left. The fast brain still runs in ms on the robot.

## Δ is a dial (the production consideration)

Staleness Δ is the ratio of cloud latency to the robot's control period — a design dial, not a law. Slow,
deliberate tasks → small Δ (a few ticks). Fast, dynamic tasks (balancing, contact) need 100–1000 Hz →
large Δ. You can shrink Δ (faster cloud via the shard lever / better GPUs) but never to zero — the cloud
is always slower than the reflex. So reconciliation is always needed, and matters *more* as Δ grows.

## Setup & run

**Weights** (cached on shared storage so every worker loads locally):
```bash
export TORCH_HOME=/mnt/cluster_storage/torch_hub
python -c "import torch; torch.hub.list('facebookresearch/vjepa2', trust_repo=True)"
# fix the upstream placeholder URL in src/hub/backbones.py:
#   VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"   (not http://localhost:8300)
curl -o $TORCH_HOME/hub/checkpoints/vjepa2-ac-vitg.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
```

**Run the notebook** top to bottom (deploys the slow brain on Ray Serve, ~90 s end to end). To rebuild it
from source: `python build_notebook.py`. To execute headless: `python run_nb.py`
(the environment's `jupyter nbconvert` is broken by an `lxml` packaging split; `run_nb.py` uses `nbclient`).

## References

- V-JEPA 2 (Meta AI, 2024–2025) — `facebookresearch/vjepa2`, MIT.
- Cross-Entropy Method for model-predictive planning.
- o1-style test-time compute (the reasoning analogue this brings to control).
- Control under latency: teleoperation / networked control; the autonomous-driving off-vehicle-compute split.

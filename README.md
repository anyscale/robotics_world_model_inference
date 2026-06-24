# robotics_world_model_inference

**The Slow & Fast Brains — test-time compute for Physical AI, on Anyscale.**

Today's robot policies (VLAs) are dazzling **reflexes**: image + instruction → action, in one shot.
But a reflex can't *deliberate*. Large language models crossed exactly this threshold with
**test-time compute** — the "think longer, answer better" unlock behind o1-style reasoning. This
template plants a flag on the robotics version of that unlock and shows the production architecture
for it on Ray + Anyscale:

> Give the robot a **world model** it can *imagine and search inside* to choose actions; run that
> search at scale in the cloud; and **distill** the result back into a fast policy that runs on the robot.

**Fast brain on the robot, slow brain in the cloud.** The cloud runs the expensive world-model
planning; the fast policy runs at the edge and pulls distilled updates.

> Thought-leadership / orchestration template. The point is the **pattern + Ray scale-out**, not
> publishable scaling numbers — those need bigger hardware than the 4× L4 dev cluster (see
> [Scaling on production hardware](#scaling-on-production-hardware)).

---

## The architecture

```
    ┌──────────────────────  THE CLOUD (Anyscale)  ──────────────────────┐
    │  V-JEPA 2-AC world model (1.3B)     Ray fans the search out:        │
    │  ┌───────────────────────┐          • DEEPER  (more compute/action) │
    │  │ imagine futures + CEM  │ × N GPUs • WIDER   (more robots at once) │
    │  │ search over actions    │                                         │
    │  └───────────────────────┘          mines (state → best action)     │
    │             └──────────── distill ──────────────┘                   │
    └───────────────────────────│────────────────────────────────────────┘
                  periodic download │ of the fast policy
    ┌───────────────────────────▼────────────────────────────────────────┐
    │  THE ROBOT:  fast distilled policy — milliseconds, no search         │
    └──────────────────────────────────────────────────────────────────── ┘
```

---

## What's here

| File | Role |
|---|---|
| `world_model_planning.ipynb` | **Start here.** The end-to-end narrative: world-model planning → why it's cloud-only → Ray scale-out (the two curves) → slow→fast distillation. |
| `slow_brain.py` | The cloud "slow brain": a `Planner` **Ray actor** that loads V-JEPA 2-AC once per GPU and plans actions by CEM search in latent space. |
| `PLAN.md` | Full build plan + narrative spine + live status log. |
| `README.md` | This file. |

## Substrate

[**V-JEPA 2-AC**](https://github.com/facebookresearch/vjepa2) (`vjepa2_ac_vit_giant`, MIT) — Meta's
open, **action-conditioned** video world model: a ViT-g encoder (~1012M) + an action-conditioned
predictor (~305M) that rolls latents forward given actions. Loaded straight from `torch.hub`. Planning
is the **Cross-Entropy Method**: sample candidate actions, roll them forward in latent space, score
against a goal latent, keep the best, repeat.

> **No custom image required.** This is pure PyTorch — it runs on the standard Anyscale base image
> plus a handful of prebuilt pip wheels. (An earlier draft targeted NVIDIA Cosmos Policy, which needs
> compiled deps the slim workers can't build; V-JEPA 2 sidesteps that entirely.)

---

## Setup

**1. Dependencies** (run as a bare `pip install` on the workspace so they register on *all* cluster
nodes — the Anyscale shim propagates them; `python -m pip` would be head-only):

```bash
pip install torch==2.6.0 torchvision==0.21.0 timm scipy einops
```

All prebuilt wheels — nothing compiles on the workers.

**2. World-model weights, cached on shared storage** so every worker loads the 11 GB checkpoint
locally instead of re-downloading. Point `TORCH_HOME` at a shared mount and pre-fetch once from a node
with internet:

```bash
export TORCH_HOME=/mnt/cluster_storage/torch_hub
python -c "import torch; torch.hub.load('facebookresearch/vjepa2','vjepa2_ac_vit_giant',trust_repo=True)"
```

> **Gotcha:** the upstream repo ships a placeholder weights URL (`http://localhost:8300`) in
> `src/hub/backbones.py`. Either edit `VJEPA_BASE_URL` to
> `https://dl.fbaipublicfiles.com/vjepa2`, or pre-place `vjepa2-ac-vitg.pt` in
> `$TORCH_HOME/hub/checkpoints/` (once cached, the URL is never contacted).

**3. Run** `world_model_planning.ipynb` top to bottom. It ships `slow_brain.py` to the workers via
`runtime_env={"py_modules": [...]}` and spins up one `Planner` per GPU.

---

## What it shows (measured on 4× L4)

| | result |
|---|---|
| Single planning action (light config) | energy 0.44, **2.4 s** on one L4 |
| **Production-strength** plan (400 × 10 × 2) | **~98 s/action**, 19 GB — ~thousands× too slow for on-robot |
| **Deeper** (test-time compute) | more compute → lower energy to goal (the "o1 curve") |
| **Wider** (fleet throughput) | ~**3.8×** more actions/sec across 4 GPUs (near-linear) |
| Slow → fast **distillation** | distilled policy acts in **~0.12 ms** vs 98 s of search |

The ~98 s/action figure is the whole argument: search is far too heavy for a robot's 10–50 ms control
loop, so it belongs in the cloud, scaled with Ray, and distilled down for the edge.

## Scaling on production hardware

Every number above is from a **4× L4 dev cluster** (~24 GB, ~120 bf16 TFLOPS) — modest inference GPUs
that make the latency gap deliberately stark. Production accelerators close most of it, with the **same**
Ray orchestration; the curves just shift down and to the left:

- **Faster GPUs** — H100 / H200 / B200 deliver ~10–30× the bf16/fp8 throughput → the ~98 s plan drops
  toward single-digit seconds.
- **More VRAM** (80–192 GB) → far larger CEM sample batches per forward → better plans at equal wall-clock.
- **Data-parallel search** — shard one plan's sample batch across GPUs with Ray, near-linear *within* an action.
- **Kernel-level wins** — fp8, `torch.compile`, FlashAttention compound for several more multiples.
- **The fast brain is unaffected** — it already runs in milliseconds and ships to the robot regardless.

The thesis is hardware-agnostic: **planning is elastic cloud compute; deployment is a cheap distilled policy.**

---

## References

- V-JEPA 2 (Meta AI, 2024–2025) — `facebookresearch/vjepa2`, MIT.
- Cross-Entropy Method for model-predictive planning.
- o1-style test-time compute (the reasoning analogue this brings to control).

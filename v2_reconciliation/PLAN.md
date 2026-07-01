# v2_reconciliation — Build Plan & Live Log

**One-liner:** A robot's fast reflex and a cloud "slow brain" world model think at different rates;
the cloud's advice is always *stale* by the time it lands. This template shows how to **reconcile**
the stale cloud correction with the live reflex — and shows that **Ray + Anyscale** is how you serve,
scale, and operate that slow brain in production.

**Audience / purpose:** Anyscale thought-leadership template. The deliverable sells **Ray Serve +
Anyscale** as the substrate for cloud world-model inference. The orchestration is the product; the
robotics is the vehicle.

**Hardware:** 4× NVIDIA L4 (24 GB each). We use **all four** and make GPU usage a config knob so the
story is "scale by changing a number, not a rewrite."

> ⚠️ **Persistence:** this cluster's disk (and any agent "memory") is wiped on shutdown. This file is
> the durable record. Keep §Decisions Log and §Status Log updated continuously. (User commits/pushes.)

---

## ☀️ MORNING SUMMARY (left overnight 2026-07-01)

**v2 is essentially complete and review-ready. Nothing broken, no decisions pending.** Overnight I did the
bounded P5 polish only (did NOT expand scope — Lever 2 stays deferred; git left to you):
- ✅ `architecture.png` — new schematic (fast reflex + reconciler on robot ↔ Ray Serve slow brain on
  Anyscale, stale corrected-destination link). Regenerate via `python make_architecture.py`.
- ✅ `README.md` — full v2 reconciliation story (dropped v1 Cosmos/distillation framing), architecture,
  measured results, Δ-as-dial, setup/run.
- ✅ Notebook: embedded the diagram + suppressed most stray logs. **Re-validated: runs CLEAN end-to-end in
  88 s** (budget 600 s), 20 cells, 4 charts embedded, output = narrative only. (3 benign lines remain:
  Serve "Started/ready" + one harmless raylet shutdown-retry — left intentionally.)

**Suggested next steps when you're back (all optional):** (1) open `world_model_reconciliation.ipynb` and
read it through / tweak prose; (2) decide whether to add a root-level README pointing at v2; (3) if happy,
commit; (4) revisit Lever 2 (fleet throughput) only if you want the "wider" scaling beat too. The full
build trail is in §Status Log below.

---

## 0. CURRENT STATE — HANDOFF (read this first if resuming cold)

**Where we are (2026-07-01):** P1 (reconciliation feasibility) ✅, P2 (Ray Serve architecture) ✅,
Lever 1 (data-parallel CEM) ✅. **Lever 2 (fleet throughput) SKIPPED by user — single robot for now.**
**NOW BUILDING: `world_model_reconciliation.ipynb`** (the deliverable, must run < 10 min).

**Env / cluster:** 4× L4 (idle). Deps intact (torch 2.6.0+cu124, ray 2.53.0, timm/scipy/einops).
Weights cache at `/mnt/cluster_storage/torch_hub` (11.76 GB `vjepa2-ac-vitg.pt` + repo). **If the disk is
wiped again, re-fetch:** `export TORCH_HOME=/mnt/cluster_storage/torch_hub`; `python -c "import torch;
torch.hub.list('facebookresearch/vjepa2', trust_repo=True)"`; edit `src/hub/backbones.py`
`VJEPA_BASE_URL` from `http://localhost:8300` → `https://dl.fbaipublicfiles.com/vjepa2`;
`curl -o $TORCH_HOME/hub/checkpoints/vjepa2-ac-vitg.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`.

**File inventory (v2_reconciliation/):**
- `vjepa_core.py` — Ray-free core: `WorldModelCore` (encode/step/cem_plan/energy). Used by everything.
- `slow_brain_service.py` — Ray Serve deployment `SlowBrain` (num_gpus=1, FastAPI `/plan`+`/health`,
  pickled bytes). `build_app(samples,cem_steps,rollout,num_replicas)`.
- `world.py` — `World` stateful oracle actor (build_problem/reset/step/snapshot) = "physical robot".
- `robot_client.py` — async one-in-flight HTTP client; GPS-model reconciliation (naive vs vector_add);
  numpy pose math, no GPU. `run_episode(...)`.
- `sharded_planner.py` — `ShardWorker` actors + `plan_sharded()`: data-parallel CEM across G GPUs.
- `p1_world.py` / `p1_run.py` / `p1_diag.py` / `p1_deepcloud.py` — P1 spike (feasibility). Reusable.
- `p2_serve_test.py` / `p2_run.py` / `p2_shard_test.py` — P2 validation drivers.
- Artifacts: `p1_feasibility.png` (money chart), `p2_shard_scaling.png` (data-parallel), `p2_reconciliation.png`.

**Measured facts (all on 4×L4):**
- Reconciliation WORKS (P1, pose-error, 4 seeds): naive climbs 0.31→0.42 with Δ; vector_add stays
  ~0.31–0.38 (below naive at every Δ≥4); reflex 0.477. Latent energy shows same but weakly (diffuse metric).
- Serving: **~2.77 s/plan** server compute, ~2.80 s round-trip, HTTP overhead ~30 ms (compute-bound).
- Real Δ: at 3 Hz control, 2.8 s ≈ **~9 control ticks** (per real measurement); at 20 ms loop ≈ ~140 ticks.
- Data-parallel CEM (Lever 1): 1 GPU 18.4 s → 2 GPU 9.0 s (2.04×) → 4 GPU 4.3 s (**4.28×**), near-linear.

**Key locked decisions:** substrate V-JEPA 2-AC + CEM; world = latent space (predictor = env oracle AND
cloud planner); reflex on small GPU; HTTP client; NO offline-distill thesis (online reconciliation is the
closer); first mechanic = vector addition (`p_target = p0 + a_cloud`, steer from p_now); **money chart =
pose-tracking-error vs Δ (primary) + energy (secondary)**; robot-client = GPS model (cloud returns a
corrected DESTINATION, robot steers toward it); Δ is a control-rate dial, demo uses moderate ~3 Hz → small
clean Δ, narrative treats Δ as a production consideration; NO "borg" wording; sell Ray+Anyscale as the
solution. Notebook budget < 10 min: precompute heavy sweeps (P1 chart, shard chart), keep live cells light
(deploy Serve, one plan, one short reconciliation episode). See §Decisions Log + §Status Log for full trail.

**Notebook plan (P4):** narrative reframe → deploy SlowBrain on Ray Serve (live) → one real plan + real
latency (live, ~2.8 s) → staleness + vector-add reconciliation (explain + one live episode, no misleading
live A/B) → **P1 money chart (precomputed)** → **data-parallel scaling chart (precomputed)** → production-
hardware / Δ-as-consideration → close. GPU budget live: SlowBrain(1)+World(1)=2 of 4.

---

## 1. Thesis (the flag we plant)

Today's robot policies (VLAs) are brilliant **reflexes**: observation → action in one shot. A reflex
can't *deliberate*. A **world model** lets the robot imagine futures and **search** for a good action —
test-time compute for control — but that search is heavy and belongs in the cloud.

The moment you put the thinking in the cloud, you inherit a hard problem the field hasn't solved:
**the cloud's answer arrives late.** It is computed for the state the robot was in when it asked, and
delivered seconds later, after the reflex has already moved on. Naively applying stale advice is
*worse* than ignoring it — it drives the robot toward where it *used to be*.

So the unlock is **reconciliation**: fuse the live reflex with the delayed cloud correction so the
robot ends up where the cloud's smarter plan intended, starting from where it actually is now. This is
the same control-under-latency problem mature teleoperated / autonomous-driving stacks already wrestle
with; we generalize it to a **learned world model** so it works for general-purpose robots.

**And this is a Ray + Anyscale workload end to end:** the slow brain is a **Ray Serve** service
(multi-GPU, autoscaling), the fleet is **Ray** fan-out of robot clients, and scaling the cloud — more
GPUs → faster plans → *less staleness to reconcile* — is a config change. **Anyscale is where the slow
brain lives.**

## 2. Substrate (reused from v1, verified)

- **World model:** `facebookresearch/vjepa2` → `vjepa2_ac_vit_giant` (encoder 1012M + action-conditioned
  predictor 305M, MIT). Pure PyTorch, loads from `torch.hub`, weights cached on shared storage
  (`TORCH_HOME=/mnt/cluster_storage/torch_hub`). No compiler wall (this is why v1 dropped Cosmos).
- **Planner:** Cross-Entropy Method (CEM) in latent space — sample actions, roll forward through the
  AC predictor, score `energy = L1(imagined_latent, goal_latent)`, keep best, refit, repeat.
- **Verified numbers (v1, 4× L4, bf16):** model load ~46 s; one predictor rollout @ batch 400 ≈ 3 s;
  full plan (400 samples × 10 steps × 2 rollout) ≈ **98 s/action**, peak 19.1 GB. Light config
  (64 × 4 × 1) ≈ 2.4 s. **Known softness:** a 1-step goal saturates CEM (energy barely moves) — we MUST
  use `rollout>1` and a far enough goal so energy has room to fall over a multi-step episode.
- **Everything lives in latent+pose space.** Encode the context and goal frames once; thereafter the
  predictor advances latents and `compute_new_pose(pose, action)` advances the 7-D end-effector pose.
  No renderer needed for intermediate states. The reconciliation arithmetic is in **pose/action space**
  (interpretable); the energy metric is in **latent space**.

## 3. Architecture — four components

| Component | Where it runs | Role |
|---|---|---|
| **Slow brain** | **Ray Serve** deployment, fans CEM across a pool of GPU actors (data-parallel, N configurable) | Full CEM plan for a snapshot. Smart but slow (seconds). The thing we serve & scale. |
| **World oracle** | GPU Ray actor (light; fractional GPU ok) | Ground-truth dynamics: advances the *true* `(z, pose)` given an action via the predictor. Stands in for physical reality. |
| **Reflex (fast brain)** | small GPU / fractional L4 | Distilled `latent → action` policy, one forward pass, ms-scale. Fast but weak. Models the on-robot accelerator. |
| **Reconciler** | with the robot client (cheap vector ops) | Fuses stale `a_cloud` with the live reflex. `naive` / `vector_add` / `blend`. |

**Robot client (HTTP, like the isaac-lab example):** holds current `(z, pose)` + a **history buffer**
keyed by `seq_id`. High-Hz loop: reflex picks action → World advances true state. Low-Hz: POST a
timestamped snapshot `(z, pose, z_goal, seq_id)` to the Serve endpoint (`requests.post`, pickled
payload), non-blocking. When a reply lands tagged `seq_id`, reconcile and fold in the correction.

## 4. The core mechanic — the "first example" (locked)

- At snapshot time the robot is at pose `p0`; it sends the snapshot, keeps `p0` in its buffer.
- Cloud returns `a_cloud` — the action that was right **for `p0`** — but Δ seconds late. By then the
  reflex has moved the robot to `p_now`.
- **Naive** = apply `a_cloud` from `p_now` → wrong, it was computed from a different start.
- **Vector-add reconciliation** = the cloud implied a *target* `p_target = compute_new_pose(p0, a_cloud)`
  ("where I should have ended up"); the corrective action is `p_target − p_now` — steer from where I am
  to where the cloud's advice intended. Requires retaining `p0` keyed by `seq_id`.
- Frontier (§8): staleness-weighted **blend** of reflex+correction, safety override (reflex wins when
  too stale / unsafe), and a **learned** reconciler. Possibly embodiment-specific. Framed as open research.

## 5. The money artifact

**Energy-to-goal vs. cloud latency (Hz)**, three lines from closed-loop episodes:

- **reflex-only** — flat, mediocre (never gets the cloud's smarts).
- **naive-apply-stale** — good at Δ≈0, **overshoots / destabilizes** as Δ grows.
- **vector-add reconciled** — stays low across the latency range.

This is the inversion of v1's "o1 curve": there, more compute → better. Here, *reconciliation is what
keeps a late-but-smart cloud from hurting you.* Supporting visual: a **pose-space trajectory plot**
(took / cloud-intended / reconciled) since poses are interpretable even though frames aren't.

## 6. Ray + Anyscale story (the sell — must be central)

- **Ray Serve = how you deploy world-model inference.** The slow brain is a `@serve.deployment`
  (`ray_actor_options={"num_gpus":...}`), model loaded once per replica, fronted by HTTP. Production
  pattern, mirrors `isaac-lab-gr00t-ray-serve`.
- **Lever 1 — within-action data-parallel (lower latency).** The Serve deployment shards one plan's CEM
  sample batch across a pool of GPU `Planner` actors and reduces. More GPUs → faster plan → **less
  staleness to reconcile.** `n_shards` is a config knob (1 → 4 on our L4s). This is the most on-thesis
  scaling lever.
- **Lever 2 — fleet throughput.** `.options(num_replicas=N)` + many robot clients as Ray tasks; the
  Serve proxy load-balances. Robots-served-per-replica scales near-linearly.
- **Anyscale** runs it elastically; autoscaling replicas; shared storage for weights. "Scale it = change
  a number." Tagline: *fast brain on the robot, slow brain on Anyscale.*

## 7. GPU plan (4× L4) — make it a knob

- **Core single-robot demo:** cloud sharded across 2–3 L4s; World oracle + reflex on the remaining GPU
  (fractional `num_gpus`), since both are light (one predictor forward / one MLP forward per tick).
- **Lever-1 beat:** sweep `n_shards` 1→4, show cloud plan latency drop ~linearly.
- **Lever-2 beat:** run multiple robot clients against the Serve fleet, show throughput.
- We can't max both levers at once on 4 GPUs — run the two scaling beats separately, bound aggressively,
  and label them as illustrative shapes (real curves need real clusters). Exact GPU packing = P2 detail.

## 8. Notebook spine (10 acts)

1. The reframe — cloud thinks, but its answer is stale. 2. Precedent — control under latency (teleop/AV),
generalized to a learned world model. 3. Substrate — latent world; reflex vs cloud; one cloud plan.
4. **Serve the slow brain** with Ray Serve; robot as HTTP client. 5. **Feel the problem** —
naive-apply-stale overshoots as latency grows. 6. **The fix** — vector-addition reconciliation.
7. **The money chart** — energy vs latency, 3 strategies. 8. **Scale on Ray/Anyscale** — Lever 1
(shard → lower latency) + Lever 2 (replicas → fleet). 9. **Frontier** — blend / safety / learned
reconciler, embodiment-specific, slow updates to reflex & world model. 10. Production-hardware note —
faster GPUs → higher cloud Hz → less staleness; same Ray orchestration.

## 9. Phases & verification gates

- **P0** — plan sign-off *(this doc)*.
- **P1 — FEASIBILITY SPIKE (critical go/no-go).** Latent closed loop with predictor as oracle. Confirm:
  (a) energy **decreases** as the robot nears goal over a multi-step rollout (needs far enough goal +
  `rollout>1` — see Risk 2); **(b) the three curves separate** — naive-stale visibly degrades with Δ,
  vector-add recovers. *If they don't separate, the demo is hollow — learn it here, cheaply, before
  building Serve/fleet.* Also: train the distilled reflex (fast, like v1) and confirm it's meaningfully
  weaker than full CEM.
- **P2** — Ray Serve deployment (HTTP, data-parallel CEM across GPU actors) + async HTTP robot client;
  verify round-trip, polling, real latency.
- **P3** — reconcilers (`naive`/`vector_add`/`blend`) + latency sweep → money chart + trajectory plot.
- **P4** — scaling beats (Lever 1 & 2) + notebook narrative + architecture diagram.
- **P5** — polish, honest caveats, production-hardware note, README.

## 10. Top risks

1. **Trivial reconciliation (the big one).** Env and planner share the predictor → the cloud may be
   "too right," so the three curves don't separate. Mitigations: staleness alone drives the error; make
   the reflex genuinely weak; optionally inject process noise so the world drifts. **P1 gates on the
   curves separating.**
2. **Flat energy curve.** v1 saw a 1-step goal saturate CEM. Use `rollout>1` + a far enough goal so
   energy has room to fall over an episode.
3. **Pose↔latent consistency.** Vector-add is in pose space; energy is in latent space. Verify the
   corrective action actually lowers latent energy (it should — action drives both).
4. **GPU budget (4× L4).** Cloud sharding + World + reflex + fleet all want GPUs. Keep World/reflex on
   fractional GPUs; run scaling beats separately.
5. **Serve + real latency in a notebook.** Real CEM is seconds; ensure the async HTTP loop and timeouts
   behave; inject extra Δ to sweep latency cheaply rather than only varying compute.

## 10b. Notebook runtime budget (HARD constraint, user)

**The final notebook must run end-to-end in under 10 minutes.** (The P1/offline feasibility sweeps may
take longer — they are batch artifacts, not the notebook.) Strategy to stay under budget (mirrors v1):
- **Precompute the heavy money-chart sweep offline** (latency × seeds × strategies) → saved `.npz` +
  `.png`; the notebook *loads & displays* it, does not re-run it live.
- **Live cells stay light:** one-time model load (~46 s) + Serve startup; one cloud plan; the
  vector-addition reconciliation on a *single* episode; a 2–3 point latency mini-sweep; a few HTTP
  round-trips for the architecture beat; tiny reflex distill (seconds).
- **Scaling beats** shown at small scale live with the honest "scale = change a number" framing.
- Budget sketch: load ~46 s + serve ~30 s + light demo ~2–3 min + scaling ~1–2 min + plotting < 10 min.

## 11. Intended file layout (v2_reconciliation/)

- `PLAN.md` *(this)* — plan + live log.
- `slow_brain_service.py` — Ray Serve deployment + GPU `Planner` actor pool (data-parallel CEM). Reuses
  v1's `WorldModel`/CEM logic.
- `world.py` — World oracle actor (ground-truth latent+pose dynamics).
- `reflex.py` — distilled fast policy (train + load).
- `reconcile.py` — `naive` / `vector_add` / `blend` reconcilers (pure functions).
- `robot_client.py` — HTTP client loop: reflex tick, snapshot POST, buffer, reconcile.
- `world_model_reconciliation.ipynb` — the narrative notebook (the deliverable).
- `architecture.png` — new diagram (reflex ↔ Serve slow brain over a latency'd wire).
- `README.md` — final wrap.

---

## Decisions Log (locked)

- **Substrate = V-JEPA 2-AC**, CEM in latent space (carried from v1; pure PyTorch, no compiler wall).
- **World = V-JEPA latent space**; predictor is both environment-oracle and the cloud's planning model.
  Reconciliation arithmetic in 7-D pose/action space; metric = latent L1 energy.
- **Cloud brain = full CEM**, served via **Ray Serve**, **data-parallel across multiple GPUs** (config
  `n_shards`, use all 4 L4s; easily scalable).
- **Reflex = distilled MLP on a small GPU** (not CPU — distilled models run on small accelerators).
  Reused as a *component* (the fast brain), not the thesis.
- **Client transport = HTTP** (pickled payloads), for realism — mirrors `isaac-lab-gr00t-ray-serve`.
- **Distillation flywheel from v1 = dropped as the thesis**; online reconciliation is the closer. (A
  slow offline update of reflex/world-model is mentioned only as §8.9 frontier.)
- **First reconciliation example = vector addition** (`p_target = p0 + a_cloud`; correct from `p_now`).
- **Money chart = energy-to-goal vs. cloud latency**, lines: reflex-only / naive-stale / reconciled.
- **No "borg" / internal-shorthand phrasing** in any deliverable.
- **Ray + Anyscale must be sold as the solution**, not used as backdrop.

## Robot-client design (LOCKED, user 2026-06-30)

- **Real-life model:** the cloud returns a **corrected destination** ("head toward `p0+a_cloud`"); the
  robot keeps acting and **steers toward that destination** (bounded steps) until the next cloud reply —
  the GPS analogy. NOT a rigid step-by-step chunk replay (that doesn't fill the long wait). The "fast
  brain" = the bounded tracking controller (runs in ms on-robot). naive anchors the destination at the
  drifted current pose; vector_add at the retained snapshot pose.
- **Δ is a dial set by control rate, not a law.** Cloud is always slower than the robot (heavy search),
  so staleness Δ≥1 always exists and must be reconciled; slowing the robot shrinks Δ but you don't want
  to slow to cloud speed (you'd lose the reflex's responsiveness/safety). **Demo uses a moderate rate
  (~3 Hz, tick_dt≈0.35 s) so the real 2.8 s cloud latency → Δ≈8** — small, clean, REAL (emergent from
  measured latency, not faked), and notebook-fast. Narrative discusses Δ as the production consideration
  (faster robot / faster cloud shift it; reconciliation matters more as Δ grows).
- **Money chart** = reuse P1's fast in-process Δ-sweep (pose-error primary). **P2 served loop** is the
  LIVE architecture demo: one short naive-vs-vector_add episode pair against the real Serve endpoint
  (async HTTP, real latency logged). Client does pose math in numpy (rotation dims unused) — no GPU.

## Open questions (resolve as we build)

- Exact GPU packing across 4 L4s for the core loop vs. the two scaling beats (P2).
- How to inject/measure latency cleanly: real CEM seconds vs. artificial Δ sweep (likely both).
- Whether the `blend`/learned reconciler ships as a runnable cell or stays a described frontier.
- Goal/episode design that makes energy fall enough for a legible curve (P1).

## Status Log (update continuously)

- **2026-06-30** — v1 archived to `v1_distillation/`; `v2_reconciliation/` created. Studied
  `isaac-lab-gr00t-ray-serve` Serve pattern (deployment + FastAPI ingress, model loaded in `__init__`,
  HTTP pickled payloads, `num_replicas` scaling, ~6–8 Hz client polling of chunked actions). Plan
  drafted and revised per user (no borg; sell Ray/Anyscale; reflex on small GPU; use all 4 L4s; cloud
  multi-GPU; HTTP client; keep this doc durable). **P0 signed off.**
- **2026-06-30 — disk wipe recovery.** Cluster disk was wiped (the persistence risk, realized):
  `/mnt/cluster_storage/torch_hub` cache gone. Re-fetched: cloned vjepa2 repo via `torch.hub.list`,
  fixed the placeholder weights URL in `src/hub/backbones.py` (`localhost:8300` →
  `https://dl.fbaipublicfiles.com/vjepa2`), downloaded `vjepa2-ac-vitg.pt` (11,760,743,310 bytes) into
  `hub/checkpoints/`. Deps intact (torch 2.6.0+cu124, ray 2.53.0, timm/scipy/einops). 4× L4 idle.
  **RE-SETUP RECIPE if wiped again:** `export TORCH_HOME=/mnt/cluster_storage/torch_hub`;
  `python -c "import torch; torch.hub.list('facebookresearch/vjepa2', trust_repo=True)"`; fix the URL;
  `curl -o $TORCH_HOME/hub/checkpoints/vjepa2-ac-vitg.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`.
- **2026-06-30 — P1 spike built & running.** `p1_world.py` (SpikeWorker actor: world-oracle + full-CEM
  cloud + tiny-CEM reflex + naive/vector_add reconcilers, latency simulated by delayed application) +
  `p1_run.py` (driver: fans episodes across 4 L4s, sweeps Δ, plots `p1_feasibility.png`). Key upstream
  facts found: CEM actions are (dx,dy,dz,0,0,0,gripper) — translation+gripper only; `compute_new_pose`
  xyz is pure addition; `poses_to_diff(now,target)` is the exact corrective-action operator. Traj is
  only 2 frames → goal built by rolling predictor `goal_steps` with a fixed +x action.
- **2026-06-30 — P1 runs 1–3 + diagnostic. KEY FINDINGS:**
  - Run 1 (quick, one-shot correction): cloud beats reflex, naive degrades with Δ, but vector_add did
    NOT stay flat — one-shot correction at high Δ is a huge OOD action (>>maxnorm). Switched to bounded
    target-tracking (Scheme C). Run 2 (full): noisy, non-monotonic, vector_add ≈/worse than naive.
  - **Diagnostic (`p1_diag.py`) isolated the cause:** (a) **substrate is SANE** — oracle replay of the
    goal-building action drives energy monotonically to 0.0 in +x/+y/+z (`0.50→…→0.0`); energy tracks
    the goal and there's real range. (b) **greedy 1-step CEM is a NOISY, myopic teacher** — plateaus at
    0.14 (+x) / 0.20 (+y, even rises), picks max action in all axes not just toward goal. The cloud
    wasn't actually "smart," so naive-vs-vector_add was buried in CEM noise. (c) my loop produced little
    sustained DRIFT (robot reached tiny cloud target then idled) → naive≈vector_add.
  - **FIX (next):** (1) cloud = multi-step CEM (`rollout>1`, deeper) so it plans a horizon, not a greedy
    step (on-thesis: test-time compute). (2) reflex BRIDGES between low-Hz cloud updates → sustained
    drift that makes naive wrong / vector_add right. Average over seeds to clear residual noise.
    Linchpin check first: does deep-rollout CEM drive energy near 0? If not, isolate feasibility with a
    near-oracle cloud (and say so), since the reconciliation claim is separable from CEM quality.
  - **Status: P1 in progress — substrate validated; redesigning cloud (deeper) + loop (real drift).**
- **2026-06-30 — P1 deep-cloud linchpin PASSED + chunk loop.** `p1_deepcloud.py`: a multi-step CEM
  plan (rollout 2–4) tracks the oracle descent almost exactly (rollout=4 → energy matches oracle
  step-for-step, peak ~8 GB). The earlier "greedy plateau at 0.14" was greedy *replanning* wandering,
  not a weak planner. So the cloud, planning a `rollout≈4` CHUNK, is a clean strong teacher. NOTE:
  rollout=8 OOMs on L4 (sequence too long) — cap rollout ≤4, shrink samples as rollout grows; set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Rebuilt `run_episode` as **chunk-based MPC**: cloud plans a rollout-chunk from the snapshot;
    arrives `delta` late; `naive` re-applies the chunk anchored at the drifted current pose (carries
    the offset), `vector_add` tracks the chunk's ABSOLUTE targets anchored at the snapshot pose
    (drift cancelled via `poses_to_diff`). Quick single-seed run was noisy (endogenous drift, not
    controlled). Added **exogenous drift** (constant off-axis +y push every tick, identical across
    strategies, unseen by planner/reconciler) so the staleness penalty is clean & monotonic: naive
    carries ~Δ·drift, vector_add ~1 tick residual → gap ∝ Δ. Seed-averaged (3 seeds).
  - Full sweep w/ exogenous drift + 2 seeds: directionally right but latent-energy signal weak/noisy
    (effect ~0.01–0.02 ≈ seed noise). Root cause: latent energy is a *diffuse* function of pose, so
    a pose offset only weakly moves energy. Also learned rollout=4 CEM is ~100s/episode (too slow for
    sweep AND notebook) → dropped to rollout=2, 32×4 CEM, cloud every 4 ticks → ~13s/episode.
- **2026-06-30 — P1 GO ✅.** Added an EXACT, world-model-independent metric — **pose-tracking error**
  (xyz distance to the goal pose) — as the primary signal, with latent energy secondary. Strengthened
  run (drift 0.02, 4 seeds, 52 episodes, ~3.5 min, live-logged). RESULT (pose error):
  - naive climbs 0.313 (Δ=0) → 0.418 (Δ=8,10) toward reflex baseline 0.477; vector_add stays ~0.31–0.38,
    **below naive at every Δ≥4** by ~0.05–0.07; coincide at Δ=0–2 (no staleness). Clean money-chart shape.
  - Latent energy (secondary) shows same direction but weak/overlapping — diffuse metric. So the NOTEBOOK
    leads with **pose error (crisp)**, shows energy as corroborating world-model-native context.
  - **DECISION:** money chart = pose-tracking-error vs Δ (primary) + energy (secondary). Cloud config
    rollout=2 / 32×4 CEM / ~13s per episode is notebook-viable. Feasibility CONFIRMED — proceed to P2.
  - Spike files: `p1_world.py` (SpikeWorker), `p1_run.py` (sweep+plot), `p1_diag.py`, `p1_deepcloud.py`,
    `p1_feasibility.png`. Live-logging via `ray.wait` is now standard (see also user feedback memory).
  - **Status: P1 COMPLETE. Next: P2 — Ray Serve deployment (HTTP) + data-parallel CEM across GPUs + async robot client.**
- **2026-06-30 — P2 serving path WORKS ✅.** Built `vjepa_core.py` (Ray-free core: encode/step/cem_plan/
  energy), `slow_brain_service.py` (Ray Serve `@serve.deployment` num_gpus=1, FastAPI `/plan` + `/health`,
  pickled bytes, model+warmup in `__init__`, `build_app(samples,cem_steps,rollout,num_replicas)`),
  `world.py` (stateful World oracle actor: build_problem/reset/step/snapshot — holds true z+pose+goal on
  GPU). `p2_serve_test.py` deployed it and POSTed 3 real snapshots:
  - **server compute ~2.77 s/plan; round-trip ~2.80 s; HTTP+pickle overhead ~30 ms (negligible — it's
    compute-bound).** Serve replica init ~30 s (model load). chunk shape (rollout,7) returned fine.
  - **Real Δ is large:** 2.8 s plan ÷ ~20 ms control tick ≈ **~140 ticks of staleness** — ~14× the Δ≤10
    swept in P1. So the client loop must map real ms→Δ and sweep a realistic (large) Δ; reconciliation
    matters MORE at this scale. This is the honest production framing for the notebook.
  - **Status: P2 — serving validated. Next: async robot-client reconciliation loop over real HTTP
    (real round-trip → Δ), then data-parallel CEM sharding lever, then fleet throughput.**
- **2026-06-30 — P2 robot-client (GPS model) ARCHITECTURE VALIDATED ✅.** `robot_client.py` (async
  one-request-in-flight HTTP client; cloud returns corrected destination; robot steers toward it with
  bounded steps; naive anchors at drifted current pose, vector_add at snapshot pose; numpy pose math,
  no GPU) + `p2_run.py` (deploy Serve + World + naive/vector_add episodes). FIRST run had a bug: fired
  requests every 6 ticks without waiting → backlog on single replica → vector_add inherited a 7s queue
  and only 1 reply (invalid A/B). FIXED to one-request-in-flight + per-episode pools + drain. Fair rerun:
  both episodes Δ=9 ticks (from real ~2.78s latency at 3Hz), 2 replies each. RESULT: vector_add was
  AHEAD mid-episode (tick19 0.299 vs naive 0.376, right after a stale correction) but ended slightly
  behind (0.397 vs 0.350) — **single-episode noise-dominated (only 2 corrections + noisy 32×4 CEM)**,
  NOT a bug (same reconciliation logic P1 validated). 
  - **DIVISION OF LABOR (firm):** P1 (controlled, seed-averaged) = the statistical money chart;
    P2 served loop = LIVE architecture + real-latency demo (show it run + reconciliation steering back
    after a stale reply), NOT a live A/B (too noisy, could mislead).
  - Real-world lesson captured: firing faster than the cloud replies → queueing latency blowup
    (saw 2.8→7.3s) → motivates the scaling levers. Files: `slow_brain_service.py`, `world.py`,
    `vjepa_core.py`, `robot_client.py`, `p2_run.py`, `p2_reconciliation.png`.
  - **Status: P2 architecture DONE. Next: the Anyscale scaling SELL — Lever 1 (data-parallel CEM
    sharding across the 4 L4s → lower plan latency → smaller Δ) and Lever 2 (num_replicas → fleet
    throughput). Then P4 notebook.**
- **2026-07-01 — LEVER 1 (data-parallel CEM) DONE ✅ — near-perfect scaling.** `sharded_planner.py`:
  `ShardWorker` actors (each holds a model copy + `build_problem`/`set_context`/`roll_and_score`) +
  `plan_sharded()` coordinator — one CEM plan, sample batch split across G GPUs; each shard rolls its
  candidates and returns tiny (action,score) arrays; coordinator does GLOBAL top-k + refit each step
  (latents stay on-GPU, only small vectors cross the wire; semantics match upstream CEM). `p2_shard_test.py`
  measured plan latency for a fixed 128-sample / 6-step / rollout-2 budget:
  - **1 GPU: 18,415 ms (1.00×) · 2 GPU: 9,036 ms (2.04×) · 4 GPU: 4,302 ms (4.28×)** — near/super-linear
    (smaller per-GPU batches are more efficient), tiny rep variance. Chart: `p2_shard_scaling.png`.
  - Thesis payoff: sharding cuts plan latency → cuts staleness Δ (at 3 Hz, 18.4 s≈55 ticks → 4.3 s≈13
    ticks). Same orchestration, config knob; production HW → sub-second. This is the headline Anyscale sell.
  - **Status: Lever 1 DONE. Next: Lever 2 (fleet throughput via num_replicas), then P4 notebook assembly.**
- **2026-07-01 — LEVER 2 SKIPPED (user: single robot for now).**
- **2026-07-01 — P4 NOTEBOOK BUILT + VALIDATED ✅.** `world_model_reconciliation.ipynb` (19 cells)
  assembled via `build_notebook.py` (nbformat). NOTE: `jupyter nbconvert` is BROKEN in this env
  (`lxml.html.clean` moved to `lxml_html_clean`) — execute via `nbclient` instead: `python run_nb.py`
  (executes cell-by-cell with real-time per-cell logging, saves outputs in place). **Executed CLEAN
  end-to-end in 89 s** (budget 600 s): connect(3s) → Serve deploy(38s) → World+goal(31s) → one real
  plan(3s) → live vector_add reconciliation episode(10s) → precomputed P1 money chart + shard chart(0s)
  → cleanup(3s). Live GPU use = SlowBrain(1)+World(1). Narrative: reframe → GPS analogy → substrate →
  serve slow brain → real plan+latency → vector-add reconciliation (live) → P1 money chart (precomputed)
  → data-parallel scaling (precomputed) → Δ-as-dial/production framing → close. No "borg"; sells Ray+Anyscale.
  - **Status: P4 DONE (notebook runs clean < budget). REMAINING (P5 polish): README refresh (point at v2
    notebook), optional architecture diagram (v2 has none yet), review rendered plots look good, final
    prose pass. Consider committing (user handles git).**
- **2026-07-01 — P4 notebook OUTPUT CLEANED.** Was drowning in worker logs (Ray Serve INFO, per-CEM-step
  "new mean", FutureWarnings). Fixed: `ray.init(log_to_driver=False)` in the notebook + `warnings.filter`
  + silence `utils.mpc_utils` logger in `WorldModelCore.__init__`. Re-executed clean in 89 s. Driver
  output now exactly the narrative (GPUs, Serve health, cloud plan+latency, reconciliation tick replies
  Δ≈9 with pose_err 0.367→0.258, released). Charts embedded. Two trivial cosmetic lines left (2 Serve
  INFO from driver-side serve.run; 1 harmless raylet graceful_shutdown warning in cleanup cell).
  Notebook = `world_model_reconciliation.ipynb`; build via `python build_notebook.py`, execute via
  `python run_nb.py` (NOT nbconvert — broken here).
  - **Status: NOTEBOOK DONE & VALIDATED. Remaining = P5: README refresh + optional arch diagram + final prose.**

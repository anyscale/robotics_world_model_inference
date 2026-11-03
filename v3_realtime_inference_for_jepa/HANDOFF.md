# HANDOFF — v3 real-time inference for JEPA world models (keep this file current)

Owner: Ian Jordan (Anyscale robotics). If you are a new agent picking this up, **read all of this first**.
If the cluster restarts, all agent memory is gone. This file plus `results/` is the record.

## House rules (from Ian, verbatim intent)
- **Never `git add` / `commit` / `push`** — Ian manages git.
- **Do not change the base image's Python deps.** Another repo (`/home/ray/default/robolab_bimanual`) depends on them.
  - Extra packages go through Ray `runtime_env={"pip": [...]}` (a per-job virtualenv). See `jepa_rt/__init__.py: RUNTIME_ENV`.
  - **Trap:** on this Anyscale workspace, a bare `pip install` (even with `--target`) appends to
    `/mnt/cluster_storage/.anyscale/requirements.txt`, which installs into every node's base env. If you ever run
    pip, check that file afterwards. It should contain only `reportlab` and `pypdfium2` (those predate this project).
  - `/mnt/cluster_storage/jepa_rt/pylib` holds timm / python-docx / lxml installed with `--target` (driver-side use
    only, via `sys.path`). Not on any default path.
- **Don't crash the cluster.** A100s are scarce; if it goes down, Ian may wait days to a week for new ones.
  - Use memory-capped batch sizes (chunk ≤ 400 candidates on one 40 GB A100).
  - Always use timeouts.
  - Never run heavy work on the head node (it's CPU-only); all GPU work goes into Ray tasks/actors/Serve replicas.
  - Shut down Serve apps when done.
- **The code will be given to blog readers.** Keep it clean, self-contained and readable: no hard-coded hacks, and comments that explain *why*.
- **Goal: sell Ray on Anyscale as the best way to do this.** Every number must still be measured and reproducible. Credibility is the pitch.

## The project
- Omar's draft blog post (Ian's manager) is in `../Real-Time JEPA World Models.pdf` / `.docx`.
  - V-JEPA 2-AC planning with CEM: 800 candidates × 10 rounds, keep 80. It takes ~16 s/action on one RTX 4090 per the paper.
  - Proposed design: two Ray Serve deployments. `WorldModel` is the GPU tier, with one replica per GPU and `max_ongoing_requests=1`. `Planner` runs the CEM loop and fans each round out over the replicas.
  - His numbers are **modeled** (0.6 s on 32 H100s). Our job is to **implement it and measure it** on 8× A100-40GB, as a notebook.
  - Then update the .docx with the real results, writing them **into a copy** and leaving the original untouched.
- Ian's previous tutorial (v2, not to be modified) is `../v2_reconciliation/` (fast reflex on robot + slow cloud brain + reconciliation).

## Environment
- Cluster: CPU head node + **one 8× A100-SXM4-40GB worker** (`10.0.168.227`). Ray 2.58, torch 2.10+cu128, Python 3.12.
- V-JEPA 2 code + checkpoint on shared storage (torch.hub layout), `TORCH_HOME=/mnt/cluster_storage/jepa_rt/torch_hub`:
  - `hub/facebookresearch_vjepa2_main` is a git clone at `204698b45b3712590f06245fbfba32d3be539812`.
  - `hub/checkpoints/vjepa2-ac-vitg.pt` is 11.7 GB, from `https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`.
  - The upstream hub entry point's checkpoint URL points at `localhost`, so `jepa_rt/model.py` builds with `pretrained=False` and loads the URL above itself.
- Test data: `hub/facebookresearch_vjepa2_main/notebooks/franka_example_traj.npz` (context frame → goal frame).

## Code layout (this dir)
| Path | What |
|---|---|
| `Dockerfile` | Custom Anyscale image for a fresh cluster: FROM anyscale/ray:2.58.0-py312-cu128, torch 2.10 cu128, timm etc., V-JEPA 2 code pinned at /opt/vjepa2 (`VJEPA2_REPO_DIR`). Not test-built (no docker here; all pinned versions verified to exist on PyPI / the PyTorch index). |
| `jepa_rt/__init__.py` | Paths (`TORCH_HOME`, `VJEPA2_REPO_DIR`, `EXAMPLE_TRAJ`), `RUNTIME_ENV` (adds pip timm only if the image lacks it), `ensure_assets()` (fetch the pinned code if missing + the checkpoint once) |
| `jepa_rt/model.py` | `WorldModelCore`: load, `encode(frame)`, `score(z, pose, z_goal, actions[S,H,7]) -> energy[S]`. `fast=True` = bf16 + KV cache across rollout steps + flash SDPA + GPU pose update; `fast=False` = reference math |
| `jepa_rt/cem.py` | `CEMDistribution`: sample / refit / plan / shift (warm start); matches Meta's `mpc_utils.cem` rules |
| `jepa_rt/serve_app.py` | Ray Serve app: `WorldModel` (GPU tier, round-robin router, compile re-enabled) + `Planner` (CEM fan-out, stateless warm start). `build_app(n_gpus)` |
| `video/` | Blog video. `capture_trace.py` (GPU, ~3 min: one blocking + one non-blocking episode with per-round detail → `trace.json`), `render_video.py` (CPU, ~2.5 min → `realtime_jepa_planning.mp4`, 1920×1080, 32 s). Copy in `/home/ray/default/Ian_review/jepa_realtime/`. |
| `scripts/check_fast_path.py` | 1 GPU: fast vs reference check + Meta's unmodified CEM timing |
| `scripts/bench_compile.py` | 1 GPU: torch.compile speedup |
| `scripts/bench_scaling.py` | Serve app at 8/4/2/1 GPUs -> results/03_scaling.json |
| `scripts/bench_closed_loop.py` | closed-loop episodes (world model as simulator), cold vs warm-started CEM -> results/04_closed_loop.json |
| `results/*.json` | every measured number, saved as soon as it's measured |

## Results so far (all on A100-40GB, settings 800×10, keep 80, horizon 2)
- Meta's reference `cem()`, unmodified, fp32, one A100: **81.9 s / action**. The paper's 16 s is on a 4090 with its own settings; we don't know theirs exactly.
- Fast `score` matches the reference: Pearson 0.9999, max |Δenergy| 4e-4 on an energy range of 0.45–0.49.
  - Fast `score`, one round of 800: **2.79 s** (→ ~28 s/action on 1 GPU, 2.9× over the reference).
  - 100 candidates: 0.37 s, i.e. near-linear in batch size, so sharding across GPUs should scale well.
- torch.compile on the block step: 0.372 → 0.272 s per 100 candidates (results/02).
- **Serve app, 8 A100s: 2.87 s per full 800×10 plan** (28.5× vs 81.9 s reference). Round = 282 ms (272 GPU + ~10 ms Ray fan-out). Encode 52 ms. (results/03a)
- Two Serve gotchas found + fixed (worth a paragraph in the blog):
  1. The default power-of-two router sometimes put 2 shards on 1 replica (with `max_ongoing_requests=1`), doubling that round. Fix: `RequestRouterConfig(request_router_class="ray.serve.experimental.round_robin_router:RoundRobinRouter")`, giving 0 collisions in 50 rounds.
  2. Serve replica processes have `TORCHDYNAMO_DISABLE=1` in their env, so torch.compile is a silent no-op. Fix: `os.environ.pop("TORCHDYNAMO_DISABLE")` in `WorldModel.__init__` before importing torch. (It's not clear yet whether Ray or Anyscale sets it. Verify before claiming in the blog.)

## Plan / status
1. [x] Env, weights, fast path, single-GPU baseline.
2. [x] Ray Serve app (`jepa_rt/serve_app.py`): `WorldModel` (1 GPU/replica) + `Planner` (CEM, fan-out via
   DeploymentHandle). The context latents go out once per plan with `ray.put`, not on every round. Measure plan latency at 1/2/4/8 replicas and compare with Omar's formula.
- Scaling (results/03_scaling.json): 1/2/4/8 GPUs = 21.4 / 10.8 / 5.5 / 2.86 s per plan (7.5x on 8). Identical plans at every GPU count (same seeds) → sharding changes speed, not the answer. The blog's latency formula, calibrated on our 1-GPU round, predicts 2.89 s at 8 GPUs (measured 2.86).
- Closed loop (results/04_closed_loop.json): task = reach a goal 21.6 cm away (made reachable by rolling a known action through the model).
  - All strategies end at ~3.5–4 cm, which looks like the model's precision floor.
  - Each step costs: cold 10 rounds 2.83 s; 3 rounds (cold or warm) 0.855 s; warm 2 rounds 0.57 s.
  - **Honest finding:** 3 cold rounds do as well as 3 warm rounds on this task. The lever is fewer rounds; warm start isn't decisive here. Don't oversell warm start.
- Non-blocking (results/05_nonblocking.json): robot takes 1 s per action, 4 mm execution noise, 3 rounds.
  - Blocking: 1.86 s per step, 25.3 s per episode.
  - Non-blocking: **1.04 s per step**, 15.3 s per episode.
  - Same final accuracy.
- Figures: `scripts/make_figures.py` → images/latency_ladder.png, scaling.png, closed_loop.png.

3. [x] Warm-started CEM (shift the previous plan, fewer rounds) and non-blocking control (plan t+1 while t executes). Measure the steps/s the robot sees and check plan quality.
4. [x] torch.compile on predictor + encoder (in the default path). [ ] Not done: fleet batching (several robots sharing one GPU tier), CUDA graphs, quantization, KubeRay manifest (the original draft mentioned one; removed from the docx copy's "Source" line because we didn't write one).
5. [x] Notebook `realtime_jepa_planning.ipynb` (executed with outputs, ~4 min) + README + figures (`images/`).
6. [x] `../Real-Time JEPA World Models (measured results).docx`, a copy of Omar's draft with measured results. The original is untouched.
   - Edited: §04 adds a "What we learned" paragraph and updates "Source"; §05 is rewritten as measured, with Figs 5–6; §06 gets "Measured:" notes under A and B (plus Fig 7) and "Implemented:" under E; the §07 takeaway sentence and the disclaimer are updated.
   - Not changed: the byline, §01–03, and the Fig 3/4 captions (≈8.0 s / ≈0.6 s modeled).
   - Regenerate with `PYTHONPATH=/mnt/cluster_storage/jepa_rt/pylib python scripts/update_blog_docx.py` (reads the original and results/, then writes the copy). The notebook source is `scripts/build_notebook.py` (rebuilds the .ipynb **without outputs**; re-execute with `jupyter nbconvert --execute --inplace`).

## Log
- 2026-11-03 13:52 UTC: setup done; baseline + fast-path check (results/01_single_gpu_baseline.json).
- 2026-11-03 16:30 UTC: notebook declared done by Ian. Added video/ (capture + render). `Planner.plan(detail=True)` is an opt-in trace (default path unchanged).
- 2026-11-03 15:40 UTC: added Dockerfile + `ensure_assets()`; the model now loads from the local pinned V-JEPA 2 repo. Notebook re-executed clean with identical numbers.
- 2026-11-03 15:05 UTC: notebook executed clean (exit 0), README, docx copy written. No Serve apps running; base env untouched (requirements.txt = reportlab, pypdfium2; `import timm` fails in base python as intended).
- 2026-11-03 14:45 UTC: scaling, closed-loop, non-blocking done + figures. Next: notebook, README, docx copy.
- 2026-11-03 14:10 UTC: Serve app works, 2.87 s/plan on 8 GPUs. Scaling sweep running (results/03_scaling.log/json); next: closed-loop warm start.

# Real-time planning with a JEPA world model on Ray

Companion code for the blog post *Real-time inference for JEPA world model architectures with Ray*.

[V-JEPA 2-AC](https://github.com/facebookresearch/vjepa2) (Meta) can steer a robot arm toward any goal image with no task-specific training. It chooses each action by *searching*: it imagines thousands of possible futures with its world model and keeps the one that lands closest to the goal. That search is slow. The paper reports 16 s per action on an RTX 4090, and Meta's released planner takes 81.8 s per action on an A100 with the settings below.

This repo implements the search as two composed **Ray Serve** deployments and measures it end to end on **one 8× A100-40GB node on Anyscale**:

![latency ladder](images/latency_ladder.png)

| | s / action | vs. reference |
|---|---|---|
| Meta's reference planner, 1× A100, fp32 | 81.8 | 1× |
| Optimized model (KV cache, bf16, `torch.compile`), 1× A100 | 21.4 | 4× |
| **Ray Serve fan-out, 8× A100** | **2.86** | **29×** |
| + 3 CEM rounds per step instead of 10 (same accuracy on our task) | **0.86** | **96×** |
| + non-blocking control: plan the next step while the arm moves | robot never waits (1.04 s steps for a 1 s motion) | |

Search settings throughout: 800 candidates × 10 rounds, keep the best 80, horizon 2. Every number is measured, and the raw data is in `results/`.

**Why 81.8 s and not 16 s?** We ran Meta's released planning code as-is (fp32, 800 candidates × 10 rounds, a two-step horizon). The paper's 16 s likely reflects different settings or precision. Every speedup here is measured against our own 81.8 s baseline on the same hardware.

## How the search works, and where Ray comes in

Planning one action uses the Cross-Entropy Method (CEM). Each **round** samples 800 candidate action sequences, rolls every one forward through the world model, scores how close its imagined outcome lands to the goal, and refits the sampling distribution to the best 80. **10 rounds** run per action. The robot executes the first action of the final plan and then re-plans from a fresh camera frame (receding-horizon control, as in the paper).

- **Within a round, the 800 candidates are independent.** That is the part we spread across GPUs: each Ray Serve replica scores its share, one shard per GPU.
- **Across rounds, the work is sequential.** Each round samples from the distribution the previous round refit, so the 10 rounds set a floor that more GPUs can't remove. Fewer rounds per step and non-blocking control attack that floor.

![architecture](images/architecture.png)

- **`WorldModel`** is the GPU tier: one replica per GPU, with the model resident between calls, taking one shard at a time.
- **`Planner`** is the driver. It runs the CEM rounds, fans each round out through a `DeploymentHandle` with `asyncio.gather`, gathers the scores and refits, and serves the HTTP `/plan` endpoint the robot calls.
- Scaling is one parameter: `build_app(n_gpus=W)`.

Ray's share of each round (fan-out, gather, refit) is about 6–8 ms of a ~280 ms round. On 1, 2, 4 and 8 GPUs the plans are identical for the same seed: sharding changes the speed, not the answer.

## Watch it

[`video/realtime_jepa_planning.mp4`](video/realtime_jepa_planning.mp4) (32 s) replays one live run of blocking vs. non-blocking control on the 8 GPUs, in real time. It shows:

- the gripper path;
- the 800 imagined end positions of each CEM round, with the 80 kept in blue;
- distance to goal over time;
- a timeline of the robot, the planner's rounds and all 8 GPUs.

Non-blocking control finishes the same 12 moves in 15.3 s instead of 25.4 s. The video is also embedded in the notebook.

## Run it

**Start here:** `realtime_jepa_planning.ipynb`. It deploys the service on 8 GPUs, calls it over HTTP as a robot would, and runs a live non-blocking control episode. Execution takes about 4 minutes, plus a one-time ~12 GB download.

**On a new Anyscale cluster:** build the `Dockerfile` in this directory as a custom image (e.g. `anyscale image build -f Dockerfile -n jepa-realtime`), start a workspace on it with 8 GPUs (≥ 40 GB each), and run the notebook.

- The image pins everything we measured with: Ray 2.58, Python 3.12, CUDA 12.8, torch 2.10, timm, and Meta's V-JEPA 2 code at a fixed commit.
- The notebook's first cell calls `jepa_rt.ensure_assets()`, which downloads the 11.7 GB checkpoint once to shared storage (`/mnt/cluster_storage`), where every GPU reads it.
- Fewer GPUs also work: pass the count to `build_app(n_gpus=...)`.

**Without the custom image:** a stock `anyscale/ray` image with torch also works. `ensure_assets()` then fetches the pinned V-JEPA 2 code to shared storage, and Ray's `runtime_env` installs `timm` into a per-job virtualenv (see `jepa_rt/__init__.py`), so the cluster image is never modified.

## Reproduce every number

Each script writes JSON to `results/`; `scripts/make_figures.py` turns that into `images/`. Times are on 8× A100-40GB.

| Script | Measures | GPUs | Time |
|---|---|---|---|
| `scripts/check_fast_path.py` | Fast path vs reference (r = 0.9999); Meta's unmodified planner, 81.8 s | 1 | ~6 min |
| `scripts/bench_compile.py` | `torch.compile` speedup (1.35×) | 1 | ~2 min |
| `scripts/bench_scaling.py --gpus 8 4 2 1` | Plan latency at 1/2/4/8 GPUs | 8 | ~10 min |
| `scripts/bench_closed_loop.py` | Closed-loop episodes: 10 vs 3 rounds, cold vs warm-started | 8 | ~6 min |
| `scripts/bench_nonblocking.py` | Blocking vs non-blocking control, in wall-clock time | 8 | ~3 min |
| `video/capture_trace.py` → `video/render_video.py` | The video (capture on GPU, render on CPU) | 8 / 0 | ~3 + 3 min |

## What's here

| Path | What |
|---|---|
| `realtime_jepa_planning.ipynb` | The walkthrough, executed, with outputs and the embedded video. |
| `jepa_rt/model.py` | V-JEPA 2-AC core, with no Ray in it: `encode(frame)`, `score(candidates) -> energy`, `rollout`. Contains the fast path: KV cache across rollout steps, bf16, FlashAttention, `torch.compile`. |
| `jepa_rt/cem.py` | CEM as `sample` / `refit` / `plan` / `shift`, with the same rules as Meta's reference `cem()`. |
| `jepa_rt/serve_app.py` | The two Ray Serve deployments, `WorldModel` and `Planner`. `build_app(n_gpus=W)`. `plan(..., detail=True)` also returns every round's candidates and per-GPU timing (used for the video). |
| `jepa_rt/__init__.py` | Paths, the Ray `runtime_env`, and `ensure_assets()`. |
| `Dockerfile` | Custom cluster image with everything pinned. |
| `scripts/` | Benchmarks (above), `make_figures.py`, `build_notebook.py` (regenerates the notebook without outputs), `edit_blog_figures.py` (blog Figures 3–4 relabeled for A100s), `update_blog_docx.py`. |
| `results/` | Raw measurements (JSON) and run logs. |
| `images/` | Figures for the notebook and blog. |
| `video/` | Video capture + render scripts, the trace, and the mp4. |

## Lessons worth knowing

1. **Ship the latents once per plan, not once per round.** One `ray.put` of the observation and goal latents (2.8 MB). Replicas read them from shared memory and cache them on the GPU, so each round moves only ~5 KB of candidate actions out and the scores back.
2. **Make the fan-out land one shard per GPU.** With `max_ongoing_requests=1`, Serve's default power-of-two-choices router occasionally put two shards on one replica, doubling that round. Serve's round-robin request router fixed it: zero collisions in 50 rounds.
3. **Check that `torch.compile` is actually on inside Serve.** Our replica processes started with `TORCHDYNAMO_DISABLE=1`, which silently turns `torch.compile` into a no-op (372 ms instead of 272 ms per shard). We pop it in the replica's `__init__` before importing torch. Ray's own Python code doesn't set this variable, so it most likely comes from the environment.
4. **The sequential rounds are the floor.** More GPUs shrink each round. To go further, tune the number of rounds per task (3 were enough here) and overlap planning with motion.

## Caveats

- **No physical arm.** The closed-loop experiments use the world model as the simulator, with a goal made reachable by rolling a known action sequence forward. They measure the planner, not real-world task success.
- **Network latency between a robot and a cloud region is not measured.** The robot client in the notebook runs in the same cluster.
- **Beyond 8 GPUs is a projection.** The dashed part of the scaling plot, and the 32-GPU numbers in section 7 of the notebook, come from the blog's latency model calibrated on our measured A100 round. Everything at 8 GPUs and below is measured.

"""Figures for the notebook / blog, from the measured results in results/*.json.

    python scripts/make_figures.py      -> images/*.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))   # noqa: E731
OUT = os.path.join(ROOT, "images")
INK, MUTED, GRID = "#1f2328", "#656d76", "#e6e8eb"
GRAY, BLUE, TEAL, ORANGE = "#8c959f", "#2f6fdb", "#1a9e8f", "#d9822b"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

base, sc, cl, nb = R("01_single_gpu_baseline.json"), R("03_scaling.json"), R("04_closed_loop.json"), R("05_nonblocking.json")
ref_s = float(np.median(base["reference_cem_fp32_s_per_action"]))

# ---------------------------------------------------------------- 1. the latency ladder (seconds per action)
warm3 = np.median([np.median(v["plan_ms"][1:]) for v in cl["warm3"].values()]) / 1e3
nb_step = np.median([np.median(np.diff([t for t, _ in v])[1:]) for v in nb["nonblocking"].values()])
exec_s = nb["config"]["exec_s"]
rungs = [
    ("Meta reference CEM\n1 A100, fp32", ref_s, GRAY),
    ("Optimized model\n1 A100 (KV cache, bf16,\ntorch.compile)", sc["1"]["plan_ms_median"] / 1e3, GRAY),
    ("Ray Serve fan-out\n8 A100", sc["8"]["plan_ms_median"] / 1e3, BLUE),
    ("+ 3 CEM rounds per step\n(instead of 10)\n8 A100", warm3, BLUE),
]
fig, ax = plt.subplots(figsize=(8.4, 3.9))
y = np.arange(len(rungs))[::-1]
for yi, (lab, v, c) in zip(y, rungs):
    ax.barh(yi, v, color=c, height=0.62)
    ax.text(v * 1.08, yi, f"{v:.1f} s" if v >= 1 else f"{v * 1e3:.0f} ms", va="center", color=INK, fontweight="bold")
    if v != ref_s:
        ax.text(v * 1.08, yi - 0.3, f"{ref_s / v:.0f}x faster", va="center", color=MUTED, fontsize=9)
ax.set_xscale("log"); ax.set_xlim(0.3, 300)
ax.set_yticks(y, [r[0] for r in rungs], fontsize=9.5)
ax.set_xlabel("planning time per action (log scale; CEM 800 candidates, horizon 2)")
ax.set_title("From 82 seconds to under one second per action", loc="left", color=INK, fontweight="bold")
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(OUT, "latency_ladder.png")); plt.close(fig)

# ---------------------------------------------------------------- 2. scaling vs the blog's latency model
W = np.array(sorted(int(k) for k in sc))
meas = np.array([sc[str(w)]["plan_ms_median"] / 1e3 for w in W])
per_round_1 = sc["1"]["round_ms_median"] / 1e3          # calibrate the blog's model on our measured 1-GPU round
Wc = np.arange(1, 33)
eff = Wc / (1 + 0.008 * (Wc - 1))
model = 10 * (per_round_1 / eff + 0.001 * (Wc - 1)) + sc["1"]["encode_ms_median"] / 1e3
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(Wc, (meas[0]) / Wc, color=MUTED, lw=1, ls=":", label="perfect linear scaling")
ax.plot(Wc[Wc <= 8], model[Wc <= 8], color=ORANGE, lw=2, label="blog's latency model, calibrated to our 1-GPU round")
ax.plot(Wc[Wc >= 8], model[Wc >= 8], color=ORANGE, lw=2, ls="--", label="... extrapolated (needs more A100s)")
ax.plot(W, meas, "o", color=BLUE, ms=8, mec="white", mew=1.5, label="measured on 8x A100 (Ray Serve)", zorder=5)
for w, m in zip(W, meas):
    ax.annotate(f"{m:.1f} s", (w, m), textcoords="offset points", xytext=(8, 4), color=INK, fontsize=9.5)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
ax.set_xlabel("GPUs in the WorldModel tier (= Serve replicas)"); ax.set_ylabel("seconds per plan (800 x 10 rounds)")
ax.set_title(f"7.5x on 8 GPUs: {meas[0]:.1f} s -> {meas[-1]:.2f} s", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.savefig(os.path.join(OUT, "scaling.png")); plt.close(fig)

# ---------------------------------------------------------------- 3. closed loop: distance to goal vs wall-clock time
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
ax = axs[0]
for strat, c, lab in (("cold10", GRAY, "10 rounds per step (paper)"), ("cold3", ORANGE, "3 rounds per step"),
                      ("warm3", BLUE, "3 rounds, warm-started from the last plan")):
    for i, v in enumerate(cl[strat].values()):
        t = np.concatenate([[0], np.cumsum(np.array(v["plan_ms"]) / 1e3)])
        ax.plot(t, v["dist_cm"], "-o", color=c, ms=3, lw=1.6, alpha=0.9, label=lab if i == 0 else None)
ax.set_xlabel("planning time (s), robot motion excluded"); ax.set_ylabel("end-effector distance to goal (cm)")
ax.set_title("3 CEM rounds per step:\nsame accuracy as 10, 3.3x faster", loc="left", fontsize=11, color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9)
ax = axs[1]
for mode, c, lab in (("blocking", GRAY, "blocking: plan, then move"), ("nonblocking", BLUE, "non-blocking: plan while moving")):
    for i, v in enumerate(nb[mode].values()):
        t, d = zip(*v)
        ax.plot(t, d, "-o", color=c, ms=3, lw=1.6, alpha=0.9, label=lab if i == 0 else None)
ax.set_xlabel(f"wall-clock time (s), robot takes {exec_s:.0f} s per action")
ax.set_title("Non-blocking control:\nthe robot never waits for the planner", loc="left", fontsize=11, color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9)
fig.savefig(os.path.join(OUT, "closed_loop.png")); plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))

# ---------------------------------------------------------------- 4. architecture diagram
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, title, sub, fc, ec, tc=INK, fs=10.5, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), title, ha="center", va="center", fontweight=weight, color=tc, fontsize=fs)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", color=MUTED, fontsize=8.5)


def arrow(x0, y0, x1, y1, label=None, color=MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12, color=color, lw=1.3, ls=ls))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.6, label, ha="center", color=MUTED, fontsize=8.5)


box(1, 18, 16, 10, "Robot", "frame + pose", "#f6f8fa", MUTED)
box(27, 16, 22, 14, "Planner", "Ray Serve deployment\nCEM rounds, fan-out,\nHTTP /plan (no GPU)", "#eef4fd", BLUE)
arrow(17.8, 25, 26.2, 25, "POST /plan")
arrow(26.2, 20, 17.8, 20, "action")
ax.add_patch(FancyBboxPatch((58, 2), 40, 42, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#fbfcfd", ec=GRID, lw=1.2))
ax.text(78, 41.5, "WorldModel deployment · num_replicas = W", ha="center", color=INK, fontsize=8.5, fontweight="bold")
for i in range(8):
    yy = 35 - i * 4.2
    box(62, yy, 32, 2.6, f"replica {i + 1}  ·  1 A100  ·  model resident", None, "#eaf6f4", TEAL, fs=8.5, weight="normal")
    arrow(49.8, 23, 61.2, yy + 1.3, None, color=BLUE)
ax.text(38, 40.5, "each CEM round: 800 candidates → W shards,\nenergies back, gather, refit · × 10 rounds", ha="center", color=BLUE, fontsize=8.5)
box(27, 2, 22, 7, "Ray object store", "latents, put once per plan", "#f6f8fa", MUTED)
arrow(38, 15.2, 38, 9.8, None, ls="--")
ax.text(39.5, 12.5, "ray.put", color=MUTED, fontsize=8.5)
fig.savefig(os.path.join(OUT, "architecture.png")); plt.close(fig)
print("wrote architecture.png")

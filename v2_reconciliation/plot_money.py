"""plot_money.py — render the money chart (pose error vs. cloud staleness) from the measured
seed values of the full P1 sweep (2026-07-01, 4xL4, cloud 128x6x2, distilled reflex, 3 seeds).
Single clean panel: reflex-only baseline, naive-stale, vector-add reconciled. Illustrative demo
numbers — the search is stochastic and under-resourced; production means a bigger, tuned cloud.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
DELTAS = [0, 2, 4, 6, 8]
REFLEX = 0.6015

naive = {0: [0.1397, 0.5135, 0.0824], 2: [0.2988, 0.3973, 0.1727], 4: [0.2049, 0.3632, 0.1106],
         6: [0.2111, 0.1841, 0.1625], 8: [0.3361, 0.3730, 0.2630]}
vec = {0: [0.1397, 0.5135, 0.0824], 2: [0.3217, 0.2948, 0.2984], 4: [0.0555, 0.0609, 0.1051],
       6: [0.1021, 0.0919, 0.1204], 8: [0.1452, 0.1762, 0.1276]}

nm = np.array([np.mean(naive[d]) for d in DELTAS]); ns = np.array([np.std(naive[d]) for d in DELTAS])
vm = np.array([np.mean(vec[d]) for d in DELTAS]); vs = np.array([np.std(vec[d]) for d in DELTAS])

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.axhline(REFLEX, color="#5f6368", ls=":", lw=2.2, label="reflex-only (ignore the cloud)")
ax.errorbar(DELTAS, nm, yerr=ns, fmt="s--", color="#ea4335", lw=2.2, ms=7, capsize=4,
            label="naive (apply stale advice as-is)")
ax.errorbar(DELTAS, vm, yerr=vs, fmt="o-", color="#1a73e8", lw=2.4, ms=7, capsize=4,
            label="vector-add reconciled")
ax.annotate("reconciliation\nkeeps it near the goal", xy=(6, 0.105), xytext=(2.5, 0.03),
            fontsize=9.5, color="#1a73e8",
            arrowprops=dict(arrowstyle="->", color="#1a73e8", lw=1.4))
ax.annotate("naive drifts back\ntoward doing nothing", xy=(8, 0.324), xytext=(5.2, 0.47),
            fontsize=9.5, color="#ea4335",
            arrowprops=dict(arrowstyle="->", color="#ea4335", lw=1.4))
ax.set_xlabel("cloud staleness  Δ  (control ticks the advice is late)")
ax.set_ylabel("final pose error to goal   (↓ better)")
ax.set_title("Reconciling the stale cloud (pose error, 4×L4 demo)")
ax.set_xticks(DELTAS); ax.set_ylim(0, 0.66); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out = f"{REPO}/images/p1_feasibility.png"
fig.savefig(out, dpi=140)
print("saved", out)
np.savez(f"{REPO}/images/p1_money_data.npz", deltas=DELTAS, reflex=REFLEX,
         naive_mean=nm, naive_std=ns, vec_mean=vm, vec_std=vs)
print("saved data npz")

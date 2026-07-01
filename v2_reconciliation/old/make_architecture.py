"""make_architecture.py — render architecture.png: fast reflex + reconciler on the robot (edge),
slow world-model brain served on Anyscale, with the stale corrected-destination coming back over a
latency'd link. Pure matplotlib, no external assets."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = "/home/ray/default/robotics_world_model_inference/v2_reconciliation"
fig, ax = plt.subplots(figsize=(11, 5.9))
ax.set_xlim(0, 11); ax.set_ylim(0, 5.9); ax.axis("off")

EDGE, CLOUD, INK, BLUE, RED, GREEN = "#e8f0fe", "#e6f4ea", "#202124", "#1a73e8", "#ea4335", "#34a853"

def box(x, y, w, h, fc, ec, title, lines, tsize=12, lsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="top", fontsize=tsize, weight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.76 - i * 0.35, ln, ha="center", va="top", fontsize=lsize, color=INK)

ax.text(5.5, 5.66, "Fast brain on the robot  ·  slow world-model brain on Anyscale  ·  reconcile the lag",
        ha="center", va="center", fontsize=13, weight="bold", color=INK)
ax.text(2.3, 5.18, "Robot  (edge)", ha="center", va="center", fontsize=12.5, weight="bold", color=BLUE)
ax.text(8.7, 5.18, "Anyscale cloud", ha="center", va="center", fontsize=12.5, weight="bold", color=GREEN)

# containers
ax.add_patch(FancyBboxPatch((0.3, 0.4), 4.0, 4.5, boxstyle="round,pad=0.03,rounding_size=0.12",
                            fc=EDGE, ec=BLUE, lw=1.8))
ax.add_patch(FancyBboxPatch((6.7, 0.4), 4.0, 4.5, boxstyle="round,pad=0.03,rounding_size=0.12",
                            fc=CLOUD, ec=GREEN, lw=1.8))

# edge inner
box(0.6, 2.9, 3.4, 1.55, "white", BLUE, "Fast reflex",
    ["one forward pass, ~ms", "steers to the destination"])
box(0.6, 0.55, 3.4, 2.05, "white", RED, "Reconciler  (the glue)",
    ["a_cloud = the cloud's planned action", "retain snapshot pose p0",
     "destination = p0 + a_cloud", "steer there from where I am now"])

# cloud inner
box(7.0, 2.45, 3.4, 2.0, "white", GREEN, "Ray Serve: slow brain",
    ["V-JEPA 2-AC world model", "CEM search in latent space", "data-parallel across GPUs", "more GPUs -> faster -> smaller Δ"])
box(7.0, 0.7, 3.4, 1.55, "white", GREEN, "World-model inference",
    ["seconds per plan", "scale = num_replicas / shards"])

# arrows
ax.add_patch(FancyArrowPatch((4.35, 3.6), (6.65, 3.6), arrowstyle="-|>", mutation_scale=20,
                             lw=2, color=BLUE, connectionstyle="arc3,rad=-0.16"))
ax.text(5.5, 4.25, "snapshot\n(latent + pose)", ha="center", va="center", fontsize=10, color=BLUE)
ax.add_patch(FancyArrowPatch((6.65, 1.45), (4.35, 1.45), arrowstyle="-|>", mutation_scale=20,
                             lw=2, color=RED, connectionstyle="arc3,rad=-0.16"))
ax.text(5.5, 0.9, "corrected destination\n(STALE by Δ)", ha="center", va="center",
        fontsize=10, color=RED, linespacing=1.4)

plt.tight_layout()
out = f"{REPO}/images/architecture.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)

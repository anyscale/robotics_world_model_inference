"""Render the blog video from video/trace.json (CPU only; no GPUs needed).

    python video/render_video.py                 -> video/realtime_jepa_planning.mp4   (1920x1080, 30 fps)
    python video/render_video.py --preview 6.0   -> video/preview_6.0s.png              (a single frame, to check layout)

Two rows, same task, same planner, same 8 GPUs, both recorded live:
  top    = blocking control      (plan, then move: the arm waits for every plan)
  bottom = non-blocking control  (plan the next step while the arm moves)
Each row: left, a top-down view of the gripper path plus the planner's 800 imagined end positions in the current CEM
round (the best 80 highlighted); right, distance to goal over time and the timeline of the robot, the planner's CEM
rounds and the 8 GPUs' shards. Everything is drawn from measured timestamps.
"""
import argparse, json, os, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--preview", type=float, default=None, help="render one frame at this time (s) to a PNG")
ap.add_argument("--out", default=os.path.join(HERE, "realtime_jepa_planning.mp4"))
args = ap.parse_args()

T = json.load(open(os.path.join(HERE, "trace.json")))
S = np.array(T["xyz_start"]); G = np.array(T["xyz_goal"])
cm = lambda xyz: (np.asarray(xyz)[..., :2] - S[:2]) * 100        # noqa: E731   top-down view, cm from start
GOAL = cm(G)
cm3 = lambda xyz: (np.asarray(xyz) - S) * 100                     # noqa: E731   3-D, for distances
GOAL3 = cm3(G)
T_END = max(T["blocking"]["t_total"], T["nonblocking"]["t_total"])
HOLD = 2.5                                                         # seconds to hold the final state
T_AXIS = np.ceil(T_END + 0.5)

INK, MUTED, GRID, BG = "#1f2328", "#656d76", "#e6e8eb", "#ffffff"
GREEN, GRAY, BLUE, ORANGE, LIGHT = "#2da44e", "#c4c9cf", "#2f6fdb", "#d9822b", "#9aa4ae"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})


# ------------------------------------------------------------------------------------------------ trace helpers
def gripper_xyz(log, t):
    """Gripper position (cm from start, 3-D) at time t: eased during each move, held while waiting."""
    pos = cm3(S)
    for m in log["moves"]:
        if t < m["t_start"]:
            break
        a, b = cm3(m["xyz_from"]), cm3(m["xyz_to"])
        if t >= m["t_end"]:
            pos = b
        else:
            u = (t - m["t_start"]) / (m["t_end"] - m["t_start"])
            u = u * u * (3 - 2 * u)
            pos = a + (b - a) * u
    return pos


def gripper_xy(log, t):
    return gripper_xyz(log, t)[:2]


def path_xy(log, t):
    pts = [cm(S)] + [cm(m["xyz_to"]) for m in log["moves"] if m["t_end"] <= t]
    return np.array(pts + [gripper_xy(log, t)])


def dist_curve(log, t, n=400):
    ts = np.linspace(0, min(t, log["t_total"]), n)
    return ts, np.array([np.linalg.norm(gripper_xyz(log, x) - GOAL3) for x in ts])


def current_round(log, t):
    """The CEM round in progress at t (or the one that just ended, within 0.35 s), with its plan index."""
    best = None
    for pi, p in enumerate(log["plans"]):
        for ri, r in enumerate(p["rounds"]):
            if r["t_start"] <= t <= r["t_end"] + 0.35:
                best = (pi, ri, r, t > r["t_end"])
    return best


def moving(log, t):
    return any(m["t_start"] <= t < m["t_end"] for m in log["moves"])


def planning(log, t):
    return any(p["t_call"] <= t < p["t_back"] for p in log["plans"])


# ------------------------------------------------------------------------------------------------ figure layout
W, H, DPI = 1920, 1080, 100
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
fig.text(0.03, 0.955, "Planning with a world model, served on Ray", fontsize=26, fontweight="bold", color=INK)
fig.text(0.03, 0.918, "V-JEPA 2-AC  ·  CEM search: 800 imagined futures per round  ·  each round fanned out over "
         "8× A100 with Ray Serve  ·  recorded live, real time", fontsize=14, color=MUTED)
clock = fig.text(0.97, 0.955, "", fontsize=24, color=INK, ha="right", fontweight="bold", family="DejaVu Sans Mono")
fig.text(0.03, 0.018, "The world model stands in for the robot (no physical arm): each executed action is rolled "
         "forward through V-JEPA 2-AC, with 4 mm execution noise and 1 s per move. Gripper shown top-down.",
         fontsize=11, color=MUTED)

ROWS = {}
for k, (mode, title, sub) in enumerate((
        ("blocking", "Blocking control", "plan, then move: the arm waits for every plan"),
        ("nonblocking", "Non-blocking control", "plan the next step while the arm moves"))):
    y0 = 0.505 - k * 0.445                               # row bottom
    ax_xy = fig.add_axes([0.035, y0 + 0.02, 0.215, 0.33])
    ax_d = fig.add_axes([0.335, y0 + 0.215, 0.635, 0.135])
    ax_tl = fig.add_axes([0.335, y0 + 0.02, 0.635, 0.165])
    ttl = fig.text(0.035, y0 + 0.375, title, fontsize=19, fontweight="bold", color=INK)
    bb = ttl.get_window_extent(renderer=fig.canvas.get_renderer())
    fig.text(bb.x1 / W + 0.008, y0 + 0.377, sub, fontsize=13, color=MUTED)
    badge = fig.text(0.97, y0 + 0.375, "", fontsize=14, fontweight="bold", ha="right", color="white",
                     bbox=dict(boxstyle="round,pad=0.35", fc=GRAY, ec="none"))

    # top-down view
    ax_xy.set_xlim(-12, 31); ax_xy.set_ylim(-12, 23); ax_xy.set_aspect("equal")
    ax_xy.set_xticks([0, 10, 20, 30]); ax_xy.set_yticks([-10, 0, 10, 20])
    ax_xy.tick_params(labelsize=9); ax_xy.grid(color=GRID, lw=0.6)
    ax_xy.set_xlabel("x (cm)", fontsize=10); ax_xy.set_ylabel("y (cm)", fontsize=10)
    ax_xy.add_patch(Circle(GOAL, 3.0, fc="#fdebd7", ec=ORANGE, lw=1.5, zorder=1))
    ax_xy.text(GOAL[0], GOAL[1] + 4.2, "goal", ha="center", color=ORANGE, fontsize=10, fontweight="bold")
    ax_xy.plot(0, 0, "s", color=MUTED, ms=6, zorder=2)
    ax_xy.text(0, -3.4, "start", ha="center", color=MUTED, fontsize=9)
    cloud = ax_xy.scatter([], [], s=5, c=LIGHT, alpha=0.5, lw=0, zorder=3)
    elite = ax_xy.scatter([], [], s=9, c=BLUE, alpha=0.85, lw=0, zorder=4)
    path, = ax_xy.plot([], [], "-", color=INK, lw=1.6, alpha=0.6, zorder=5)
    grip, = ax_xy.plot([], [], "o", ms=13, mfc=GREEN, mec="white", mew=2, zorder=6)
    cloud_lbl = ax_xy.text(-11, 21.3, "", fontsize=10, color=INK, va="top", fontweight="bold")
    # permanent legend for the candidate cloud
    ax_xy.scatter([-10.2], [-6.3], s=30, c=LIGHT, lw=0, zorder=8); ax_xy.scatter([-10.2], [-8.8], s=30, c=BLUE, lw=0, zorder=8)
    ax_xy.text(-9.0, -6.3, "one imagined future (800 per round)", fontsize=9, color=MUTED, va="center")
    ax_xy.text(-9.0, -8.8, "80 closest to goal, steer the next round", fontsize=9, color=BLUE, va="center")

    # distance to goal
    ax_d.set_xlim(0, T_AXIS); ax_d.set_ylim(0, 24); ax_d.set_ylabel("cm to goal", fontsize=10)
    ax_d.set_yticks([0, 10, 20]); ax_d.grid(color=GRID, lw=0.6); ax_d.tick_params(labelbottom=False, labelsize=9)
    dline, = ax_d.plot([], [], color=GREEN, lw=2.4)
    dtxt = ax_d.text(T_AXIS - 0.1, 20, "", ha="right", fontsize=12, color=INK, fontweight="bold")

    # timeline: robot lane, planner lane, 8 GPU lanes
    ax_tl.set_xlim(0, T_AXIS); ax_tl.set_ylim(-0.5, 10.6)
    ax_tl.set_yticks([10, 8.9] + list(np.arange(7, -1, -1) * 0.95))
    ax_tl.set_yticklabels(["robot", "planner"] + [f"GPU {i + 1}" for i in range(8)], fontsize=8.5)
    ax_tl.tick_params(axis="y", length=0); ax_tl.tick_params(axis="x", labelsize=9)
    ax_tl.set_xlabel("time (s)", fontsize=10); ax_tl.grid(axis="x", color=GRID, lw=0.6)
    ax_tl.spines["left"].set_visible(False)
    robot_coll = PolyCollection([], zorder=3); plan_coll = PolyCollection([], zorder=3)
    gpu_coll = PolyCollection([], facecolors=BLUE, zorder=3)
    for c in (robot_coll, plan_coll, gpu_coll):
        ax_tl.add_collection(c)
    cursors = [ax.axvline(0, color=INK, lw=1.2, alpha=0.7) for ax in (ax_d, ax_tl)]
    ROWS[mode] = dict(log=T[mode], cloud=cloud, elite=elite, path=path, grip=grip, cloud_lbl=cloud_lbl,
                      dline=dline, dtxt=dtxt, badge=badge, robot=robot_coll, plan=plan_coll, gpu=gpu_coll,
                      cursors=cursors)


def rect(t0, t1, yc, h):
    return [(t0, yc - h / 2), (t1, yc - h / 2), (t1, yc + h / 2), (t0, yc + h / 2)]


def draw(t):
    clock.set_text(f"t = {min(t, T_END):5.1f} s")
    for mode, R in ROWS.items():
        log = R["log"]
        tt = min(t, log["t_total"])
        done = t >= log["t_total"]
        # gripper + path
        pth = path_xy(log, tt)
        R["path"].set_data(pth[:, 0], pth[:, 1])
        g = gripper_xy(log, tt)
        R["grip"].set_data([g[0]], [g[1]])
        mv = moving(log, tt) and not done
        R["grip"].set_markerfacecolor(GREEN if mv or done else "#8c959f")
        # candidate cloud of the current CEM round
        cr = None if done else current_round(log, tt)
        if cr:
            pi, ri, r, ended = cr
            pts = cm(np.array(r["cand_end_xyz"])); e = np.array(r["energy"])
            top = np.argsort(e)[:80]
            alpha = 0.25 if ended else 0.55
            R["cloud"].set_offsets(pts); R["cloud"].set_alpha(alpha)
            R["elite"].set_offsets(pts[top]); R["elite"].set_alpha(0.4 if ended else 0.9)
            n_r = len(log["plans"][pi]["rounds"])
            R["cloud_lbl"].set_text(f"planning: plan {pi + 1} · CEM round {ri + 1}/{n_r}")
        else:
            R["cloud"].set_offsets(np.empty((0, 2))); R["elite"].set_offsets(np.empty((0, 2)))
            R["cloud_lbl"].set_text("")
        # distance curve
        ts, ds = dist_curve(log, tt)
        R["dline"].set_data(ts, ds)
        R["dtxt"].set_text(f"{ds[-1]:.1f} cm" + (f"  ·  12 moves in {log['t_total']:.1f} s" if done else ""))
        # timeline
        R["robot"].set_verts([rect(m["t_start"], min(m["t_end"], tt), 10, 0.8) for m in log["moves"] if m["t_start"] < tt])
        R["robot"].set_facecolors(GREEN)
        waits, prev = [], 0.0
        for m in log["moves"] + [{"t_start": log["t_total"], "t_end": log["t_total"]}]:
            if m["t_start"] > prev + 0.05 and prev < tt:
                waits.append(rect(prev, min(m["t_start"], tt), 10, 0.8))
            prev = m["t_end"]
        R["robot"].set_verts([rect(m["t_start"], min(m["t_end"], tt), 10, 0.8) for m in log["moves"] if m["t_start"] < tt] + waits)
        R["robot"].set_facecolors([GREEN] * sum(m["t_start"] < tt for m in log["moves"]) + [GRAY] * len(waits))
        rounds = [r for p in log["plans"] for r in p["rounds"] if r["t_start"] < tt]
        R["plan"].set_verts([rect(r["t_start"], min(r["t_end"], tt) - 0.02, 8.9, 0.7) for r in rounds])
        R["plan"].set_facecolors(BLUE)
        R["gpu"].set_verts([rect(a, min(b, tt), (7 - i) * 0.95, 0.7) for r in rounds
                            for i, (a, b) in enumerate(sorted(r["shards"])) if a < tt])
        for c in R["cursors"]:
            c.set_xdata([tt, tt]); c.set_alpha(0 if done else 0.7)
        # status badge
        if done:
            R["badge"].set_text(f"  {len(log['moves'])} MOVES DONE · {log['t_total']:.1f} s  "); R["badge"].get_bbox_patch().set_fc(INK)
        elif mv and planning(log, tt):
            R["badge"].set_text("  MOVING + PLANNING NEXT STEP  "); R["badge"].get_bbox_patch().set_fc(BLUE)
        elif mv:
            R["badge"].set_text("  MOVING  "); R["badge"].get_bbox_patch().set_fc(GREEN)
        else:
            R["badge"].set_text("  WAITING FOR PLANNER  "); R["badge"].get_bbox_patch().set_fc("#8c959f")


if args.preview is not None:
    draw(args.preview)
    out = os.path.join(HERE, f"preview_{args.preview:.1f}s.png")
    fig.savefig(out, dpi=DPI); print("wrote", out)
else:
    n = int(round((T_END + HOLD) * args.fps))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                           "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    for k in range(n):
        draw(k / args.fps)
        fig.canvas.draw()
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    # end card: the measured latency ladder from the notebook
    card = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    card.text(0.5, 0.9, "Planning time per action, measured on 8× A100 with Ray Serve", ha="center", fontsize=28,
              fontweight="bold", color=INK)
    lad = plt.imread(os.path.join(os.path.dirname(HERE), "images", "latency_ladder.png"))
    ax = card.add_axes([0.1, 0.12, 0.8, 0.72]); ax.imshow(lad); ax.axis("off")
    card.text(0.5, 0.06, "Code + notebook: real-time V-JEPA 2-AC planning with Ray Serve on Anyscale", ha="center",
              fontsize=16, color=MUTED)
    card.canvas.draw()
    frame = np.asarray(card.canvas.buffer_rgba()).tobytes()
    for _ in range(int(4 * args.fps)):
        ff.stdin.write(frame)
        n += 1
    ff.stdin.close(); ff.wait()
    print(f"wrote {args.out} ({n} frames, {n / args.fps:.1f} s)")

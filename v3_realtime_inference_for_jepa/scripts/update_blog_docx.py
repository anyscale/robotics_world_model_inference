"""Write measured results into a COPY of Omar's draft. The original .docx is never modified."""
import copy
import json
import os

import docx
from docx.shared import Inches

ROOT = "/home/ray/default/robotics_world_model_inference"
V3 = f"{ROOT}/v3_realtime_inference_for_jepa"
SRC = f"{ROOT}/Real-Time JEPA World Models.docx"
DST = f"{ROOT}/Real-Time JEPA World Models (measured results).docx"
R = lambda f: json.load(open(f"{V3}/results/{f}"))   # noqa: E731
sc, base = R("03_scaling.json"), R("01_single_gpu_baseline.json")

d = docx.Document(SRC)
P = list(d.paragraphs)
assert P[46].text.startswith("05") and P[47].text.startswith("A modeled projection") and P[45].text.startswith("Source.")
assert P[53]._p.xpath(".//w:drawing") and P[71].text.startswith("The four setups")


def set_runs(p, runs):
    """Replace a paragraph's text with [(text, bold, italic), ...], keeping its paragraph formatting."""
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    for text, b, i in runs:
        r = p.add_run(text)
        r.bold, r.italic = b or None, i or None
    return p


def insert_after(p, runs, like=None):
    """New paragraph after `p`, formatted like `like` (default: p), with the given runs."""
    tmpl = (like or p)._p
    new = copy.deepcopy(tmpl)
    for r in new.xpath("./w:r | ./w:hyperlink"):
        new.remove(r)
    p._p.addnext(new)
    para = docx.text.paragraph.Paragraph(new, p._parent)
    return set_runs(para, runs)


def insert_image_after(p, image, like_img_para, width=6.5):
    new = copy.deepcopy(like_img_para._p)
    for r in new.xpath("./w:r"):
        new.remove(r)
    p._p.addnext(new)
    para = docx.text.paragraph.Paragraph(new, p._parent)
    para.add_run().add_picture(image, width=Inches(width))
    return para


img_para, caption_para, body_para = P[53], P[10], P[48]
ref_s = sum(base["reference_cem_fp32_s_per_action"]) / 3
s1, s8 = sc["1"]["plan_ms_median"] / 1e3, sc["8"]["plan_ms_median"] / 1e3

# --- 04 · Implementation: what we learned + source
learned = insert_after(P[44], [("What we learned building it. ", True, False), (
    "Three details decided the latency. First, the observation and goal latents (2.8 MB) go into Ray's object store "
    "once per plan with ray.put; every replica reads them from shared memory and caches them on its GPU, so each "
    "round moves only about 5 KB of candidate actions out and the energies back, and Ray's whole share of a round "
    "(fan-out, gather, refit) is under 10 ms. Second, with max_ongoing_requests=1 Serve's default power-of-two-choices "
    "router occasionally placed two shards on one replica while another sat idle, doubling that round; Serve's "
    "round-robin request router lands the shards of a round on consecutive replicas, one per GPU (zero collisions "
    "in 50 rounds). Third, our Serve replica processes started with TORCHDYNAMO_DISABLE=1 in their environment, which "
    "silently turned torch.compile into a no-op; clearing it in the replica's constructor recovered a 1.35× speedup.",
    False, False)], like=P[44])
set_runs(P[45], [("Source. ", True, False), (
    "The full implementation, including model loading, the fast scoring path, the complete CEM refinement step, the "
    "two Serve deployments, every benchmark script and a runnable notebook, lives in the companion repository: "
    "[repository link to be added].", False, False)])

# --- 05 · Results: measured
set_runs(P[47], [("Measured on 8× A100, and what that implies for 32× H100", False, False)])
set_runs(P[48], [(
    f"We built the design above and measured it on one Anyscale node with eight A100-40GB GPUs, using the search "
    f"settings from Section 01 (800 candidates × 10 rounds, keep 80, horizon 2). The anchor is Meta's reference "
    f"planner, run unmodified in fp32 on one A100: {ref_s:.1f} s per action. Before adding GPUs we made the model "
    f"itself faster on one GPU: a KV cache across rollout steps (the predictor is frame-causal, so earlier frames "
    f"never need recomputing), bf16 with FlashAttention, and torch.compile. The result matches the reference energies "
    f"(Pearson r = 0.9999) and brings one GPU to {s1:.1f} s per action, 4× faster.", False, False)])
fig5 = insert_image_after(P[48], f"{V3}/images/latency_ladder.png", img_para)
cap5 = insert_after(fig5, [("Figure 5.", True, True), (
    f" Measured planning time per action. Ray Serve fans each round out over 8 A100s ({s8:.2f} s, 29× the reference). "
    "Running 3 search rounds per step instead of 10 reached the same accuracy in closed-loop tests (Section 06), "
    "giving 0.86 s per action, 96× the reference.", False, True)], like=caption_para)
set_runs(P[49], [(
    f"Then the fan-out. Each round's 800 candidates are split across W WorldModel replicas, one GPU each. Going from "
    f"1 to 8 GPUs cuts a plan from {s1:.1f} s to {s8:.2f} s, a 7.5× speedup. Same seeds give identical plans at every "
    f"GPU count, so sharding changes the speed of the search but not its answer. The latency model proposed in the "
    f"first draft of this post holds up: calibrated on our measured single-GPU round (2.14 s), it predicts 2.94 s at "
    f"8 GPUs against {s8:.2f} s measured.", False, False)])
set_runs(P[51], [("A100-40GB replicas on one 8-GPU node. Points are measured; the line is the latency model.", False, False)])
set_runs(P[52], [(f"{s8:.2f} s / action at 8 A100 workers", True, False),
                 (f" — 7.5× faster than one A100 and {ref_s / s8:.0f}× faster than Meta's reference planner", False, False)])
# replace the projection chart with the measured scaling chart
for r in P[53].runs:
    if r._r.xpath(".//w:drawing"):
        r._r.getparent().remove(r._r)
P[53].add_run().add_picture(f"{V3}/images/scaling.png", width=Inches(6.0))
set_runs(P[54], [(
    "Model: time(W) = 10 × ( r / effective(W) + 0.001s × (W − 1) ), where effective(W) = W / (1 + 0.008×(W−1)) and "
    "r is the measured single-GPU round, 2.14 s on an A100. Projected beyond our 8 GPUs, the same model gives about "
    "1.2 s per action on 32 A100s. Assuming the original draft's conservative 2× per-GPU speedup for an H100, it gives "
    "about 0.8 s on 32 H100s (0.3 s with 3 rounds per step). The 32-GPU figures are projections; everything at 8 GPUs "
    "and below is measured.", False, False)])
set_runs(P[55], [("What is still not measured. ", True, False), (
    "Network latency between an on-premises robot and a cloud region; a physical arm (our closed-loop tests use the "
    "world model as the simulator); and the real H100 speedup for this model's operation mix. Two items from the "
    "original list are now measured. Cold start stays off the request path: replicas load and compile in about 50 s "
    "at deploy time, before they take traffic. Batching candidates differently does not change accuracy: the plans "
    "are identical at 1, 2, 4 and 8 GPUs.", False, False)])

# --- 06 · Further optimization: measured notes under A, B, E
a_meas = insert_after(P[60], [("Measured. ", True, False), (
    "We simulated an arm that takes 1 s per action and lands with 4 mm of execution error. A blocking controller "
    "spent 1.86 s per step (plan, then move). Non-blocking control, planning step t+1 from the world model's "
    "prediction of where step t will land, spent 1.04 s per step, i.e. the motion itself plus 40 ms. It reached the "
    "same final accuracy, and a 12-step episode took 15.3 s instead of 25.3 s.", False, False)], like=P[60])
fig7 = insert_image_after(a_meas, f"{V3}/images/closed_loop.png", img_para)
insert_after(fig7, [("Figure 7.", True, True), (
    " Closed-loop episodes toward a goal 21.6 cm away, with the world model standing in for the robot. Left: 3 search "
    "rounds per step reach the same accuracy as 10 in a third of the planning time. Right: non-blocking control "
    "removes the planner from the robot's critical path.", False, True)], like=caption_para)
set_runs(cap5, [("Figure 5.", True, True), (
    f" Measured planning time per action. Ray Serve fans each round out over 8 A100s ({s8:.2f} s, 29× the reference). "
    "Running 3 search rounds per step instead of 10 reached the same accuracy in closed-loop tests (Section 06), "
    "giving 0.86 s per action, 96× the reference.", False, True)])
# the scaling chart is Figure 6: give it a caption
insert_after(P[53], [("Figure 6.", True, True), (
    " Plan latency vs. number of GPU replicas, measured on 8× A100 (points) against the latency model calibrated on "
    "the measured single-GPU round (line; dashed beyond 8 GPUs = projection).", False, True)], like=caption_para)
insert_after(P[62], [("Measured. ", True, False), (
    "In closed-loop tests, 3 rounds per step (0.86 s on 8 A100s) reached the same accuracy as the full 10 (2.83 s): "
    "both settle about 4 cm from a goal that starts 21.6 cm away. Warm-starting from the previous plan worked, but "
    "on this task a fresh 3-round search did equally well, so the gain came from running fewer rounds. How many "
    "rounds a task needs is worth tuning per task.", False, False)], like=P[62])
insert_after(P[68], [("Implemented. ", True, False), (
    "KV caching across rollout steps, together with bf16, FlashAttention and torch.compile, is part of the measured "
    "system. It gives 4× on a single GPU and matches the reference energies to r = 0.9999.", False, False)], like=P[68])

# --- 07 · Takeaways + disclaimer
t = P[71].text
old = ("The projection puts four eight-GPU H100 nodes at roughly 0.6 seconds per action, with distribution rather "
       "than chip choice doing most of the work, and the ten sequential rounds setting a floor more hardware will not "
       "move. Attack that floor first, with non-blocking control and warm-started search.")
assert old in t
set_runs(P[71], [(t.replace(old, (
    f"Measured on one node of eight A100s, Ray Serve takes the full search from {ref_s:.0f} s to {s8:.1f} s per "
    "action, with distribution doing most of the work and Ray adding under 10 ms per round. The ten sequential rounds "
    "set the floor that more hardware cannot move. Attacking that floor with fewer rounds per step brings it to "
    "0.86 s, and non-blocking control hides what remains behind the arm's own motion.")), False, False)])
set_runs(P[78], [(P[78].text + " Latency measurements in Sections 05 and 06 were taken on one 8× A100-40GB node on "
                  "Anyscale (November 2026); figures for 32 GPUs remain projections.", False, True)])

d.save(DST)
print("wrote", DST)

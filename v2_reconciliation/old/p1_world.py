"""
p1_world.py — P1 feasibility spike (NOT the final architecture; no Ray Serve / HTTP yet).

A single GPU Ray actor that plays every role at once so we can answer the go/no-go question:
when the cloud's correction is STALE, does vector-addition reconciliation beat naively applying it?

Roles inside the one actor:
  - World oracle : the V-JEPA 2-AC predictor advances the TRUE (latent, pose) given an action.
  - Cloud brain  : full-budget CEM in latent space -> the "good but slow" action for a snapshot.
  - Reflex       : tiny-budget CEM -> the "fast but weak" on-robot action.
  - Reconcilers  : naive / vector_add, applied when a stale cloud action "arrives".

Latency is simulated: the cloud action is computed on the state at request time t0, but only
APPLIED at t0+Delta. Real async HTTP comes in P2. Everything lives in latent+pose space; we
encode the context/goal frames once, then roll forward.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

TORCH_HOME = os.environ.setdefault("TORCH_HOME", "/mnt/cluster_storage/torch_hub")
os.environ.setdefault("HF_HOME", "/mnt/cluster_storage/hf_cache")
_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"
CROP = 256


def _ensure_repo_on_path():
    for p in (_REPO, f"{_REPO}/notebooks"):
        if p not in sys.path:
            sys.path.insert(0, p)


import ray


@ray.remote(num_gpus=1)
class SpikeWorker:
    def __init__(self):
        _ensure_repo_on_path()
        from app.vjepa_droid.transforms import make_transforms

        enc, pred = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_ac_vit_giant", trust_repo=True
        )
        self.encoder = enc.to("cuda").eval()
        self.predictor = pred.to("cuda").eval()
        self.device = "cuda"
        self.tpf = int((CROP // self.encoder.patch_size) ** 2)  # tokens per frame
        self.transform = make_transforms(
            random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
            motion_shift=False, crop_size=CROP,
        )

    # ---- encode / dynamics ------------------------------------------------
    @torch.no_grad()
    def encode(self, image):
        clip = self.transform(np.expand_dims(image, 0))[None, :]
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2)
        clip = clip.repeat(1, 1, 2, 1, 1).to(self.device)
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        return F.layer_norm(h, (h.size(-1),))  # [1, tpf, D]

    @torch.no_grad()
    def _step(self, reps, actions, poses):
        """World-model step. reps [B,T,N_T,D], actions/poses [B,T,7] -> (next_rep[B,1,N_T,D], next_pose)."""
        from utils.mpc_utils import compute_new_pose

        nxt = self.predictor(reps.flatten(1, 2), actions, poses)[:, -self.tpf:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),)).view(reps.size(0), 1, reps.size(2), reps.size(3))
        return nxt, compute_new_pose(poses[:, -1:], actions[:, -1:])

    @torch.no_grad()
    def env_step(self, z, pose, action):
        """Ground-truth dynamics: advance one tick. z [1,1,N_T,D], pose/action [1,1,7]."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_next, pose_next = self._step(z, action, pose)
        return z_next, pose_next

    def _energy(self, z, z_goal):
        from utils.mpc_utils import l1
        return l1(z[:, -1].flatten(1), z_goal[:, -1].flatten(1)).item()

    # ---- planners ---------------------------------------------------------
    @torch.no_grad()
    def _cem(self, z, pose, z_goal, samples, cem_steps, rollout=1):
        from utils.mpc_utils import cem
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = cem(
                context_frame=z, context_pose=pose, goal_frame=z_goal,
                world_model=self._step, rollout=rollout, samples=samples,
                cem_steps=cem_steps, topk=max(4, samples // 10),
                momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05, close_gripper=None,
            )
        return action[:, 0:1]  # [1,1,7] single-step action

    # ---- episode rollout --------------------------------------------------
    @torch.no_grad()
    def run_episode(self, z0_np, p0_np, zg_np, *, strategy, delta, cloud_period=3, horizon=24,
                    rollout=4, reflex_samples=4, reflex_steps=1, cloud_samples=64, cloud_steps=8,
                    reflex_gain=0.5, seed=0, env_drift=None, pg_np=None):
        """Chunk-based MPC closed loop under one reconciliation strategy and cloud latency `delta`.

        The cloud plans a `rollout`-step CHUNK from the snapshot state; it arrives `delta` ticks later,
        by which point the robot has drifted (executing the previous chunk / reflex-bridging).
          - naive       : execute the chunk's actions anchored at the CURRENT (drifted) pose -> the
                          in-flight motion is double-counted -> overshoot, error grows with delta.
          - vector_add  : track the chunk's ABSOLUTE target poses anchored at the SNAPSHOT pose -> the
                          drift is cancelled (poses_to_diff back onto the intended trajectory).
          - reflex      : ignore the cloud; weak CEM every tick (flat baseline).
        strategy: 'reflex' | 'naive' | 'vector_add'
        """
        from utils.mpc_utils import poses_to_diff, compute_new_pose
        torch.manual_seed(seed)
        maxnorm = 0.05

        z = torch.tensor(z0_np, device=self.device)
        pose = torch.tensor(p0_np, device=self.device)
        z_goal = torch.tensor(zg_np, device=self.device)
        # exogenous, strategy-independent disturbance (latency + miscalibration/push) added every tick
        drift = torch.zeros(1, 1, 7, device=self.device, dtype=pose.dtype)
        if env_drift is not None:
            drift = torch.tensor(env_drift, device=self.device, dtype=pose.dtype).reshape(1, 1, 7)
        # exact, world-model-independent metric: xyz distance from the goal pose
        goal_xyz = None
        if pg_np is not None:
            goal_xyz = torch.tensor(pg_np, device=self.device, dtype=pose.dtype).reshape(1, 1, 7)[..., :3]

        def pose_err():
            return float(torch.linalg.norm(pose[..., :3] - goal_xyz).item()) if goal_xyz is not None else float("nan")

        def bounded_step_to(p_now, p_target):
            a = poses_to_diff(p_now[:, 0].flatten().cpu(), p_target[:, 0].flatten().cpu())
            a = a.to(self.device).to(p_now.dtype).reshape(1, 1, 7)
            a[..., :3] = torch.clip(a[..., :3], -maxnorm, maxnorm)
            return a

        def abs_targets(anchor, chunk):
            """Cumulative absolute target poses produced by applying chunk actions from `anchor`."""
            tgts, P = [], anchor
            for k in range(chunk.size(1)):
                P = compute_new_pose(P, chunk[:, k:k + 1])
                tgts.append(P.clone())
            return tgts

        pending = []          # chunks in flight: {arrive, p0, chunk}
        active, aidx = None, 0
        energies = [self._energy(z, z_goal)]
        perrs = [pose_err()]
        poses = [pose.flatten().cpu().numpy()]

        for t in range(horizon):
            # request a fresh chunk on schedule (planned from the TRUE state now)
            if strategy != "reflex" and t % cloud_period == 0:
                chunk = self._cem_plan(z, pose, z_goal, cloud_samples, cloud_steps, rollout)
                pending.append({"arrive": t + delta, "p0": pose.clone(), "chunk": chunk})

            # a stale chunk arrives -> becomes the active plan, anchored per strategy
            for pc in [pc for pc in pending if pc["arrive"] == t]:
                anchor = pose.clone() if strategy == "naive" else pc["p0"]
                active, aidx = abs_targets(anchor, pc["chunk"]), 0
            pending = [pc for pc in pending if pc["arrive"] > t]

            # act: track the active plan one target/tick; else reflex-bridge (also creates the drift)
            if strategy != "reflex" and active is not None and aidx < len(active):
                a_apply = bounded_step_to(pose, active[aidx]); aidx += 1
            else:
                a_apply = self._cem(z, pose, z_goal, reflex_samples, reflex_steps) * reflex_gain

            # the TRUE world also gets the unmodeled disturbance (planner/reconciler never see it)
            z, pose = self.env_step(z, pose, a_apply + drift)
            energies.append(self._energy(z, z_goal))
            perrs.append(pose_err())
            poses.append(pose.flatten().cpu().numpy())

        return {"strategy": strategy, "delta": delta, "energy": energies,
                "final_energy": energies[-1], "min_energy": float(np.min(energies)),
                "pose_err": perrs, "final_pose_err": perrs[-1], "min_pose_err": float(np.nanmin(perrs)),
                "poses": np.stack(poses)}

    @torch.no_grad()
    def _cem_plan(self, z, pose, z_goal, samples, cem_steps, rollout):
        """Full CEM plan -> action sequence [1, rollout, 7] (the cloud's multi-step plan)."""
        from utils.mpc_utils import cem
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return cem(
                context_frame=z, context_pose=pose, goal_frame=z_goal,
                world_model=self._step, rollout=rollout, samples=samples,
                cem_steps=cem_steps, topk=max(4, samples // 10),
                momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05, close_gripper=None,
            )

    # ---- diagnostics ------------------------------------------------------
    @torch.no_grad()
    def deep_cem_test(self, z0_np, p0_np, zg_np, *, rollout, samples=96, cem_steps=8):
        """Linchpin: does a single multi-step CEM plan, executed open-loop, drive energy toward 0?"""
        torch.cuda.empty_cache()
        try:
            z0 = torch.tensor(z0_np, device=self.device); p0 = torch.tensor(p0_np, device=self.device)
            z_goal = torch.tensor(zg_np, device=self.device)
            plan = self._cem_plan(z0, p0, z_goal, samples, cem_steps, rollout)  # [1, rollout, 7]
            z, p = z0, p0
            e = [self._energy(z, z_goal)]
            for h in range(rollout):
                z, p = self.env_step(z, p, plan[:, h:h + 1])
                e.append(self._energy(z, z_goal))
            peak = torch.cuda.max_memory_allocated() / 1e9
            return {"rollout": rollout, "samples": samples, "energy": e,
                    "plan_xyz": plan[0, :, :3].float().cpu().numpy(), "peak_gb": peak}
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"rollout": rollout, "samples": samples, "error": "OOM"}


    @torch.no_grad()
    def diagnose(self, z0_np, p0_np, zg_np, *, a_star=None, steps=8, cloud_samples=160, cloud_steps=6):
        """Is the substrate well-behaved? (1) oracle replay of a_star -> energy should fall to ~0.
        (2) greedy CEM -> does it track the oracle (CEM quality)? (3) what xyz does CEM pick?"""
        if a_star is None:
            a_star = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        a = torch.tensor(a_star, device=self.device, dtype=torch.float32).reshape(1, 1, 7)
        z_goal = torch.tensor(zg_np, device=self.device)

        z = torch.tensor(z0_np, device=self.device); p = torch.tensor(p0_np, device=self.device)
        e_oracle = [self._energy(z, z_goal)]
        for _ in range(steps):
            z, p = self.env_step(z, p, a); e_oracle.append(self._energy(z, z_goal))

        z = torch.tensor(z0_np, device=self.device); p = torch.tensor(p0_np, device=self.device)
        e_cem = [self._energy(z, z_goal)]; cem_xyz = []
        for _ in range(steps):
            ac = self._cem(z, p, z_goal, cloud_samples, cloud_steps)
            cem_xyz.append(ac.flatten().float().cpu().numpy()[:3])
            z, p = self.env_step(z, p, ac); e_cem.append(self._energy(z, z_goal))

        return {"e_oracle": e_oracle, "e_cem": e_cem, "cem_xyz": np.stack(cem_xyz)}

    # ---- goal construction ------------------------------------------------
    @torch.no_grad()
    def build_problem(self, ctx_img, goal_img, pose_np, *, goal_steps=5, a_star=None):
        """Encode ctx, build a K-step-reachable goal by rolling the predictor with a fixed action.

        Returns (z0, p0, z_goal) as numpy. The constructed goal is reachable in `goal_steps` steps
        of `a_star`, so energy has room to fall over a multi-step episode (Risk-2 mitigation).
        """
        z0 = self.encode(ctx_img)[:, None]                  # [1,1,N_T,D]
        pose = torch.tensor(pose_np, device=self.device, dtype=torch.float32).reshape(1, 1, 7)
        if a_star is None:
            a_star = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # modest +x translation per step
        a = torch.tensor(a_star, device=self.device, dtype=torch.float32).reshape(1, 1, 7)

        z, p = z0, pose
        e_trace = []
        for _ in range(goal_steps):
            z, p = self.env_step(z, p, a)
            e_trace.append(float(p[:, 0, 0].item()))
        z_goal, p_goal = z, p   # the goal latent AND the goal pose (for exact pose-error metric)
        # sanity: also report the energy of the encoded "adjacent" goal frame for reference
        z_goal_real = self.encode(goal_img)[:, None]
        return {
            "z0": z0.float().cpu().numpy(),
            "p0": pose.float().cpu().numpy(),
            "z_goal": z_goal.float().cpu().numpy(),
            "p_goal": p_goal.float().cpu().numpy(),
            "goal_x_trace": e_trace,
            "energy_real_goal_vs_built": self._energy(z_goal_real, z_goal),
        }

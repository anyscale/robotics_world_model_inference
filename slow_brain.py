"""
slow_brain.py — the cloud "slow brain": V-JEPA 2-AC world-model planning on Ray.

The robot's reflex policy (fast brain) acts in milliseconds. This module is the
*slow* brain: it imagines candidate futures inside a 1.3B action-conditioned world
model and searches (Cross-Entropy Method) for the action that best drives the
imagined latent toward a goal latent. That search is far too heavy to run on-robot
(~tens of seconds/action on an L4), so we run it in the cloud and scale it with Ray.

Substrate: facebookresearch/vjepa2  `vjepa2_ac_vit_giant`  (encoder 1012M + predictor 305M, MIT).
Loaded once per GPU via a Ray actor; weights are cached on shared storage (TORCH_HOME).
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Weights + repo are cached on shared cluster storage so every worker loads instantly.
TORCH_HOME = os.environ.setdefault("TORCH_HOME", "/mnt/cluster_storage/torch_hub")
os.environ.setdefault("HF_HOME", "/mnt/cluster_storage/hf_cache")
_REPO = f"{TORCH_HOME}/hub/facebookresearch_vjepa2_main"
CROP = 256


def _ensure_repo_on_path():
    """The official V-JEPA 2 repo (fetched by torch.hub) carries the planning code."""
    for p in (_REPO, f"{_REPO}/notebooks"):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_world_model(device="cuda"):
    """Load the action-conditioned world model (encoder, predictor) onto `device`."""
    _ensure_repo_on_path()
    encoder, predictor = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_ac_vit_giant", trust_repo=True
    )
    return encoder.to(device).eval(), predictor.to(device).eval()


class WorldModel:
    """Encode frames to latents and plan actions by CEM search in latent space."""

    def __init__(self, encoder, predictor, device="cuda"):
        _ensure_repo_on_path()
        from app.vjepa_droid.transforms import make_transforms

        self.encoder, self.predictor, self.device = encoder, predictor, device
        self.tpf = int((CROP // encoder.patch_size) ** 2)  # tokens per frame
        self.transform = make_transforms(
            random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
            motion_shift=False, crop_size=CROP,
        )

    @torch.no_grad()
    def encode(self, image):
        """HWC uint8 frame -> normalized latent [1, tokens_per_frame, D]."""
        clip = self.transform(np.expand_dims(image, 0))[None, :]
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2)
        clip = clip.repeat(1, 1, 2, 1, 1).to(self.device)
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        return F.layer_norm(h, (h.size(-1),))

    def _step(self, reps, actions, poses):
        from utils.mpc_utils import compute_new_pose

        B, T, N_T, D = reps.size()
        nxt = self.predictor(reps.flatten(1, 2), actions, poses)[:, -self.tpf:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),)).view(B, 1, N_T, D)
        return nxt, compute_new_pose(poses[:, -1:], actions[:, -1:])

    @torch.no_grad()
    def plan(self, z_ctx, pose, z_goal, samples=128, cem_steps=5, rollout=1):
        """CEM search -> (best_action[7], final_energy). Energy = L1(imagined, goal)."""
        from utils.mpc_utils import cem, l1

        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = cem(
                context_frame=z_ctx, context_pose=pose, goal_frame=z_goal,
                world_model=self._step, rollout=rollout, samples=samples,
                cem_steps=cem_steps, topk=max(4, samples // 10),
                momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05, close_gripper=None,
            )  # [1, rollout, 7]
            # Replay the chosen plan to report the energy it achieves vs the goal.
            reps, poses = z_ctx, pose
            for h in range(rollout):
                reps_next, poses = self._step(reps[:, -1:], action[:, h:h + 1], poses)
                reps = torch.cat([reps, reps_next], dim=1)
            energy = l1(reps[:, -1].flatten(1), z_goal.flatten(1)).item()
        return action[0, 0].float().cpu().numpy(), energy


import ray


@ray.remote(num_gpus=1)
class Planner:
    """A 'slow brain' bound to one cloud GPU. Loads the world model once, then plans.

    One Planner per GPU; scale the fleet by creating more. This is the unit Ray
    fans out — deeper search (more compute/plan) or wider throughput (more planners).
    """

    def __init__(self):
        enc, pred = load_world_model(device="cuda")
        self.wm = WorldModel(enc, pred, device="cuda")

    def encode_frame(self, image):
        return self.encode_pair(image, image)[0]

    def encode_pair(self, ctx_img, goal_img):
        """Encode a (context, goal) image pair to latents kept on-device (returned to CPU)."""
        z_ctx = self.wm.encode(ctx_img)[:, None]
        z_goal = self.wm.encode(goal_img)[:, None]
        return z_ctx.float().cpu().numpy(), z_goal.float().cpu().numpy()

    def plan_from_images(self, ctx_img, goal_img, pose, samples=128, cem_steps=5, rollout=1):
        """Full slow-brain action: encode -> CEM search -> (action[7], energy, seconds)."""
        import time

        z_ctx = self.wm.encode(ctx_img)[:, None]
        z_goal = self.wm.encode(goal_img)[:, None]
        p = torch.as_tensor(pose, device="cuda", dtype=torch.float32).reshape(1, 1, 7)
        torch.cuda.synchronize()
        t0 = time.time()
        action, energy = self.wm.plan(z_ctx, p, z_goal, samples=samples,
                                      cem_steps=cem_steps, rollout=rollout)
        torch.cuda.synchronize()
        # Mean-pooled context+goal latent = the feature the distilled fast policy learns from.
        feat = torch.cat([z_ctx.mean(2), z_goal.mean(2)], -1).flatten().float().cpu().numpy()
        return {"action": action, "energy": energy, "seconds": time.time() - t0, "feat": feat}

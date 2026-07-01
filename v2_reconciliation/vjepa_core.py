"""
vjepa_core.py — the shared, Ray-free V-JEPA 2-AC core used by both the cloud slow-brain
(Ray Serve) and the World oracle (Ray actor). Encode frames to latents, roll latents forward
given actions (the dynamics), CEM-plan a chunk, and score energy to a goal latent.

Pure PyTorch; loads the action-conditioned world model from the torch.hub cache on shared storage.
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


class WorldModelCore:
    """Encode / roll-forward / CEM-plan / score, all in V-JEPA latent + 7-D pose space."""

    def __init__(self, device="cuda"):
        _ensure_repo_on_path()
        import logging
        import warnings
        warnings.filterwarnings("ignore")
        logging.getLogger("utils.mpc_utils").setLevel(logging.WARNING)  # silence per-CEM-step "new mean" logs
        from app.vjepa_droid.transforms import make_transforms

        enc, pred = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant", trust_repo=True)
        self.encoder = enc.to(device).eval()
        self.predictor = pred.to(device).eval()
        self.device = device
        self.tpf = int((CROP // self.encoder.patch_size) ** 2)
        self.transform = make_transforms(
            random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
            motion_shift=False, crop_size=CROP,
        )

    @torch.no_grad()
    def encode(self, image):
        """HWC uint8 -> latent [1, 1, tokens_per_frame, D]."""
        clip = self.transform(np.expand_dims(image, 0))[None, :]
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2)
        clip = clip.repeat(1, 1, 2, 1, 1).to(self.device)
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        return F.layer_norm(h, (h.size(-1),))[:, None]

    @torch.no_grad()
    def _step(self, reps, actions, poses):
        from utils.mpc_utils import compute_new_pose
        nxt = self.predictor(reps.flatten(1, 2), actions, poses)[:, -self.tpf:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),)).view(reps.size(0), 1, reps.size(2), reps.size(3))
        return nxt, compute_new_pose(poses[:, -1:], actions[:, -1:])

    @torch.no_grad()
    def step(self, z, pose, action):
        """One dynamics step. z [1,1,N_T,D]; pose/action [1,1,7] -> (z_next, pose_next)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self._step(z, action, pose)

    @torch.no_grad()
    def energy(self, z, z_goal):
        from utils.mpc_utils import l1
        return l1(z[:, -1].flatten(1), z_goal[:, -1].flatten(1)).item()

    @torch.no_grad()
    def cem_plan(self, z, pose, z_goal, samples=32, cem_steps=4, rollout=2):
        """CEM plan -> action chunk [1, rollout, 7]."""
        from utils.mpc_utils import cem
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return cem(
                context_frame=z, context_pose=pose, goal_frame=z_goal,
                world_model=self._step, rollout=rollout, samples=samples,
                cem_steps=cem_steps, topk=max(4, samples // 10),
                momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05, close_gripper=None,
            )

    def t(self, arr):
        """numpy -> device tensor."""
        return torch.as_tensor(arr, device=self.device, dtype=torch.float32)

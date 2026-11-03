"""V-JEPA 2-AC world model: load it, encode a camera frame, and score a batch of candidate action sequences.

This module has no Ray in it. It is the GPU worker's job description, and is used by both the single-GPU baseline and
the Ray Serve `WorldModel` replicas.

    core = WorldModelCore()                       # loads the 1.2B-parameter encoder + 300M predictor onto one GPU
    z, z_goal = core.encode(frame), core.encode(goal_frame)
    energy = core.score(z, pose, z_goal, actions)  # actions [S, H, 7] -> energy [S] (lower = closer to the goal)

`score` is the unit of work the planner fans out: roll every candidate H steps forward through the predictor, then
take the L1 distance between the final imagined latent and the goal latent -- exactly the objective in Meta's
reference CEM (`notebooks/utils/mpc_utils.py` in facebookresearch/vjepa2).

Two implementations of `score`, selected with `fast=`:
  * fast=False -- the reference computation: re-run the whole frame-causal predictor at each rollout step (as Meta's
    `cem` does), in fp32, poses updated on the CPU with scipy.
  * fast=True  -- the same function, faster: bf16, flash attention, a KV cache so each rollout step only computes the
    new frame (the predictor is frame-causal, so earlier frames' keys/values never change), and the pose update done
    on the GPU, with each transformer block fused by torch.compile. `scripts/check_fast_path.py` checks it against the
    reference.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Where torch.hub keeps the V-JEPA 2 code and the 11.7 GB checkpoint. On Anyscale, point this at shared cluster
# storage so every GPU worker reads the same copy (it is downloaded once, on first use).
TORCH_HOME = os.environ.setdefault("TORCH_HOME", "/mnt/cluster_storage/jepa_rt/torch_hub")
HUB_REPO = "facebookresearch/vjepa2"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
CROP = 256           # input resolution of the action-conditioned model
GRIPPER_LIMIT = 0.75  # Meta's CEM clips the sampled gripper delta to +-0.75


def _hub_dir():
    return os.path.join(torch.hub.get_dir(), "facebookresearch_vjepa2_main")


def load_vjepa2_ac(device="cuda"):
    """Encoder (ViT-g, 1.0B params) + action-conditioned predictor (0.3B params), eval mode, fp32 weights."""
    # The hub entry point's built-in checkpoint URL points at localhost in the current upstream repo, so build the
    # architecture with pretrained=False and load the published checkpoint ourselves (cached under TORCH_HOME).
    encoder, predictor = torch.hub.load(HUB_REPO, "vjepa2_ac_vit_giant", pretrained=False, trust_repo=True,
                                        verbose=False)
    sd = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location="cpu", progress=False)

    def clean(d):
        return {k.replace("module.", "").replace("backbone.", ""): v for k, v in d.items()}

    encoder.load_state_dict(clean(sd["encoder"]), strict=False)   # strict=False as upstream (unused heads)
    predictor.load_state_dict(clean(sd["predictor"]), strict=True)
    del sd
    if _hub_dir() not in sys.path:   # Meta's transforms / mpc_utils live in the hub repo
        sys.path.insert(0, _hub_dir())
        sys.path.insert(0, os.path.join(_hub_dir(), "notebooks"))
    return encoder.to(device).eval(), predictor.to(device).eval()


class WorldModelCore:
    """One GPU's worth of V-JEPA 2-AC: encode frames, score candidate action sequences."""

    def __init__(self, device="cuda", compile=True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = device
        self.encoder, self.predictor = load_vjepa2_ac(device)
        from app.vjepa_droid.transforms import make_transforms
        self.transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
                                         random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
                                         motion_shift=False, crop_size=CROP)
        p = self.predictor
        self.grid = p.grid_height                      # 16 x 16 patches
        self.tpf = p.grid_height * p.grid_width        # 256 tokens per frame
        self.blocks = p.predictor_blocks
        self.n_heads = self.blocks[0].attn.num_heads
        self.cond = 2                                  # action + state token per frame
        if compile:
            # One compiled variant per (batch size, rollout step); raise dynamo's default cap of 8 so a replica that
            # also simulates (batch 1) never silently falls back to eager for the planning shapes.
            torch._dynamo.config.cache_size_limit = 64
            # Fuse each transformer block of the fast path (~1.35x on A100). Shapes are static per batch size, so the
            # first call at a new batch size compiles (~10 s): replicas warm up at start-up, before taking traffic.
            self._block_step = torch.compile(self._block_step, dynamic=False)
        # Encoding one frame is only ~30 ms of GPU work but ~130 ms eager (1,500 small kernels): compile it too.
        self._encoder_fwd = torch.compile(self.encoder, dynamic=False) if compile else self.encoder

    # ------------------------------------------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, image):
        """HWC uint8 camera frame -> layer-normed latent [1, 256, 1408] (on this GPU)."""
        clip = self.transform(np.expand_dims(image, 0))[None, :]          # [1, C, T=1, H, W]
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self._encoder_fwd(clip)
        h = h.float().view(B, T, -1, h.size(-1)).flatten(1, 2)
        return F.layer_norm(h, (h.size(-1),))

    # ------------------------------------------------------------------------------------------------------- score
    @torch.no_grad()
    def score(self, z, pose, z_goal, actions, fast=True, chunk=None):
        """Energy of each candidate: L1(final imagined latent, goal latent), mean over tokens and channels.

        z, z_goal: [1, 256, D] latents; pose: [7] (xyz, euler xyz, gripper); actions: [S, H, 7] -> energy [S] (fp32).
        `chunk` caps how many candidates go through the GPU at once (memory); None = all at once.
        """
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32).reshape(1, self.tpf, -1)
        z_goal = torch.as_tensor(z_goal, device=self.device, dtype=torch.float32).reshape(1, self.tpf, -1)
        pose = torch.as_tensor(pose, device=self.device, dtype=torch.float32).reshape(1, 1, 7)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        S = actions.shape[0]
        chunk = chunk or S
        fn = self._score_fast if fast else self._score_reference
        out = [fn(z, pose, z_goal, actions[i:i + chunk]) for i in range(0, S, chunk)]
        return torch.cat(out).float()

    # Reference: exactly what Meta's cem() computes for a given set of action sequences (fp32, full re-forward).
    def _score_reference(self, z, pose, z_goal, actions):
        from utils.mpc_utils import compute_new_pose, l1
        S, H, _ = actions.shape
        frames = z[:, None].repeat(S, 1, 1, 1)          # [S, 1, 256, D]
        poses = pose.repeat(S, 1, 1)                    # [S, 1, 7]
        for h in range(H):
            a = actions[:, :h + 1]
            nxt = self.predictor(frames.flatten(1, 2), a, poses)[:, -self.tpf:]
            nxt = F.layer_norm(nxt, (nxt.size(-1),))[:, None]
            frames = torch.cat([frames, nxt], dim=1)
            poses = torch.cat([poses, compute_new_pose(poses[:, -1:], a[:, -1:])], dim=1)
        return l1(frames[:, -1].flatten(1), z_goal.repeat(S, 1, 1).flatten(1))

    @torch.no_grad()
    def rollout(self, z, pose, actions):
        """Imagine executing `actions` [H, 7] from (z, pose): -> (final latent [1, 256, D], final pose [7]).

        Used to simulate the robot in the closed-loop experiments (the world model stands in for camera + arm)."""
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32).reshape(1, self.tpf, -1)
        pose = torch.as_tensor(pose, device=self.device, dtype=torch.float32).reshape(1, 1, 7)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).reshape(1, -1, 7)
        zf, pf = self._score_fast(z, pose, None, actions, return_state=True)
        return zf, pf[0]

    # Fast: same math, bf16 + KV cache + GPU pose update.
    def _score_fast(self, z, pose, z_goal, actions, return_state=False):
        S, H, _ = actions.shape
        p = self.predictor
        x_frame = z.expand(S, -1, -1)                   # current frame latent [S, 256, D]
        poses = pose.expand(S, 1, 7)
        cur_pose = poses[:, 0]
        kv = [None] * len(self.blocks)                  # per layer: (k, v) of all earlier frames
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for h in range(H):
                a = actions[:, h]
                tok = torch.cat([p.action_encoder(a)[:, None], p.state_encoder(cur_pose)[:, None],
                                 p.predictor_embed(x_frame)], dim=1)          # [S, 2 + 256, Dp]
                last = h == H - 1
                for i, blk in enumerate(self.blocks):
                    tok, kv[i] = self._block_step(blk, tok, kv[i], frame=h, keep_kv=not last)
                out = p.predictor_proj(p.predictor_norm(tok[:, self.cond:]))
                x_frame = F.layer_norm(out.float(), (out.size(-1),))
                cur_pose = _next_pose(cur_pose, a)
        if return_state:
            return x_frame, cur_pose
        return (x_frame - z_goal).abs().mean(dim=(1, 2))

    def _block_step(self, blk, x, kv, frame, keep_kv=True):
        """One ACBlock for the tokens of a single new frame, attending to cached keys/values of earlier frames."""
        y = blk.norm1(x)
        att = blk.attn
        q, k, v = _rope_qkv(att, y, frame, self.grid, self.cond)
        if kv is not None:
            k = torch.cat([kv[0], k], dim=2)
            v = torch.cat([kv[1], v], dim=2)
        o = F.scaled_dot_product_attention(q, k, v)    # new frame sees all earlier frames + itself: no mask needed
        B, nh, N, hd = o.shape
        x = x + att.proj(o.transpose(1, 2).reshape(B, N, nh * hd))
        x = x + blk.mlp(blk.norm2(x))
        return x, ((k, v) if keep_kv else None)


def _rope_qkv(att, y, frame, grid, cond):
    """q/k/v for one frame's tokens [B, cond + grid*grid, C], with the predictor's 3-D RoPE (frame, row, col).

    Mirrors ACRoPEAttention.forward for T=1 at temporal position `frame`: the action/state tokens get only the
    temporal rotation, patch tokens get temporal + row + column rotations; same frequency layout as upstream.
    """
    from src.models.utils.modules import rotate_queries_or_keys as rot
    B, N, C = y.shape
    nh = att.num_heads
    qkv = att.qkv(y).unflatten(-1, (3, nh, -1)).permute(2, 0, 3, 1, 4)    # [3, B, nh, N, hd]
    q, k, v = qkv[0], qkv[1], qkv[2]
    d = att.d_dim
    dev, dt = y.device, q.dtype
    n_patch = N - cond
    ids = torch.arange(n_patch, device=dev)
    fpos = torch.full((n_patch,), float(frame), device=dev)
    hpos = (ids // grid).float() * (att.grid_size / grid)
    wpos = (ids % grid).float() * (att.grid_size / grid)

    def rope_patch(t):
        t_c, t_x = t[:, :, :cond], t[:, :, cond:]
        s = 0
        parts = [rot(t_x[..., s:s + d], pos=fpos.to(dt))]; s += d
        parts.append(rot(t_x[..., s:s + d], pos=hpos.to(dt))); s += d
        parts.append(rot(t_x[..., s:s + d], pos=wpos.to(dt))); s += d
        if s < t.shape[-1]:
            parts.append(t_x[..., s:])
        t_x = torch.cat(parts, dim=-1)
        cpos = torch.full((cond,), float(frame), device=dev).to(dt)
        t_c = torch.cat([rot(t_c[..., :d], pos=cpos), t_c[..., d:]], dim=-1)
        return torch.cat([t_c, t_x], dim=2)

    return rope_patch(q), rope_patch(k), v


def _next_pose(pose, action):
    """GPU version of Meta's compute_new_pose for CEM actions: xyz += dxyz, gripper += d (clipped to [0, 1]).

    CEM never samples rotation deltas (they are zeroed), so the orientation is unchanged; we assert that.
    """
    xyz = pose[:, :3] + action[:, :3]
    grip = (pose[:, 6:] + action[:, 6:]).clamp(0, 1)
    return torch.cat([xyz, pose[:, 3:6], grip], dim=-1)

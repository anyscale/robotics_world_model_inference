"""Real-time planning with the V-JEPA 2-AC world model, distributed with Ray Serve.

Paths and the Ray runtime environment. Importing this module needs no torch, so a CPU-only driver can use it.
"""
import importlib.util
import os
import shutil
import tarfile
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))

# Shared cluster storage: the 11.7 GB checkpoint is downloaded here once and read by every GPU worker.
TORCH_HOME = os.environ.get("TORCH_HOME", "/mnt/cluster_storage/jepa_rt/torch_hub")
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
CHECKPOINT_PATH = os.path.join(TORCH_HOME, "hub", "checkpoints", "vjepa2-ac-vitg.pt")

# The V-JEPA 2 code (model definitions, transforms, example trajectory), pinned to the commit we measured with. The
# Dockerfile bakes it into the image and sets VJEPA2_REPO_DIR; otherwise ensure_assets() fetches it to shared storage.
VJEPA2_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"   # the commit everything here was measured with
VJEPA2_REPO_DIR = os.environ.get("VJEPA2_REPO_DIR", os.path.join(TORCH_HOME, "hub", "facebookresearch_vjepa2_main"))
EXAMPLE_TRAJ = os.path.join(VJEPA2_REPO_DIR, "notebooks", "franka_example_traj.npz")

# Ray runtime environment for every worker: ship this package, and point workers at the same storage. If the image
# lacks timm (the one package V-JEPA 2 needs beyond torch), Ray installs it into a per-job virtualenv, so the cluster
# image itself is never modified. With the provided Dockerfile it is already there and no install happens.
RUNTIME_ENV = {
    "py_modules": [_HERE],
    "env_vars": {"TORCH_HOME": TORCH_HOME, "VJEPA2_REPO_DIR": VJEPA2_REPO_DIR,
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
}
if importlib.util.find_spec("timm") is None:
    RUNTIME_ENV["pip"] = ["timm==1.0.30"]


def ensure_assets():
    """One-time setup on shared storage, before any GPU replica starts. Safe to call every time.

    * the V-JEPA 2 code at the pinned commit, if it is not already present (the Dockerfile bakes it into the image);
    * the V-JEPA 2-AC checkpoint (~12 GB), so replicas read one copy instead of each downloading their own.
    """
    if not os.path.isdir(VJEPA2_REPO_DIR):
        url = f"https://github.com/facebookresearch/vjepa2/archive/{VJEPA2_COMMIT}.tar.gz"
        print(f"fetching V-JEPA 2 code @ {VJEPA2_COMMIT[:7]} -> {VJEPA2_REPO_DIR}")
        parent = os.path.dirname(VJEPA2_REPO_DIR)
        os.makedirs(parent, exist_ok=True)
        tgz, _ = urllib.request.urlretrieve(url)
        with tarfile.open(tgz) as t:
            t.extractall(parent, filter="data")
        tmp = os.path.join(parent, f"vjepa2-{VJEPA2_COMMIT}")
        if not os.path.isdir(VJEPA2_REPO_DIR):        # another process may have won the race
            shutil.move(tmp, VJEPA2_REPO_DIR)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        tmp = CHECKPOINT_PATH + f".part{os.getpid()}"
        print(f"downloading {CHECKPOINT_URL} -> {CHECKPOINT_PATH} (one time, ~12 GB)")
        urllib.request.urlretrieve(CHECKPOINT_URL, tmp)
        os.replace(tmp, CHECKPOINT_PATH)

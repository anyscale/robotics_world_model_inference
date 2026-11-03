"""Real-time planning with the V-JEPA 2-AC world model, distributed with Ray Serve."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Ray runtime environment for every worker. `pip` installs timm (the only package V-JEPA 2 needs beyond the base
# image) into a per-job virtualenv on each node; the image itself is never modified.
TORCH_HOME = os.environ.get("TORCH_HOME", "/mnt/cluster_storage/jepa_rt/torch_hub")
RUNTIME_ENV = {
    "pip": ["timm==1.0.30"],
    "py_modules": [_HERE],
    "env_vars": {"TORCH_HOME": TORCH_HOME, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
}

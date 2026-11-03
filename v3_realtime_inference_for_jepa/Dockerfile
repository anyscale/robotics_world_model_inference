# Container image for real-time V-JEPA 2-AC planning on Ray (Anyscale).
#
# Use it as a custom image for a new Anyscale cluster/workspace, e.g. paste it into the console's
# "Build a container image" form, or build it with the Anyscale CLI from this directory:
#     anyscale image build -f Dockerfile -n jepa-realtime
# then select the image when you create the workspace. Hardware: 8 GPUs with >= 40 GB each (we measured on
# 8x A100-40GB); fewer GPUs work too, use build_app(n_gpus=...) to match.
#
# Not in the image: the 11.7 GB checkpoint. The notebook's first cell calls jepa_rt.ensure_assets(), which downloads
# it once to shared storage ($TORCH_HOME on /mnt/cluster_storage) where every GPU worker reads the same copy.

# Same Ray version, Python and CUDA as the measurements in this repo.
FROM anyscale/ray:2.58.0-py312-cu128

# PyTorch build we measured with (CUDA 12.8 wheels).
RUN pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

# What Meta's V-JEPA 2 code imports (timm, einops, opencv, scipy, tqdm), what the notebook uses (matplotlib, requests),
# and python-docx for scripts/update_blog_docx.py.
RUN pip install --no-cache-dir \
        timm==1.0.30 \
        einops==0.8.2 \
        opencv-python-headless==4.12.0.88 \
        scipy==1.17.0 \
        tqdm==4.67.1 \
        matplotlib==3.10.8 \
        requests==2.32.3 \
        python-docx==1.2.0

# Meta's V-JEPA 2 code (model definitions, transforms, example trajectory), pinned to the commit we measured with.
# jepa_rt loads the model from here via torch.hub (source="local"), so nothing is pulled from GitHub at run time.
ARG VJEPA2_COMMIT=204698b45b3712590f06245fbfba32d3be539812
RUN sudo mkdir -p /opt/vjepa2 && sudo chown ray /opt/vjepa2 && \
    curl -fsSL "https://github.com/facebookresearch/vjepa2/archive/${VJEPA2_COMMIT}.tar.gz" \
        | tar -xz --strip-components=1 -C /opt/vjepa2 && \
    test -f /opt/vjepa2/hubconf.py && test -f /opt/vjepa2/notebooks/franka_example_traj.npz

ENV VJEPA2_REPO_DIR=/opt/vjepa2 \
    TORCH_HOME=/mnt/cluster_storage/jepa_rt/torch_hub \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail the build early if anything the GPU workers import is missing.
RUN python -c "import torch, torchvision, timm, einops, cv2, scipy, tqdm, ray, ray.serve, fastapi; \
print('torch', torch.__version__, '| ray', ray.__version__)"

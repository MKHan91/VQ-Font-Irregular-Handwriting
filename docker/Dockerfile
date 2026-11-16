# ============= VQ-Font-Irregular-Handwriting Dockerfile =============
FROM 10.50.62.199:5050/embodied-ai/docker-base:cu128-torch-cu124
LABEL description="VQ-Font-Irregular-Handwriting task image"

WORKDIR /home/sona

RUN python3 -m pip install --no-cache-dir \
    tensorboard \
    torchmetrics \
    kornia \
    timm \
    gdown \
    sconf \
    omegaconf \
    lmdb \
    scikit-image \
    lpips \
    pytorch-lightning==1.9.0 \
    matplotlib \
    Pillow \
    tqdm \
    einops \
    PyYAML \
    cityscapesscripts \
    scipy \
    opencv-python \
    "numpy<2.0" \
    albumentations \
    pandas \
    seaborn

CMD ["/bin/bash"]

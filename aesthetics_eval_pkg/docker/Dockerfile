# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive     PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-venv git wget ca-certificates     && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip && pip install -r /workspace/requirements.txt

# Optional: install PyTorch w/ CUDA (edit to match your platform)
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY . /workspace
RUN pip install -e .

# Default command prints help
CMD ["aesthetics-eval", "--help"]

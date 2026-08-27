# Base Dockerfile for AIRAS ML Experiments
# This provides a reproducible environment for all experiment stages

# Base image pinned to a version AND its digest.
FROM python:3.11.16-slim-trixie@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# `make` is the evaluation entry point (see Makefile / `make evaluate`)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager.
# Pinned to a version AND its digest: `:latest` moves, so an unpinned uv is a
# floating input to every build. Bump both lines together.
COPY --from=ghcr.io/astral-sh/uv:0.12.6@sha256:88bc6eb1ccd4b82efd0e1b530caffabddf50dc2bf612e66c14ea25b8ee8a4d3d /uv /usr/local/bin/uv

# Set working directory
WORKDIR /workspace

# Copy dependency files. uv.lock is REQUIRED: a missing lock file fails the
# build here rather than silently re-resolving dependencies later.
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv.
# This layer is cached unless pyproject.toml / uv.lock change.
# --locked: install exactly what uv.lock pins AND verify the lock is still in
# sync with pyproject.toml (--frozen would use the lock without checking).
# So the build fails on dependency drift instead of resolving around it.
RUN uv sync --locked --no-cache --group eval

# From here on, every `uv run` (src.main, `make evaluate`, ...) uses the venv
# built above as-is: no resolution, no network, no writes at run time.
# Set as an env var rather than only on CMD so it also covers the commands the
# GitHub workflows pass to `docker run` and the `uv run` calls in the Makefile.
ENV UV_NO_SYNC=1

# Also put the venv first on PATH. The execution platform derives its own entry
# point by reading this repository, and it has produced a bare
# `python -u -m src.main ...` (no `uv run`) for this code. Without this line
# that form runs the image's system interpreter, which has none of the
# dependencies. With it, `uv run python ...` and bare `python ...` are the same
# interpreter, so the run does not depend on which form was derived.
ENV PATH="/workspace/.venv/bin:$PATH"

# Bake the two downloads into the image.
#
# Seyval containers are disposable, and on the BYO Slurm cluster the parent of
# the per-run working directory is mounted read-only, so a run-time download
# cannot be cached anywhere: every run would re-fetch the same ~180 MB over the
# compute node's link. Doing it here means it happens once per image instead.
#
# Only `src/preprocess.py` is copied first, so this layer is invalidated by a
# change to the pinned URL / SHA-256 and not by every edit to the experiment
# code. It is also the single source of truth for that pin — the Dockerfile
# does not repeat it.
COPY src/preprocess.py ./src/preprocess.py
RUN python -c "from src import preprocess; preprocess.load_benchmark_table('.cache')" \
 && python -c "import torchvision; torchvision.datasets.CIFAR10(root='.cache', train=True, download=True)"

# Copy the rest of the application
COPY . .

# Create results directory
RUN mkdir -p .research/results

# Default command (can be overridden in workflow)
CMD ["bash"]

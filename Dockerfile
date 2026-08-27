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
# Everything that installs into the venv happens in ONE RUN.
#
# The builder is kaniko, which walks the whole filesystem to snapshot a layer
# after every RUN and COPY. Once torch and the CUDA libraries are unpacked the
# tree is several GB across tens of thousands of files, so each *additional*
# instruction after this point costs a full-filesystem snapshot — measured at
# ~15 minutes for one. Adding steps here is nearly free; adding them below is
# not. Merge, do not append.
#
# The second half installs the reference implementation of the paper this study
# reproduces — zero-cost proxies, the NAS-Bench-201 network builder, their
# initialisation and their CIFAR dataloader:
#
#   Abdelfattah et al., "Zero-Cost Proxies for Lightweight NAS", ICLR 2021
#   https://github.com/mohsaied/zero-cost-nas  (Apache-2.0), pinned to a commit
#
# --no-build-isolation is required: its setup.py does `import torch` at build
# time and exits if torch is missing, so it must build against the venv created
# by the `uv sync` above. setuptools and gitpython are what that setup.py needs.
RUN uv sync --locked --no-cache --group eval \
 && uv pip install --no-cache setuptools gitpython \
 && uv pip install --no-cache --no-build-isolation \
      "foresight @ git+https://github.com/mohsaied/zero-cost-nas@b5059bc42e2275534f584bc21a2d28ab8427cd8e"

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

# Copy the rest of the application
COPY . .

# NO further RUN/COPY after this point: each one costs kaniko a full-filesystem
# snapshot over the multi-GB venv. `.research/results` is not created here —
# src/inference.py makes it with mkdir(parents=True, exist_ok=True) when it
# writes, which costs nothing at build time.

# Default command (can be overridden in workflow)
CMD ["bash"]

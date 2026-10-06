# syntax=docker/dockerfile:1.7
#
# FoldJAX on a CUDA host. The environment is the frozen lockfile, nothing else;
# no weights are baked in. Mount a FoldJAX store at /foldjax (FOLDJAX_HOME):
# weights, chemistry assets, the compile cache and generated runtimes all live
# there, so they survive the container and are shared across image rebuilds.
#
#   docker build -t foldjax .
#   docker build -t foldjax:af3 --build-arg ALPHAFOLD3=1 .
#   docker build -t foldjax:cu12 --build-arg CUDA_EXTRA=cuda12 \
#     --build-arg CUDA_BASE=nvidia/cuda:12.9.1-base-ubuntu24.04 .
#   docker run --rm --gpus all -v "$HOME/.cache/foldjax:/foldjax" foldjax doctor
#
# See docs/install.md for the full recipe.

ARG UV_VERSION=0.10.3
# JAX's CUDA wheels carry their own CUDA libraries; the base image supplies the
# NVIDIA container runtime contract, and the host only its driver.
ARG CUDA_BASE=nvidia/cuda:13.0.2-base-ubuntu24.04

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${CUDA_BASE}

# `cuda13` or `cuda12`; the two extras conflict by construction.
ARG CUDA_EXTRA=cuda13
# 1 adds `--extra alphafold3` and the toolchain AlphaFold 3 needs to compile its
# C++ extension on first use (CMake fetches its sources with git).
ARG ALPHAFOLD3=0
# 1 adds `--extra openfold3-preprocess` (OpenFold3 raw-job featurization).
ARG OPENFOLD3_PREPROCESS=1
# 1 adds `--extra templates` (Kalign realignment for `--templates auto`).
ARG TEMPLATES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && if [ "$ALPHAFOLD3" = "1" ]; then \
         apt-get install -y --no-install-recommends build-essential zlib1g-dev; \
       fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# uv-managed Python lives in the image, not in a home directory, so the
# container also runs as an unprivileged `--user`.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/foldjax/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /opt/foldjax/src

# Dependencies first, so a source edit does not reinstall 3 GB of wheels.
# `--no-default-groups`: the default groups are the developer's (`dev`, and
# `gpu`, which duplicates the cuda13 extra and would conflict with cuda12).
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    extras="--extra ${CUDA_EXTRA}"; \
    if [ "$ALPHAFOLD3" = "1" ]; then extras="$extras --extra alphafold3"; fi; \
    if [ "$OPENFOLD3_PREPROCESS" = "1" ]; then \
      extras="$extras --extra openfold3-preprocess"; \
    fi; \
    if [ "$TEMPLATES" = "1" ]; then extras="$extras --extra templates"; fi; \
    echo "$extras" > /opt/foldjax/extras; \
    uv sync --frozen --no-default-groups --no-install-project $extras

COPY README.md LICENSE NOTICE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable $(cat /opt/foldjax/extras)

# AlphaFold 3's first-use extension build runs `uv build`, which needs a
# writable cache whichever user the container runs as.
ENV PATH=/opt/foldjax/venv/bin:$PATH \
    FOLDJAX_HOME=/foldjax \
    UV_CACHE_DIR=/tmp/uv-cache
VOLUME ["/foldjax"]
WORKDIR /work

ENTRYPOINT ["foldjax"]
CMD ["--help"]

# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Image stages: dependencies -> builder -> deploy -> docs
#                         \-> ci
# Dependencies: custom packages and all selected pyproject extras (no dev).
# Builder: dependencies plus PhysicsNeMo installed non-editable.
# Deploy: uninstall mlflow/wandb only; physicsnemo stays non-editable from builder.
# CI: dependencies plus dev group, netcdf4 hack, FigNet/Makani, and CI-only packages.
# Python packages use uv (UV_SYSTEM_PYTHON=1). Build-only source and uv caches are
# mounted into RUN instructions so they do not become part of the image layers.

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:26.06-py3
FROM ${BASE_CONTAINER} AS dependencies

ARG TARGETPLATFORM

# Install uv (use system Python for installs; set so --system is default)
# Pinned to 0.11.14 (latest stable as of May 2026) which bundles
# rustls-webpki >= 0.103.13 (fixes GHSA-82j2-j2ch-gfr8).
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/
# Use system Python, allow installs into the externally managed base environment,
# and copy installed files out of the temporary cache mount.
ENV UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy \
    _CUDA_COMPAT_TIMEOUT=90

# Update Python packaging tools; install OS dependencies; and prepare the
# inherited constraint file in one layer. Package downloads live only in the
# BuildKit cache mount, and apt metadata is removed before the layer is committed.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install "pip>=23.2.1" "setuptools>=77.0.3" && \
    apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 zip unzip && \
    git lfs install && \
    FILE="/etc/pip/constraint.txt" && \
    if [ -f "$FILE" ]; then \
        sed -i '/packaging/d' "$FILE"; \
    else \
        echo "File not found: $FILE"; \
    fi && \
    { [ -f /etc/pip/constraint.txt ] || touch /etc/pip/constraint.txt; } && \
    rm -rf /var/lib/apt/lists/*

# Activate the constraint only after the initial
# pip/setuptools bootstrap and constraint-file preparation have completed.
ENV UV_CONSTRAINT=/etc/pip/constraint.txt

#######################################################################
# Step 1: Dependencies that need custom if-else handling (wheels, etc.)
#######################################################################

# Install pyspng for arm64
ARG PYSPNG_ARM64_WHEEL

# Install Numcodecs (separate install: Numcodecs ARM pip has issues)
ARG NUMCODECS_ARM64_WHEEL

# Install onnxruntime (custom wheel for ARM)
ARG ONNXRUNTIME_ARM64_WHEEL

# Install torch-geometric and torch-scatter
ARG TORCH_SCATTER_ARM64_WHEEL
ARG TORCH_SCATTER_AMD64_WHEEL

# Install pyg-lib
ARG PYGLIB_ARM64_WHEEL
ARG PYGLIB_AMD64_WHEEL

# Install torch_cluster
ARG TORCH_CLUSTER_ARM64_WHEEL
ARG TORCH_CLUSTER_AMD64_WHEEL

ARG NATTEN_ARM64_WHEEL
ARG NATTEN_AMD64_WHEEL

# Expose the selected wheel names and CUDA build configuration as one
# image-config entry.
ENV PYSPNG_ARM64_WHEEL=${PYSPNG_ARM64_WHEEL:-unknown} \
    NUMCODECS_ARM64_WHEEL=${NUMCODECS_ARM64_WHEEL:-unknown} \
    ONNXRUNTIME_ARM64_WHEEL=${ONNXRUNTIME_ARM64_WHEEL:-unknown} \
    TORCH_SCATTER_ARM64_WHEEL=${TORCH_SCATTER_ARM64_WHEEL:-unknown} \
    TORCH_SCATTER_AMD64_WHEEL=${TORCH_SCATTER_AMD64_WHEEL:-unknown} \
    PYGLIB_ARM64_WHEEL=${PYGLIB_ARM64_WHEEL:-unknown} \
    PYGLIB_AMD64_WHEEL=${PYGLIB_AMD64_WHEEL:-unknown} \
    TORCH_CLUSTER_ARM64_WHEEL=${TORCH_CLUSTER_ARM64_WHEEL:-unknown} \
    TORCH_CLUSTER_AMD64_WHEEL=${TORCH_CLUSTER_AMD64_WHEEL:-unknown} \
    NATTEN_ARM64_WHEEL=${NATTEN_ARM64_WHEEL:-unknown} \
    NATTEN_AMD64_WHEEL=${NATTEN_AMD64_WHEEL:-unknown} \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX" \
    NATTEN_CUDA_ARCH="8.0;8.6;9.0;10.0;12.0"

# Install relatively stable third-party dependencies as one cacheable layer.
# Only the installer and release-provided wheels participate in its bind-mount
# cache checksum; ordinary PhysicsNeMo source changes do not invalidate it.
RUN --mount=type=bind,source=docker/install-container-dependencies.sh,target=/tmp/install-container-dependencies.sh,ro \
    --mount=type=bind,source=deps,target=/deps,ro \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    bash /tmp/install-container-dependencies.sh

# Install the remaining third-party dependencies from the canonical project
# metadata. Only pyproject.toml participates in this layer's bind-mount cache
# checksum. The torch-sparse build dependency is declared under
# [tool.uv.extra-build-dependencies].
RUN --mount=type=bind,source=pyproject.toml,target=/tmp/pyproject.toml,ro \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    cd /tmp && \
    uv pip install -r pyproject.toml \
        --extra cu13 \
        --extra utils-extras \
        --extra mesh-extras \
        --extra datapipes-extras \
        --extra gnns \
        --extra sym

# Branch the release builder after dependency installation. The CI stage below
# branches directly from dependencies and therefore never installs PhysicsNeMo.
FROM dependencies AS builder

# Install PhysicsNeMo itself without resolving dependencies again. Source-only
# changes now produce a small application layer and reuse both layers above.
RUN --mount=type=bind,target=/physicsnemo,rw \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    cd /physicsnemo && \
    uv pip install --no-deps .

#######################################################################
# CI image: dependencies + dev group + FigNet/Makani + CI-only packages
#######################################################################
FROM dependencies AS ci

ARG TARGETPLATFORM

# Dev dependency group, FigNet/Makani dependencies, and other CI-only specs.
# The source is needed only while resolving/installing the dev group.
RUN --mount=type=bind,target=/physicsnemo,rw \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install "netcdf4>1.7.3" dask && \
    cd /physicsnemo && \
    uv pip install --group dev && \
    FORCE_CUDA_EXTENSION=1 uv pip install --no-build-isolation "torch-harmonics==0.8.0" && \
    uv pip install "tensorly>=0.8.1" "tensorly-torch>=0.4.0" "torchinfo>=1.8" "webdataset>=0.2" && \
    uv pip install "moto[s3]>=5.0.28" && \
    uv pip install "numpy-stl" "scikit-image>=0.24.0" "shapely" && \
    uv pip install "multi-storage-client[boto3]>=0.33.0"

# FigNet/Makani and related CI-only deps
# Install Makani via direct URL
# RUN uv pip install --no-deps "git+https://github.com/NVIDIA/makani.git@v0.2.1#egg=makani"

# E2Grid install
# RUN uv pip install --no-deps --no-build-isolation "git+https://github.com/NVlabs/earth2grid.git@11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3"

#######################################################################
# Deploy image: builder with mlflow/wandb removed; physicsnemo already non-editable from builder
#######################################################################
FROM builder AS deploy

# Remove mlflow and wandb (CVE concerns), then remove any uv cache inherited
# from the base image. Caches used above were temporary BuildKit mounts.
RUN uv pip uninstall mlflow wandb && uv cache clean

# Set Git Hash as a environment variable
ARG PHYSICSNEMO_GIT_HASH
ENV PHYSICSNEMO_GIT_HASH=${PHYSICSNEMO_GIT_HASH:-unknown}

#######################################################################
# Docs image: deploy + docs build dependencies
#######################################################################
FROM deploy AS docs

ARG TARGETPLATFORM

# Install packages for Sphinx build
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install "recommonmark>=0.7.1" "sphinx>=5.1.1" "nvidia-sphinx-theme>=0.0.7" "pydocstyle>=6.1.1" "nbsphinx>=0.8.9" "nbconvert>=6.4.3" "jinja2>=3.0.3" && \
    wget https://github.com/jgm/pandoc/releases/download/3.1.6.2/pandoc-3.1.6.2-1-amd64.deb && \
    dpkg -i pandoc-3.1.6.2-1-amd64.deb && \
    rm -f pandoc-3.1.6.2-1-amd64.deb

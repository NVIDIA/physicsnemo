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
# Image stages: builder -> ci | deploy -> docs
# Builder: all custom if-else deps + all pyproject extras (no dev), project installed non-editable.
# Deploy: uninstall mlflow/wandb only; physicsnemo stays non-editable from builder.
# CI: add dev group, netcdf4 hack, FigNet/Makani, other CI-only packages; physicsnemo uninstalled
# Python packages use uv (UV_SYSTEM_PYTHON=1). Optional: RUN --mount=type=cache,target=/root/.cache/uv and ENV UV_LINK_MODE=copy for faster rebuilds.

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:26.01-py3
FROM ${BASE_CONTAINER} AS builder

ARG TARGETPLATFORM

# Install uv (use system Python for installs; set so --system is default)
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=1
# Base image Python is PEP 668 externally-managed; allow system installs in container
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Update pip and setuptools
RUN uv pip install "pip>=23.2.1" "setuptools>=77.0.3"

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 zip unzip && \
    git lfs install

ENV _CUDA_COMPAT_TIMEOUT=90

# Copy physicsnemo source
COPY . /physicsnemo/

#######################################################################
# Step 1: Dependencies that need custom if-else handling (wheels, etc.)
#######################################################################

# Remove packaging==23.2 from constraint.txt in the PyTorch container
RUN FILE="/etc/pip/constraint.txt" && \
    if [ -f "$FILE" ]; then \
        sed -i '/packaging/d' "$FILE"; \
    else \
        echo "File not found: $FILE"; \
    fi

# Tell uv to respect the container's constraint file (inherited by all stages)
# Create an empty constraint file if one does not exist to avoid uv errors
RUN [ -f /etc/pip/constraint.txt ] || touch /etc/pip/constraint.txt
ENV UV_CONSTRAINT=/etc/pip/constraint.txt

# Install pyspng for arm64
ARG PYSPNG_ARM64_WHEEL
ENV PYSPNG_ARM64_WHEEL=${PYSPNG_ARM64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$PYSPNG_ARM64_WHEEL" != "unknown" ]; then \
        echo "Custom pyspng wheel for $TARGETPLATFORM exists, installing!" && \
        uv pip install /physicsnemo/deps/${PYSPNG_ARM64_WHEEL}; \
    else \
        echo "No custom wheel for pyspng found. Installing pyspng for: $TARGETPLATFORM from pypi" && \
        uv pip install "pyspng>=0.1.0"; \
    fi

# Install Numcodecs (separate install: Numcodecs ARM pip has issues)
ARG NUMCODECS_ARM64_WHEEL
ENV NUMCODECS_ARM64_WHEEL=${NUMCODECS_ARM64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Pip install for numcodecs for $TARGETPLATFORM exists, installing!" && \
        uv pip install numcodecs; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$NUMCODECS_ARM64_WHEEL" != "unknown" ]; then \
        echo "Numcodecs wheel for $TARGETPLATFORM exists, installing!" && \
        uv pip install --reinstall /physicsnemo/deps/${NUMCODECS_ARM64_WHEEL}; \
    else \
        echo "Numcodecs wheel for $TARGETPLATFORM is not present. Will attempt to install from PyPi index, but might fail" && \
        uv pip install numcodecs; \
    fi

# Install vtk and pyvista
ARG VTK_ARM64_WHEEL
ENV VTK_ARM64_WHEEL=${VTK_ARM64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$VTK_ARM64_WHEEL" != "unknown" ]; then \
        echo "VTK wheel $VTK_ARM64_WHEEL for $TARGETPLATFORM exists, installing!" && \
        uv pip install /physicsnemo/deps/${VTK_ARM64_WHEEL}; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Installing vtk for: $TARGETPLATFORM" && \
        uv pip install "vtk>=9.2.6"; \
    else \
        echo "No custom wheel or wheel on PyPi found. Installing vtk for: $TARGETPLATFORM from source" && \
        apt-get update && apt-get install -y libgl1-mesa-dev && \
        git clone https://gitlab.kitware.com/vtk/vtk.git && cd vtk && git checkout tags/v9.4.1 && git submodule update --init --recursive && \
        mkdir build && cd build && cmake -GNinja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON /workspace/vtk/ && ninja && \
        python setup.py bdist_wheel && \
        uv pip install dist/vtk-*.whl && \
        cd ../../ && rm -r vtk; \
    fi
RUN uv pip install "pyvista>=0.40.1"

# Install onnxruntime (custom wheel for ARM)
ARG ONNXRUNTIME_ARM64_WHEEL
ENV ONNXRUNTIME_ARM64_WHEEL=${ONNXRUNTIME_ARM64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        uv pip install "onnxruntime-gpu>1.19.0"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$ONNXRUNTIME_ARM64_WHEEL" != "unknown" ]; then \
        uv pip install --no-deps /physicsnemo/deps/${ONNXRUNTIME_ARM64_WHEEL}; \
    else \
        echo "Skipping onnxruntime_gpu install."; \
    fi

# Install torch-geometric and torch-scatter
RUN uv pip install "torch_geometric>=2.6.1"

ARG TORCH_SCATTER_ARM64_WHEEL
ENV TORCH_SCATTER_ARM64_WHEEL=${TORCH_SCATTER_ARM64_WHEEL:-unknown}

ARG TORCH_SCATTER_AMD64_WHEEL
ENV TORCH_SCATTER_AMD64_WHEEL=${TORCH_SCATTER_AMD64_WHEEL:-unknown}

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX"

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ "$TORCH_SCATTER_AMD64_WHEEL" != "unknown" ]; then \
        echo "Installing torch_scatter for: $TARGETPLATFORM" && \
        uv pip install --reinstall /physicsnemo/deps/${TORCH_SCATTER_AMD64_WHEEL}; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$TORCH_SCATTER_ARM64_WHEEL" != "unknown" ]; then \
        echo "Installing torch_scatter for: $TARGETPLATFORM" && \
        uv pip install --reinstall /physicsnemo/deps/${TORCH_SCATTER_ARM64_WHEEL}; \
    else \
        echo "No custom wheel present for scatter, building from source"; \
        mkdir -p /physicsnemo/deps/; \
        cd /physicsnemo/deps/; \
        git clone https://github.com/rusty1s/pytorch_scatter.git; \
        cd pytorch_scatter; \
        git checkout tags/2.1.2; \
        FORCE_CUDA=1 MAX_JOBS=64 python setup.py bdist_wheel && \
        uv pip install --reinstall dist/*.whl && \
        cd ../ && rm -r pytorch_scatter; \
    fi

# Install pyg-lib
ARG PYGLIB_ARM64_WHEEL
ENV PYGLIB_ARM64_WHEEL=${PYGLIB_ARM64_WHEEL:-unknown}

ARG PYGLIB_AMD64_WHEEL
ENV PYGLIB_AMD64_WHEEL=${PYGLIB_AMD64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ "$PYGLIB_AMD64_WHEEL" != "unknown" ]; then \
        echo "Installing pyg_lib for: $TARGETPLATFORM" && \
        uv pip install --reinstall /physicsnemo/deps/${PYGLIB_AMD64_WHEEL}; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$PYGLIB_ARM64_WHEEL" != "unknown" ]; then \
        echo "Installing pyg_lib for: $TARGETPLATFORM" && \
        uv pip install --reinstall /physicsnemo/deps/${PYGLIB_ARM64_WHEEL}; \
    else \
        echo "No custom wheel present for pyg_lib, building from source"; \
        uv pip install ninja wheel && \
        uv pip install --no-build-isolation "git+https://github.com/pyg-team/pyg-lib.git@0.5.0"; \
    fi

# Install torch_cluster
ARG TORCH_CLUSTER_ARM64_WHEEL
ENV TORCH_CLUSTER_ARM64_WHEEL=${TORCH_CLUSTER_ARM64_WHEEL:-unknown}

ARG TORCH_CLUSTER_AMD64_WHEEL
ENV TORCH_CLUSTER_AMD64_WHEEL=${TORCH_CLUSTER_AMD64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ "$TORCH_CLUSTER_AMD64_WHEEL" != "unknown" ]; then \
        echo "Installing torch_cluster for: $TARGETPLATFORM" && \
        uv pip install --reinstall /physicsnemo/deps/${TORCH_CLUSTER_AMD64_WHEEL}; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$TORCH_CLUSTER_ARM64_WHEEL" != "unknown" ]; then \
        echo "Installing torch_cluster for: $TARGETPLATFORM" && \
        uv pip install --reinstall /physicsnemo/deps/${TORCH_CLUSTER_ARM64_WHEEL}; \
    else \
        echo "No custom wheel present for cluster, building from source"; \
        mkdir -p /physicsnemo/deps/; \
        cd /physicsnemo/deps/; \
        git clone --branch 1.6.3 --depth 1 https://github.com/rusty1s/pytorch_cluster.git; \
        cd pytorch_cluster; \
        FORCE_CUDA=1 MAX_JOBS=64 python setup.py bdist_wheel && \
        uv pip install --reinstall dist/*.whl && \
        cd ../ && rm -r pytorch_cluster; \
    fi


# natten and torch_sparse need torch at build time (--no-build-isolation)
ENV NATTEN_CUDA_ARCH="8.0;8.6;9.0;10.0;12.0"

ARG NATTEN_ARM64_WHEEL
ENV NATTEN_ARM64_WHEEL=${NATTEN_ARM64_WHEEL:-unknown}

ARG NATTEN_AMD64_WHEEL
ENV NATTEN_AMD64_WHEEL=${NATTEN_AMD64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ "$NATTEN_AMD64_WHEEL" != "unknown" ]; then \
        echo "Installing natten for: $TARGETPLATFORM" && \
        uv pip install --reinstall /physicsnemo/deps/${NATTEN_AMD64_WHEEL}; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$NATTEN_ARM64_WHEEL" != "unknown" ]; then \
        echo "Installing natten for: $TARGETPLATFORM" && \
        uv pip install --reinstall /physicsnemo/deps/${NATTEN_ARM64_WHEEL}; \
    else \
        echo "No custom wheel present for natten, building from source"; \
        mkdir -p /physicsnemo/deps/; \
        cd /physicsnemo/deps/; \
        git clone --recursive --branch v0.21.5 --depth 1 https://github.com/SHI-Labs/NATTEN.git; \
        cd NATTEN; \
        MAX_JOBS=64 python setup.py bdist_wheel && \
        uv pip install --reinstall dist/*.whl && \
        cd ../ && rm -r NATTEN; \
    fi

RUN uv pip install --no-build-isolation "torch_sparse"

# All pyproject extras (no dev); installs physicsnemo non-editable
RUN cd /physicsnemo && uv pip install ".[cu13,utils-extras,mesh-extras,datapipes-extras,gnns,perf]"

# Cleanup builder stage
RUN rm -rf /physicsnemo/

#######################################################################
# CI image: builder + dev group + FigNet/Makani + CI-only packages
#######################################################################
FROM builder AS ci

ARG TARGETPLATFORM

# UV: use system Python and respect container constraint (same as builder)
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_CONSTRAINT=/etc/pip/constraint.txt

RUN uv pip install "netcdf4>1.7.3" dask

COPY . /physicsnemo/

# Dev dependency-group (pytest, ruff, etc.)
RUN cd /physicsnemo && uv pip install --group dev

# FigNet/Makani and related CI-only deps
RUN FORCE_CUDA_EXTENSION=1 uv pip install --no-build-isolation "torch-harmonics==0.8.0"
RUN uv pip install "tensorly>=0.8.1" "tensorly-torch>=0.4.0" "torchinfo>=1.8" "webdataset>=0.2"
# Install Makani via direct URL
# RUN uv pip install --no-deps "git+https://github.com/NVIDIA/makani.git@v0.2.1#egg=makani"

# Other CI-only specs (moto, scikit-image, etc.)
RUN uv pip install "moto[s3]>=5.0.28"
RUN uv pip install "numpy-stl" "scikit-image>=0.24.0" "sparse-dot-mkl" "shapely"
RUN uv pip install "multi-storage-client[boto3]>=0.33.0"

# E2Grid install
# RUN uv pip install --no-deps --no-build-isolation "git+https://github.com/NVlabs/earth2grid.git@11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3"

# Uninstall the non-editable physicsnemo from builder
RUN uv pip uninstall nvidia-physicsnemo

# Cleanup
RUN rm -rf /physicsnemo/

#######################################################################
# Deploy image: builder with mlflow/wandb removed; physicsnemo already non-editable from builder
#######################################################################
FROM builder AS deploy

# UV: use system Python and respect container constraint (same as builder)
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_CONSTRAINT=/etc/pip/constraint.txt

# Remove mlflow and wandb (CVE concerns)
RUN uv pip uninstall mlflow wandb

# Set Git Hash as a environment variable
ARG PHYSICSNEMO_GIT_HASH
ENV PHYSICSNEMO_GIT_HASH=${PHYSICSNEMO_GIT_HASH:-unknown}

# Remove uv cache to save image size
RUN uv cache clean

#######################################################################
# Docs image: deploy + docs build dependencies
#######################################################################
FROM deploy AS docs

ARG TARGETPLATFORM

# UV: use system Python and respect container constraint (same as builder)
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_CONSTRAINT=/etc/pip/constraint.txt

# Install packages for Sphinx build
RUN uv pip install "protobuf==3.20.3"
RUN uv pip install "recommonmark==0.7.1" "sphinx==5.1.1" "sphinx-rtd-theme==1.0.0" "pydocstyle==6.1.1" "nbsphinx==0.8.9" "nbconvert==6.4.3" "jinja2==3.0.3"
RUN wget https://github.com/jgm/pandoc/releases/download/3.1.6.2/pandoc-3.1.6.2-1-amd64.deb && dpkg -i pandoc-3.1.6.2-1-amd64.deb

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:25.04-py3
FROM ${BASE_CONTAINER} as builder

ARG TARGETPLATFORM

# Update pip and setuptools
RUN pip install "pip>=23.2.1" "setuptools>=77.0.3"

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

ENV _CUDA_COMPAT_TIMEOUT=90

# copy physicsnemo source
COPY . /physicsnemo/

# Install pyspng for arm64
ARG PYSPNG_ARM64_WHEEL
ENV PYSPNG_ARM64_WHEEL=${PYSPNG_ARM64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$PYSPNG_ARM64_WHEEL" != "unknown" ]; then \
        echo "Custom pyspng wheel for $TARGETPLATFORM exists, installing!" && \
        pip install --no-cache-dir /physicsnemo/deps/${PYSPNG_ARM64_WHEEL}; \
    else \
        echo "No custom wheel for pyspng found. Installing pyspng for: $TARGETPLATFORM from pypi" && \
        pip install --no-cache-dir "pyspng>=0.1.0"; \
    fi

# Install other dependencies
# Remove packaging==23.2 from constraint.txt in the PyTorch container
RUN FILE="/etc/pip/constraint.txt" && \
    if [ -f "$FILE" ]; then \
        sed -i '/packaging/d' "$FILE"; \
    else \
        echo "File not found: $FILE"; \
    fi
RUN pip install --no-cache-dir "nvidia_dali_cuda120>=1.35.0"
RUN pip install --no-cache-dir "h5py>=3.7.0" "netcdf4>=1.6.3" "ruamel.yaml>=0.17.22" "scikit-learn>=1.0.2" "cftime>=1.6.2" "einops>=0.7.0"
RUN pip install --no-cache-dir "hydra-core>=1.2.0" "termcolor>=2.1.1" "wandb>=0.13.7" "pydantic>=1.10.2" "imageio" "moviepy" "tqdm>=4.60.0"

# Install Numcodecs (This needs a separate install because Numcodecs ARM pip install has issues)
# A fix is being added here: https://github.com/zarr-developers/numcodecs/pull/315 but the public release is not ready yet.
ARG NUMCODECS_ARM64_WHEEL
ENV NUMCODECS_ARM64_WHEEL=${NUMCODECS_ARM64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Pip install for numcodecs for $TARGETPLATFORM exists, installing!" && \
        pip install --no-cache-dir numcodecs; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$NUMCODECS_ARM64_WHEEL" != "unknown" ]; then \
        echo "Numcodecs wheel for $TARGETPLATFORM exists, installing!" && \
        pip install --force-reinstall --no-cache-dir /physicsnemo/deps/${NUMCODECS_ARM64_WHEEL}; \
    else \
        echo "Numcodecs wheel for $TARGETPLATFORM is not present. Will attempt to install from PyPi index, but might fail" && \
        pip install --no-cache-dir numcodecs; \
    fi

# install vtk and pyvista
ARG VTK_ARM64_WHEEL
ENV VTK_ARM64_WHEEL=${VTK_ARM64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$VTK_ARM64_WHEEL" != "unknown" ]; then \
        echo "VTK wheel $VTK_ARM64_WHEEL for $TARGETPLATFORM exists, installing!" && \
        pip install --no-cache-dir /physicsnemo/deps/${VTK_ARM64_WHEEL}; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Installing vtk for: $TARGETPLATFORM" && \
        pip install --no-cache-dir "vtk>=9.2.6"; \
    else \
        echo "No custom wheel or wheel on PyPi found. Installing vtk for: $TARGETPLATFORM from source" && \
        apt-get update && apt-get install -y libgl1-mesa-dev && \
        git clone https://gitlab.kitware.com/vtk/vtk.git && cd vtk && git checkout tags/v9.4.1 && git submodule update --init --recursive && \
        mkdir build && cd build && cmake -GNinja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON /workspace/vtk/ && ninja && \
        python setup.py bdist_wheel && \
        pip install --no-cache-dir dist/vtk-*.whl && \
        cd ../../ && rm -r vtk; \
    fi
RUN pip install --no-cache-dir "pyvista>=0.40.1"

# Install DGL, below instructions only work for containers with CUDA >= 12.1
# (https://www.dgl.ai/pages/start.html)
ARG DGL_BACKEND=pytorch
ENV DGL_BACKEND=$DGL_BACKEND
ENV DGLBACKEND=$DGL_BACKEND

ARG DGL_ARM64_WHEEL
ENV DGL_ARM64_WHEEL=${DGL_ARM64_WHEEL:-unknown}

# TODO: this is a workaround as dgl is not yet shipping arm compatible wheels for CUDA 12.x: https://github.com/NVIDIA/physicsnemo/issues/432
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$DGL_ARM64_WHEEL" != "unknown" ]; then \
        echo "Custom DGL wheel $DGL_ARM64_WHEEL for $TARGETPLATFORM exists, installing!" && \
        pip install --no-cache-dir --no-deps /physicsnemo/deps/${DGL_ARM64_WHEEL}; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Installing DGL for: $TARGETPLATFORM" && \
        pip install --no-cache-dir --no-deps dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html; \
    else \
        echo "No custom wheel or wheel on PyPi found. Installing DGL for: $TARGETPLATFORM from source" && \
        git clone https://github.com/dmlc/dgl.git && cd dgl/ && git checkout tags/v2.4.0 && git submodule update --init --recursive && \
        DGL_HOME="/workspace/dgl" bash script/build_dgl.sh -g && \
        cd python && \
        python setup.py install && \
        python setup.py build_ext --inplace && \
        cd ../../ && rm -r /workspace/dgl; \
    fi

# Install onnx
# Need to install Onnx from custom wheel as Onnx does not support ARM wheels
ARG ONNXRUNTIME_ARM64_WHEEL
ENV ONNXRUNTIME_ARM64_WHEEL=${ONNXRUNTIME_ARM64_WHEEL:-unknown}

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        pip install "onnxruntime-gpu>1.19.0"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ "$ONNXRUNTIME_ARM64_WHEEL" != "unknown" ]; then \
        pip install --no-cache-dir --no-deps /physicsnemo/deps/${ONNXRUNTIME_ARM64_WHEEL}; \
    else \
        echo "Skipping onnxruntime_gpu install."; \
    fi

# Install torch-geometric and torch-scatter

RUN pip install --no-cache-dir "torch_geometric>=2.6.1"

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX"
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ -e "/physicsnemo/deps/torch_scatter-2.1.2-cp312-cp312-linux_x86_64.whl" ]; then \
        echo "Installing torch_scatter and for: $TARGETPLATFORM" && \
        pip install --force-reinstall --no-cache-dir /physicsnemo/deps/torch_scatter-2.1.2-cp312-cp312-linux_x86_64.whl; \
    else \
        echo "No custom wheel present for scatter, building from source"; \
        mkdir -p /physicsnemo/deps/; \
        cd /physicsnemo/deps/; \
        git clone https://github.com/rusty1s/pytorch_scatter.git; \
        cd pytorch_scatter; \
        git checkout tags/2.1.2; \
        FORCE_CUDA=1 MAX_JOBS=64 python setup.py bdist_wheel && \
        pip install dist/*.whl --force-reinstall --no-cache-dir && \
        cd ../ && rm -r pytorch_scatter; \
    fi

# cleanup of stage
RUN rm -rf /physicsnemo/

# CI image
FROM builder as ci

ARG TARGETPLATFORM

# TODO: Remove hacky downgrade of netCDF4 package. netCDF4 v1.7.1 has following
# issue: https://github.com/Unidata/netcdf4-python/issues/1343
# This workaround is only added for the CI systems which run pytest only once.
# For more details, refer: https://github.com/NVIDIA/physicsnemo/issues/608
RUN pip install --no-cache-dir "netcdf4>=1.6.3,<1.7.1"

RUN pip install --no-cache-dir "mlflow>=2.1.1"

COPY . /physicsnemo/

# Install torch_cluster
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ -e "/physicsnemo/deps/torch_cluster-1.6.3-cp312-cp312-linux_x86_64.whl" ]; then \
        echo "Installing torch_cluster and for: $TARGETPLATFORM" && \
        pip install --force-reinstall --no-cache-dir /physicsnemo/deps/torch_cluster-1.6.3-cp312-cp312-linux_x86_64.whl; \
    else \
        echo "No custom wheel present for cluster, building from source"; \
        mkdir -p /physicsnemo/deps/; \
        cd /physicsnemo/deps/; \
        git clone https://github.com/rusty1s/pytorch_cluster.git; \
        cd pytorch_cluster; \
        git checkout tags/1.6.3; \
        FORCE_CUDA=1 MAX_JOBS=64 python setup.py bdist_wheel && \
        pip install dist/*.whl --force-reinstall --no-cache-dir && \
        cd ../ && rm -r pytorch_cluster; \
    fi

# TODO: Install pyg-lib

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Installing tensorflow and warp-lang for: $TARGETPLATFORM" && \
        pip install --no-cache-dir "tensorflow>=2.9.0" "warp-lang>=0.6.0"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        echo "Installing tensorflow and warp-lang for: $TARGETPLATFORM is not supported presently"; \
    fi

# Install the fignet and makani dependencies
RUN pip install --no-cache-dir "torch-harmonics>=0.6.5,<0.7.1" "tensorly>=0.8.1" "tensorly-torch>=0.4.0" "jaxtyping>=0.2" "torchinfo>=1.8" "webdataset>=0.2"

# TODO(akamenev): install Makani via direct URL, see comments in pyproject.toml.
RUN pip install --no-cache-dir --no-deps -e git+https://github.com/NVIDIA/modulus-makani.git@v0.1.0#egg=makani

RUN pip install --no-cache-dir "black==22.10.0" "interrogate==1.5.0" "coverage==6.5.0" "protobuf==3.20.3" "moto[s3]>=5.0.28"

# Install scikit-image and stl
RUN pip install --no-cache-dir "numpy-stl" "scikit-image>=0.24.0" "sparse-dot-mkl" "shapely" "numpy<2.0"

# Install MSC
RUN pip install --no-cache-dir "multi-storage-client[boto3]>=0.14.0"

# cleanup of stage
RUN rm -rf /physicsnemo/

# Deployment image
FROM builder as deploy
COPY . /physicsnemo/
RUN cd /physicsnemo/ && pip install .

# Set Git Hash as a environment variable
ARG PHYSICSNEMO_GIT_HASH
ENV PHYSICSNEMO_GIT_HASH=${PHYSICSNEMO_GIT_HASH:-unknown}

# Clean up
RUN rm -rf /physicsnemo/

# Docs image
FROM deploy as docs

ARG TARGETPLATFORM

# Install CI packages
RUN pip install --no-cache-dir "protobuf==3.20.3"
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Installing tensorflow and warp-lang for: $TARGETPLATFORM" && \
        pip install --no-cache-dir "tensorflow==2.9.0" "warp-lang>=0.6.0"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        echo "Installing tensorflow and warp-lang for: $TARGETPLATFORM is not supported presently"; \
    fi
# Install packages for Sphinx build
RUN pip install --no-cache-dir "recommonmark==0.7.1" "sphinx==5.1.1" "sphinx-rtd-theme==1.0.0" "pydocstyle==6.1.1" "nbsphinx==0.8.9" "nbconvert==6.4.3" "jinja2==3.0.3"
RUN wget https://github.com/jgm/pandoc/releases/download/3.1.6.2/pandoc-3.1.6.2-1-amd64.deb && dpkg -i pandoc-3.1.6.2-1-amd64.deb

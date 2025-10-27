#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

set -euo pipefail

echo "🧹 Cleaning up and setting up prerequisites..."
echo "Installing opm-simulators program (including Flow). This may take a few minutes."

# Prompt user for build type
echo "🔧 Choose build configuration:"
echo "   1) CPU-only version (recommended for most users)"
echo "   2) CUDA version (requires NVIDIA GPU and CUDA)"
read -p "Enter your choice [1 or 2]: " build_choice

case $build_choice in
    1)
        BUILD_TYPE="CPU"
        echo "✅ Building CPU-only version..."
        ;;
    2)
        BUILD_TYPE="CUDA"
        echo "✅ Building CUDA version..."
        ;;
    *)
        echo "❌ Invalid choice. Defaulting to CPU-only version."
        BUILD_TYPE="CPU"
        ;;
esac

# If CUDA version chosen, detect GPU info
if [ "$BUILD_TYPE" = "CUDA" ]; then
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi not found. Cannot build CUDA version."
        echo "🔧 Falling back to CPU-only version."
        BUILD_TYPE="CPU"
    else
        GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' 2>/dev/null || echo "Not found")
        echo "⛏️ GPU Arch: sm_${GPU_ARCH}, ⚡ CUDA ${CUDA_VERSION}"
        
        # Validate GPU architecture
        if [ -z "$GPU_ARCH" ]; then
            echo "❌ Could not detect GPU architecture. Falling back to CPU-only version."
            BUILD_TYPE="CPU"
        fi
    fi
fi

apt-get update -y
apt-get install -y software-properties-common
add-apt-repository -y ppa:opm/ppa
apt-get update -y

# Essential build tools and version control
apt-get install -y build-essential cmake gfortran pkg-config git-core

# Documentation tools
apt-get install -y doxygen ghostscript \
  texlive-latex-recommended gnuplot

# Dependencies without MPI (removed mpi-default-dev)
apt-get install -y libsuitesparse-dev \
  libboost-all-dev libtrilinos-zoltan-dev libfmt-dev libcjson-dev
  
# DUNE core modules
apt-get install -y libdune-common-dev \
  libdune-geometry-dev libdune-istl-dev libdune-grid-dev

# === Define workspace ===
cd /workspace/project
BUILD_ROOT=/workspace/project/opm_2024_10_src
INSTALL_DIR=/usr  # Install system-wide
mkdir -p "$BUILD_ROOT" && cd "$BUILD_ROOT"

# === Modules ===
MODULES=(opm-common opm-grid opm-simulators opm-upscaling)

# === Clone all modules ===
for repo in "${MODULES[@]}"; do
  echo "📦 Cloning $repo..."
  rm -rf "$repo" || true
  git clone https://github.com/OPM/$repo.git
  cd $repo
  git checkout release/2024.10/final || true
  cd ..
done

# === Build all modules ===
for repo in "${MODULES[@]}"; do
  echo "⚙️ Building $repo ($BUILD_TYPE version)..."
  cd "$BUILD_ROOT/$repo"
  mkdir -p build && cd build
  
  # Base CMake configuration
  CMAKE_CMD="cmake .. \
    -DCMAKE_INSTALL_PREFIX=\"$INSTALL_DIR\" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_CXX_STANDARD=17 \
    -DBUILD_SHARED_LIBS=ON \
    -DMPI_C_COMPILER=mpicc \
    -DMPI_CXX_COMPILER=mpicxx \
    -DENABLE_MPI=ON \
    -DUSE_MPI=ON \
    -DUSE_OPENMP=ON \
    -DENABLE_OPENMP=ON \
    -DBUILD_TESTING=OFF"
  
  # Add CUDA-specific flags if building CUDA version
  if [ "$BUILD_TYPE" = "CUDA" ]; then
    CMAKE_CMD="$CMAKE_CMD \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
      -DCMAKE_CUDA_STANDARD=17 \
      -DCMAKE_CUDA_FLAGS=\"-std=c++17\" \
      -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH \
      -DUSE_CUDA=ON \
      -DENABLE_CUDA=ON"
  else
    CMAKE_CMD="$CMAKE_CMD \
      -DUSE_CUDA=OFF \
      -DENABLE_CUDA=OFF"
  fi
  
  # Execute CMake command
  eval $CMAKE_CMD
  
  make -j"$(nproc)"
  make install
done

# Verify flow is installed in /usr/bin
if [ -f "/usr/bin/flow" ]; then
    echo "✅ OPM Flow 2024.10/final installed system-wide ($BUILD_TYPE version)."
    echo "➡️ You can now run: flow --version"
else
    echo "❌ Flow executable not found in /usr/bin"
    echo "Checking for flow in other locations..."
    find /usr -name "flow" -type f 2>/dev/null || echo "Flow not found"
fi

echo "🎉 Installation completed successfully!"
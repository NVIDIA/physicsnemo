#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${TARGETPLATFORM:?TARGETPLATFORM must be set by the container builder}"

readonly DEPS_DIR="/deps"
BUILD_ROOT="$(mktemp -d /tmp/physicsnemo-dependencies.XXXXXX)"
readonly BUILD_ROOT

cleanup() {
    rm -rf "${BUILD_ROOT}"
}
trap cleanup EXIT

install_pyspng() {
    if [[ "${TARGETPLATFORM}" == "linux/arm64" && "${PYSPNG_ARM64_WHEEL}" != "unknown" ]]; then
        echo "Custom pyspng wheel for ${TARGETPLATFORM} exists, installing!"
        uv pip install "${DEPS_DIR}/${PYSPNG_ARM64_WHEEL}"
    else
        echo "No custom wheel for pyspng found. Installing pyspng for: ${TARGETPLATFORM} from PyPI"
        uv pip install "pyspng>=0.1.0"
    fi
}

install_numcodecs() {
    if [[ "${TARGETPLATFORM}" == "linux/amd64" ]]; then
        echo "Installing numcodecs from PyPI for ${TARGETPLATFORM}"
        uv pip install numcodecs
    elif [[ "${TARGETPLATFORM}" == "linux/arm64" && "${NUMCODECS_ARM64_WHEEL}" != "unknown" ]]; then
        echo "Installing custom numcodecs wheel for ${TARGETPLATFORM}"
        uv pip install --reinstall "${DEPS_DIR}/${NUMCODECS_ARM64_WHEEL}"
    else
        echo "Custom numcodecs wheel is not present for ${TARGETPLATFORM}; attempting installation from PyPI"
        uv pip install numcodecs
    fi
}

install_onnxruntime() {
    if [[ "${TARGETPLATFORM}" == "linux/amd64" ]]; then
        uv pip install "onnxruntime-gpu>1.19.0"
    elif [[ "${TARGETPLATFORM}" == "linux/arm64" && "${ONNXRUNTIME_ARM64_WHEEL}" != "unknown" ]]; then
        uv pip install --no-deps "${DEPS_DIR}/${ONNXRUNTIME_ARM64_WHEEL}"
    else
        echo "Skipping onnxruntime-gpu installation for ${TARGETPLATFORM}"
    fi
}

install_torch_scatter() {
    local wheel="unknown"
    if [[ "${TARGETPLATFORM}" == "linux/amd64" ]]; then
        wheel="${TORCH_SCATTER_AMD64_WHEEL}"
    elif [[ "${TARGETPLATFORM}" == "linux/arm64" ]]; then
        wheel="${TORCH_SCATTER_ARM64_WHEEL}"
    fi

    if [[ "${wheel}" != "unknown" ]]; then
        echo "Installing torch_scatter wheel for ${TARGETPLATFORM}"
        uv pip install --reinstall "${DEPS_DIR}/${wheel}"
        return
    fi

    echo "No custom torch_scatter wheel present; building from source"
    local source_dir="${BUILD_ROOT}/pytorch_scatter"
    git clone --branch 2.1.2 --depth 1 https://github.com/rusty1s/pytorch_scatter.git "${source_dir}"
    (
        cd "${source_dir}"
        FORCE_CUDA=1 MAX_JOBS=64 python setup.py bdist_wheel
        uv pip install --reinstall dist/*.whl
    )
    rm -rf "${source_dir}"
}

install_pyg_lib() {
    local wheel="unknown"
    if [[ "${TARGETPLATFORM}" == "linux/amd64" ]]; then
        wheel="${PYGLIB_AMD64_WHEEL}"
    elif [[ "${TARGETPLATFORM}" == "linux/arm64" ]]; then
        wheel="${PYGLIB_ARM64_WHEEL}"
    fi

    if [[ "${wheel}" != "unknown" ]]; then
        echo "Installing pyg_lib wheel for ${TARGETPLATFORM}"
        uv pip install --reinstall "${DEPS_DIR}/${wheel}"
        return
    fi

    echo "No custom pyg_lib wheel present; building from source"
    uv pip install ninja wheel
    uv pip install --no-build-isolation "git+https://github.com/pyg-team/pyg-lib.git@0.5.0"
}

install_torch_cluster() {
    local wheel="unknown"
    if [[ "${TARGETPLATFORM}" == "linux/amd64" ]]; then
        wheel="${TORCH_CLUSTER_AMD64_WHEEL}"
    elif [[ "${TARGETPLATFORM}" == "linux/arm64" ]]; then
        wheel="${TORCH_CLUSTER_ARM64_WHEEL}"
    fi

    if [[ "${wheel}" != "unknown" ]]; then
        echo "Installing torch_cluster wheel for ${TARGETPLATFORM}"
        uv pip install --reinstall "${DEPS_DIR}/${wheel}"
        return
    fi

    echo "No custom torch_cluster wheel present; building from source"
    local source_dir="${BUILD_ROOT}/pytorch_cluster"
    git clone --branch 1.6.3 --depth 1 https://github.com/rusty1s/pytorch_cluster.git "${source_dir}"
    (
        cd "${source_dir}"
        FORCE_CUDA=1 MAX_JOBS=64 python setup.py bdist_wheel
        uv pip install --reinstall dist/*.whl
    )
    rm -rf "${source_dir}"
}

install_natten() {
    local wheel="unknown"
    if [[ "${TARGETPLATFORM}" == "linux/amd64" ]]; then
        wheel="${NATTEN_AMD64_WHEEL}"
    elif [[ "${TARGETPLATFORM}" == "linux/arm64" ]]; then
        wheel="${NATTEN_ARM64_WHEEL}"
    fi

    if [[ "${wheel}" != "unknown" ]]; then
        echo "Installing NATTEN wheel for ${TARGETPLATFORM}"
        uv pip install --reinstall "${DEPS_DIR}/${wheel}"
        return
    fi

    echo "No custom NATTEN wheel present; building from source"
    local source_dir="${BUILD_ROOT}/NATTEN"
    git clone --recursive --branch v0.21.5 --depth 1 https://github.com/SHI-Labs/NATTEN.git "${source_dir}"
    (
        cd "${source_dir}"
        MAX_JOBS=64 python setup.py bdist_wheel
        uv pip install --reinstall dist/*.whl
    )
    rm -rf "${source_dir}"
}

main() {
    install_pyspng
    install_numcodecs
    install_onnxruntime
    install_torch_scatter
    install_pyg_lib
    install_torch_cluster
    install_natten
}

main "$@"

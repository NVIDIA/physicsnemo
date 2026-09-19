#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate `.github/ci-requirements.lock` -- the pinned transitive closure of
# `.github/ci-requirements.txt`.
#
# Why this file exists:
#   testmon stores a `system_packages` fingerprint in its cached DB at nightly
#   time (sorted name+major.minor of every dist in the venv, via
#   `importlib.metadata.distributions()`).  PR jobs recompute it and a single
#   mismatch invalidates ALL tests.  `.github/ci-requirements.txt` `==`-pins
#   every direct CI-only dep, but several of them (moto, scikit-image, numpy-stl,
#   shapely, multi-storage-client, tensorstore) are NOT in `uv.lock`, so their
#   transitive closure (responses, xmltodict, jsonpath-ng, lazy-loader,
#   tifffile, pywavelets, imageio, antlr4-python3-runtime, ...) gets resolved
#   fresh against PyPI on every job.  The lock file pins that closure so the
#   resolution at nightly time and the resolution at PR time agree.
#
# When to run:
#   * After bumping any `==` pin in `.github/ci-requirements.txt`.
#   * After a `uv.lock` bump that touches the closure of the CI-only deps
#     (rare; the PyG / cuml / cudf / pyarrow stack moves together).
#   * After bumping the CUDA / Python / uv baseline in the workflow `env:`
#     blocks.
#
# How to run:
#   The lock must be generated in a resolution context that matches the CI
#   runners (Linux x86_64, glibc, CUDA 13.3, Python 3.12).  Running this on
#   macOS or with a different Python will produce a lock incompatible with the
#   runners and silently break the testmon-stability guarantee.
#
#     docker run --rm -v "$PWD:/work" -w /work \
#       nvidia/cuda:13.3.1-devel-ubuntu24.04 \
#       bash .github/regen-ci-deps-lock.sh
#
#   After it completes, review the diff and commit the updated
#   `.github/ci-requirements.lock` alongside whatever change triggered the
#   regen.

set -euo pipefail

# These MUST match the workflow `env:` values in
# .github/workflows/github-{pr,nightly-uv}.yml.  Bump in lockstep.
EXTRAS_TAG="${EXTRAS_TAG:-cu13,natten-cu13,utils-extras,mesh-extras,nn-extras,model-extras,datapipes-extras,uq-extras,gnns,sym,transformer-engine-cu13}"
UV_VERSION="${UV_VERSION:-0.12.5}"
# Matches the `--find-links` URL committed to .github/ci-requirements.txt.
# Bump the torch-X.Y.Z+cu130 segment in lockstep with the locked torch version.
PYG_FIND_LINKS_URL="${PYG_FIND_LINKS_URL:-https://data.pyg.org/whl/torch-2.12.0+cu130.html}"

cd "$(dirname "$0")/.."

# ----------------------------------------------------------------------------
# Bootstrap: install just enough to run uv if the caller landed in a bare
# CUDA container.  Skipped when uv is already on PATH (developer dev box).
# ----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
    echo "::: bootstrapping uv (uv not on PATH) ..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      ca-certificates curl python3 python3-dev python3-venv build-essential
    rm -rf /var/lib/apt/lists/*
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "error: uv not found on PATH and not running as root in an apt-based container." >&2
    echo "Install uv ${UV_VERSION} manually (https://astral.sh/uv/${UV_VERSION}/install.sh) and rerun." >&2
    exit 1
  fi
fi

actual_uv="$(uv --version | awk '{print $2}')"
if [ "$actual_uv" != "$UV_VERSION" ]; then
  echo "::warning::uv version mismatch (have ${actual_uv}, want ${UV_VERSION})."
  echo "::warning::Resolutions across uv versions can disagree; the lock may differ from what CI sees."
fi
echo "uv:         ${actual_uv}"
echo "EXTRAS_TAG: ${EXTRAS_TAG}"

# ----------------------------------------------------------------------------
# Step 1: build .venv from the committed lockfile, exactly as setup-uv-env
# does in CI.  This is the resolution context for the compile below.
# ----------------------------------------------------------------------------
echo "::: uv sync --frozen --group dev ..."
extra_flags=()
IFS=',' read -ra extras <<< "$EXTRAS_TAG"
for e in "${extras[@]}"; do
  e_trimmed="$(echo "$e" | xargs)"
  if [ -n "$e_trimmed" ]; then
    extra_flags+=(--extra "$e_trimmed")
  fi
done
rm -rf .venv
# This is a temporary hack for transformer engine and cuda13.
# It can be removed when this PR merges upstream:
# https://github.com/NVIDIA/TransformerEngine/pull/3251
# That PR was open against v2.18, so we need TE > 2.18
# to resolve before removing this env hack:
#
# The PyG sdists (torch-cluster/scatter/sparse) are excluded from BOTH
# syncs, mirroring setup-uv-env: `uv sync` would build them CPU-only at
# ~10 min each only for Step 2 to replace them, and for torch_cluster the
# CPU-only wheel would be cached keyed on the sdist alone, so Step 2's
# `--reinstall-package torch_cluster` would reuse it instead of rebuilding
# with FORCE_CUDA.  Excluding them here makes Step 2 the only install.
pyg_skip=(
  --no-install-package torch-cluster
  --no-install-package torch-scatter
  --no-install-package torch-sparse
)
UV_LINK_MODE=copy UV_FROZEN=1 \
  uv sync --frozen --group dev "${extra_flags[@]}" "${pyg_skip[@]}" \
  --no-install-package transformer-engine-torch
SP="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
UV_LINK_MODE=copy UV_FROZEN=1 \
  CUDA_HOME="$SP/nvidia/cu13" \
  CPLUS_INCLUDE_PATH="$SP/nvidia/cudnn/include:$SP/nvidia/nccl/include" \
  MAX_JOBS=16 \
  uv sync --frozen --group dev "${extra_flags[@]}" "${pyg_skip[@]}"

# ----------------------------------------------------------------------------
# Step 2: install the CUDA builds of the PyG packages.  scatter/sparse/
# pyg_lib come from the --find-links index; torch_cluster is built from
# its PyPI sdist (no pt212cu130 wheel exists -- see ci-requirements.txt),
# hence --no-build-isolation-package + FORCE_CUDA + the fixed arch list,
# mirroring setup-uv-env.  None of them are present after Step 1, so this
# is their first and only install.  Done BEFORE the compile so the closure records
# the CUDA local version segments (e.g. torch_scatter==2.1.2+pt212cu130)
# that CI will actually install.
# ----------------------------------------------------------------------------
echo "::: install CI-only deps + PyG CUDA wheels ..."
UV_LINK_MODE=copy FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0 \
  uv pip install --python .venv/bin/python \
  --reinstall-package torch_scatter \
  --reinstall-package torch_sparse \
  --reinstall-package torch_cluster \
  --reinstall-package pyg_lib \
  --no-build-isolation-package torch_cluster \
  -r .github/ci-requirements.txt

# ----------------------------------------------------------------------------
# Step 3: capture the post-install venv state as constraints, so the compile
# in step 4 cannot pick transitives that disagree with the lockfile-pinned
# closure (cudf / cuml / pylibcudf / pyarrow / etc.).  uv pip freeze emits
# `name==version` for normal installs and `-e <path>` / `<name> @ <url>` for
# editable + VCS installs; we filter to plain `name==version` lines only so
# the constraints file is `-c` compatible.
# ----------------------------------------------------------------------------
echo "::: capture venv state as constraints ..."
venv_constraints="${RUNNER_TEMP:-/tmp}/uv-venv-state.txt"
trap 'rm -f "$venv_constraints"' EXIT
uv pip freeze --python .venv/bin/python \
  | grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==[^@[:space:]]+$' \
  > "$venv_constraints"
echo "    captured $(wc -l < "$venv_constraints") pinned packages from .venv"

# ----------------------------------------------------------------------------
# Step 4: compile the transitive closure of ci-requirements.txt against the
# captured venv state.  The output is what setup-uv-env passes via
# `--constraint` so the layered install in CI is deterministic.
# ----------------------------------------------------------------------------
echo "::: uv pip compile ..."
# --no-build-isolation: resolving the torch_cluster sdist requires building
# its metadata, and its setup.py imports torch; --python already points at
# the torch-bearing .venv built in step 1.
uv pip compile \
  --python .venv/bin/python \
  --no-build-isolation \
  --find-links "$PYG_FIND_LINKS_URL" \
  --constraint "$venv_constraints" \
  --output-file .github/ci-requirements.lock \
  .github/ci-requirements.txt

echo
echo "::: wrote .github/ci-requirements.lock"
echo "Review the diff and commit the updated lock alongside your change."

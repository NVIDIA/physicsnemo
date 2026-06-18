#!/bin/bash
cat <<'EOF'
#############################################################
# Author: Clement Etienam
# PHYSICSNEMO + PYTORCH (CUDA 12.8) VENV SETUP
# - Attempts official Transformer Engine build (needs cuDNN 9.x)
# - If cuDNN/TE build fails, installs a safe shim (use_te=False)
#############################################################
EOF

set -euo pipefail

echo "🚀 Setting up venv for PhysicsNeMo + PyTorch (CUDA 12.8)"

# --- Basic libs for VTK/mayavi GUI stubs (ok in headless too)
apt-get update
# apt-get install -y --no-install-recommends \
  # libx11-dev libxt6 libgl1-mesa-dev libglu1-mesa-dev libxrender1 \
  # build-essential ninja-build cmake git curl ca-certificates pkg-config \
  # pybind11-dev

# --- Sanity: GPU present?
if ! command -v nvidia-smi >/dev/null; then
  echo "❌ No NVIDIA GPU detected."; exit 1
fi

# --- Pick python
if command -v python3.10 >/dev/null; then
  PYTHON_BIN=python3.10
elif command -v python3.9 >/dev/null; then
  PYTHON_BIN=python3.9
else
  PYTHON_BIN=python3
fi
echo "✅ Using $($PYTHON_BIN --version)"

# --- Create venv
VENV_NAME="physicsnemo_venv"
$PYTHON_BIN -m venv "$VENV_NAME"
# shellcheck source=/dev/null
source "$VENV_NAME/bin/activate"

# --- Toolchain / CUDA env (HPC SDK 25.3 / CUDA 12.8)
export CC=gcc CXX=g++
export CFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC"
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.3/compilers/bin:$PATH
export CUDA_HOME=/opt/nvidia/hpc_sdk/LLinux_aarch64/25.3/cuda/12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export CPATH=$CUDA_HOME/include:${CPATH:-}
export NVTE_CUDA_INCLUDE_PATH=$CUDA_HOME/include

echo "✅ Python: $(python --version)"
echo "✅ Pip:    $(pip --version)"

# --- Clean conflicting installs
pip uninstall -y torch torchvision torchaudio \
  transformer-engine transformer_engine dgl || true

# --- Base wheels
pip install --upgrade pip setuptools wheel typing-extensions==4.12.2 --no-build-isolation
pip install Cython numpy --no-build-isolation

# --- Install PyTorch 2.6.0 (CUDA 12.8)
echo "🔥 Installing PyTorch 2.6.0 (cu128)"
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio --no-build-isolation

# --- Try to ensure cuDNN headers are present (needed by TE build)
NEED_CUDNN=0
if [ ! -f "$CUDA_HOME/include/cudnn.h" ] && [ ! -f "/usr/include/cudnn.h" ]; then
  NEED_CUDNN=1
fi

if [ "$NEED_CUDNN" -eq 1 ]; then
  echo "🔧 cuDNN headers not found. Installing cuDNN 9 for CUDA 12..."
  # Install NVIDIA CUDA repo keyring if not present
  # if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    # curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/Linux_aarch64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
    # dpkg -i /tmp/cuda-keyring.deb
    # #apt-get update
  # fi
  # Install cuDNN 9 runtime + dev headers for CUDA 12
  # apt-get install -y --no-install-recommends \
    # libcudnn9-cuda-12 libcudnn9-dev-cuda-12

  # Point includes explicitly (covers both default and alt layouts)
  if [ -f "/usr/include/cudnn.h" ]; then
    export CPATH=/usr/include:${CPATH}
  fi
  if [ -d "/usr/include/x86_64-linux-gnu" ]; then
    export CPATH=/usr/include/Linux_aarch64-linux-gnu:${CPATH}
  fi
fi

echo "🔎 Check cudnn.h..."
if [ -f "$CUDA_HOME/include/cudnn.h" ]; then
  echo "   ✔ found at \$CUDA_HOME/include"
elif [ -f "/usr/include/cudnn.h" ]; then
  echo "   ✔ found at /usr/include"
else
  echo "   ⚠️ cudnn.h still not found; TE build may fail (will fallback to shim)."
fi

# --- Attempt official Transformer Engine install (PyTorch extension)
TE_OK=0
echo "⚙️ Installing Transformer Engine (PyTorch extension)..."
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1  # avoid OOM
set +e
pip install --no-build-isolation "transformer_engine[pytorch]" && TE_OK=1
set -e

# If the wheel-only meta resolved without torch ext, try GitHub stable
if [ "$TE_OK" -eq 1 ]; then
  python - <<'PY'
try:
    import transformer_engine.pytorch as te
    print("✅ TE import OK (PyPI).")
except Exception as e:
    print("❌ TE import failed after PyPI:", e); raise SystemExit(1)
PY
else
  echo "🔁 Retrying TE from GitHub @stable..."
  pip uninstall -y transformer-engine transformer_engine || true
  set +e
  pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
  TE_PIP_STATUS=$?
  set -e
  if [ $TE_PIP_STATUS -eq 0 ]; then
    python - <<'PY'
try:
    import transformer_engine.pytorch as te
    print("✅ TE import OK (GitHub stable).")
except Exception as e:
    print("❌ TE import still failing:", e); raise SystemExit(1)
PY
    TE_OK=1
  fi
fi

# --- If TE still not importable, install a harmless shim (works with use_te=False)
if [ "$TE_OK" -ne 1 ]; then
  echo "🧩 Installing Transformer Engine shim (safe with use_te=False)..."
  python - <<'PY'
import os, site, pathlib, textwrap
sd = site.getsitepackages()[0]
p = pathlib.Path(sd) / "transformer_engine" / "pytorch"
p.mkdir(parents=True, exist_ok=True)
(p / "__init__.py").write_text("# shim for use_te=False\n__all__ = []\n")
print("Shim installed at:", p)
PY
  export TRANSFORMER_ENGINE_DISABLE=1
  echo "ℹ️ Using shim; set 'use_te=False' in Transolver."
else
  echo "✅ Transformer Engine ready. You can use 'use_te=True' or keep it False."
fi

# --- PhysicsNeMo + CuPy + small utils
pip install nvidia-physicsnemo==1.3.0 cupy-cuda12x --no-build-isolation
pip install vtk termcolor
pip install \
  xlsxwriter PyWavelets scikit-mps kneed pyDOE FyeldGenerator py-cpuinfo gdown pyvista \
  gstools scikit-image accelerate einops loky xgboost numba scikit-learn pandas openpyxl \
  gpytorch mlflow tqdm wandb pillow sympy fsspec pyaml --no-build-isolation mayavi
pip install hydra-core --no-build-isolation h5py filelock==3.14 setuptools==77.0.3 fsspec==2025.9.0

# --- Verifications
echo "🧪 System checks..."
nvidia-smi
command -v nvcc && nvcc --version || echo "⚠️ nvcc not found (ok if you only need runtime)."
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "is_available:", torch.cuda.is_available())
PY

# --- Test PhysicsNeMo import + Transolver instantiation with use_te=False
python - <<'PY'
try:
    from physicsnemo.models.transolver import Transolver
    m = Transolver(
        functional_dim=4, out_dim=1,
        n_layers=8, n_hidden=256, n_head=8,
        use_te=False, unified_pos=True, structured_shape=(22,46,112),
    )
    print("✅ PhysicsNeMo Transolver constructed (use_te=False).")
except Exception as e:
    print("❌ PhysicsNeMo import failed:", e)
PY

pip freeze > installed_physicsnemo_env.txt

echo ""
echo "#############################################################"
echo "✅ Setup complete."
echo "To activate later: source $VENV_NAME/bin/activate"
echo "#############################################################"

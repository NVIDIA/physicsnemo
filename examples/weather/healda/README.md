<!-- markdownlint-disable -->
# HealDA: Highlighting the importance of initial errors in end-to-end AI weather forecasts

<p align="center">
<img src="../../../docs/img/healda.png" width="800"/>
</p>

[📖 Paper](https://d1qx31qr3h6wln.cloudfront.net/publications/healDA-arxiv-v1.pdf) · 📦 Checkpoints (coming soon)

---

## Problem Overview

Machine-learning (ML) weather models now rival leading numerical weather prediction (NWP) systems in medium-range skill. However, almost all still rely on NWP data assimilation (DA) to provide initial conditions, tying them to expensive infrastructure and limiting the practical speed and accuracy gains of ML.

**HealDA** is a global ML-based data assimilation system that maps satellite and conventional observations (microwave sounders, aircraft, radiosondes, surface stations) to a 1° atmospheric state on the HEALPix grid. HealDA analyses can initialize off-the-shelf ML forecast models (e.g., FourCastNet3, Aurora, FengWu) without fine-tuning, enabling end-to-end ML weather forecasting with less than one day loss of skill compared to ERA5 initialization.

---

## Installation

### Using uv (recommended)

```bash
# 1. From PhysicsNeMo root directory
cd /path/to/physicsnemo

# 2. Create .venv and install PNM
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install earth2grid
uv pip install setuptools hatchling
uv pip install --no-build-isolation \
    "earth2grid @ https://github.com/NVlabs/earth2grid/archive/main.tar.gz"

# 5. Install healda dependencies
uv pip install -r examples/weather/healda/requirements.txt

# 6. Run commands
cd examples/weather/healda
python examples/weather/healda/train.py --help
PYTHONPATH=. pytest tests/ -v
```

### Using pip

```bash
# 1. Install PhysicsNeMo
pip install nvidia-physicsnemo

# 2. Install earth2grid
pip install setuptools hatchling
pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz

# 3. Install example dependencies
pip install -r requirements.txt
```

> **Warning:** Include `--no-build-isolation` when installing earth2grid to avoid building against the wrong PyTorch version.

---

## Data Preparation

HealDA requires preprocessed observation data and ERA5 target fields.

See [`datasets/etl/`](datasets/etl/) for ETL scripts to prepare observation data into a parquet data format. We source observational data from the UFS Replay.

---

## Training

```bash
python train.py --help
```

<!-- TODO: Add training configuration and examples -->

---

## Inference

```bash
python inference.py --help
```

<!-- TODO: Add inference examples -->

---

## Citation

```bibtex
@article{healda2025,
  title={HealDA: Highlighting the importance of initial errors in end-to-end AI weather forecasts},
  author={Gupta, Aayush and Subramaniam, Akshay and Pritchard, Michael S. and Kashinath, Karthik and Frolov, Sergey and Lieberman, Kelsey and Miller, Christopher and Silverman, Nicholas and Brenowitz, Noah D.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

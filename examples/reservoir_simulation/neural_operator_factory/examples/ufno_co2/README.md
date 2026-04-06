# U-FNO for CO2 Sequestration

Reproduces the main results from:

> **U-FNO — An enhanced Fourier neural operator-based**
> **deep-learning model for multiphase flow**
> Wen, G., Li, Z., Azizzadenesheli, K., Anandkumar, A., & Benson, S. M. (2022).
> *Advances in Water Resources*, 163, 104180.
> [arXiv:2109.03697](https://arxiv.org/abs/2109.03697)

## Overview

This example trains three FNO-based architectures on the CO2 sequestration dataset
for both **gas saturation** and **pressure buildup** prediction:

| Architecture | FNO Layers | U-Net Layers | Conv Layers | ~Parameters |
|--------------|-----------|-------------|-------------|-------------|
| **FNO**      | 6         | 0           | 0           | 31M         |
| **Conv-FNO** | 3         | 0           | 3           | 31M         |
| **U-FNO**    | 3         | 3           | 0           | 33M         |

All models use width=36, Fourier modes (10, 10, 10), and a 2-layer MLP decoder (36→128→1).

## Dataset

- **Grid**: 96 (vertical) × 200 (radial) × 24 (time steps)
- **Input channels**: 12 (permeability, porosity, injection config, scalar params, grid widths)
- **Output**: Gas saturation `sg` or pressure buildup `dP` (separate models)
- **Samples**: 4,500 train / 500 val / 500 test

The dataset is publicly available at:
<https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo?usp=sharing>

## Usage

All commands are run from the `neural_operator_factory/` directory.

### Training (SLURM)

```bash
# U-FNO on gas saturation (flagship result)
sbatch examples/ufno_co2/train.sbatch U-FNO saturation_training_config

# U-FNO on pressure buildup
sbatch examples/ufno_co2/train.sbatch U-FNO pressure_training_config

# FNO baseline on saturation
sbatch examples/ufno_co2/train.sbatch FNO saturation_training_config

# FNO on pressure
sbatch examples/ufno_co2/train.sbatch FNO pressure_training_config

# Conv-FNO on saturation
sbatch examples/ufno_co2/train.sbatch Conv-FNO saturation_training_config

# Conv-FNO on pressure
sbatch examples/ufno_co2/train.sbatch Conv-FNO pressure_training_config

# Run all 6 experiments
for arch in FNO Conv-FNO U-FNO; do
    for var in saturation_training_config pressure_training_config; do
        sbatch examples/ufno_co2/train.sbatch $arch $var
    done
done
```

### Evaluation (SLURM)

```bash
# Evaluate best gas saturation model (auto-detects checkpoint)
sbatch examples/ufno_co2/eval.sbatch saturation

# Evaluate best pressure buildup model
sbatch examples/ufno_co2/eval.sbatch pressure

# Explicit checkpoint
CHECKPOINT=checkpoints/best_model_saturation_ufno_custom.pth \
    sbatch examples/ufno_co2/eval.sbatch saturation
```

## Results

### Paper Reference (Table I.14)

#### Gas Saturation — Test Set

| Model | MPE (mean) | MPE (std) | R² plume |
|-------|-----------|----------|----------|
| FNO | 0.0276 | 0.0160 | 0.961 |
| Conv-FNO | 0.0224 | 0.0125 | 0.970 |
| **U-FNO** | **0.0161** | **0.0105** | **0.981** |

#### Pressure Buildup — Test Set

| Model | MRE (mean) | MRE (std) | R² |
|-------|-----------|----------|-----|
| FNO | 0.0082 | 0.0052 | 0.989 |
| Conv-FNO | 0.0078 | 0.0048 | 0.990 |
| **U-FNO** | **0.0068** | **0.0045** | **0.992** |

### NOF Reproduction (this example, 500 test samples)

#### Gas Saturation

| Model | NOF MPE | Paper MPE | NOF R² | Paper R² |
|-------|---------|-----------|--------|----------|
| FNO | 0.0303 | 0.0276 | 0.984 | 0.961 |
| Conv-FNO | 0.0234 | 0.0224 | 0.988 | 0.970 |
| **U-FNO** | **0.0182** | **0.0161** | **0.993** | **0.981** |

#### Pressure Buildup

| Model | NOF MRE | Paper MRE | NOF R² | Paper R² |
|-------|---------|-----------|--------|----------|
| FNO | 0.0089 | 0.0082 | 0.976 | 0.989 |
| Conv-FNO | 0.0087 | 0.0078 | 0.984 | 0.990 |
| **U-FNO** | **0.0068** | **0.0068** | **0.991** | **0.992** |

### Training Time (8× GPU DDP)

| Model | Saturation (100 ep) | Pressure (140 ep) |
|-------|--------------------|--------------------|
| FNO | ~39 min | ~54 min |
| Conv-FNO | ~55 min | ~78 min |
| U-FNO | ~72 min | ~102 min |

## Loss Function

Matches the paper's Equation 12: relative L2 loss +
radial derivative regularization.
Active cell masking is applied
(cells outside the reservoir are zero-padded).

## Files

```text
ufno_co2/
├── README.md
├── train.sbatch                       # SLURM training (8 GPU)
├── eval.sbatch                        # SLURM evaluation (1 GPU)
├── evaluate_pressure.py               # Pressure eval with dnorm_dP
├── evaluate_saturation.py             # Saturation eval with MPE
└── conf/
    ├── FNO/                           # Pure FNO (6 Fourier layers)
    │   ├── model_config.yaml
    │   ├── saturation_training_config.yaml
    │   └── pressure_training_config.yaml
    ├── Conv-FNO/                      # Conv-FNO (3 Fourier + 3 Conv)
    │   ├── model_config.yaml
    │   ├── saturation_training_config.yaml
    │   └── pressure_training_config.yaml
    └── U-FNO/                         # U-FNO (3 Fourier + 3 U-Net)
        ├── model_config.yaml
        ├── saturation_training_config.yaml
        └── pressure_training_config.yaml
```

## Reference

```bibtex
@article{wen2022ufno,
  title={{U-FNO--An enhanced Fourier neural operator-based
    deep-learning model for multiphase flow}},
  author={Wen, Gege and Li, Zongyi and Azizzadenesheli,
    Kamyar and Anandkumar, Anima and Benson, Sally M},
  journal={Advances in Water Resources},
  volume={163},
  pages={104180},
  year={2022},
  publisher={Elsevier}
}
```

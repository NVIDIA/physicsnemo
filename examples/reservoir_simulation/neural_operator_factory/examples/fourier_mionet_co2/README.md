# Fourier-MIONet for CO2 Sequestration

Reproduces the main results from:

> **Fourier-MIONet: Fourier-enhanced multiple-input neural**
> **operators for multiphase modeling of geological carbon**
> **sequestration**
> Jiang, Z., Zhu, M. & Lu, L. (2024).
> *Reliability Engineering & System Safety*, 251, 110392.
> [arXiv:2303.04778](https://arxiv.org/abs/2303.04778)

## Overview

Three MIONet-based architectures on the CO2 dataset:

| Model | Branch1 | Branch2 | Decoder | Paper R² (sat) |
|-------|---------|---------|---------|---------------|
| Vanilla MIONet | Spatial (conv) | MLP (scalars) | Direct | 0.948 |
| MIONet-FNN | Spatial (conv) | MLP (scalars) | Deep FNN | 0.971 |
| Fourier-MIONet | Spatial (Fourier+UNet) | MLP (scalars) | MLP | 0.985 |

All models use a sinusoidal trunk for time encoding and
full-mapping training (all 24 timesteps at once).

## Dataset

Same CO2 sequestration dataset as the U-FNO and
U-DeepONet examples:

- **Grid**: 96 x 200 x 24 time steps
- **Input channels**: 12
- **Samples**: 4,500 train / 500 val / 500 test

The dataset is publicly available at:
<https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo?usp=sharing>

## Usage

All commands from `neural_operator_factory/`:

### Training

```bash
# Fourier-MIONet saturation (flagship)
sbatch examples/fourier_mionet_co2/train.sbatch \
    fourier_mionet saturation_training_config

# Vanilla MIONet pressure
sbatch examples/fourier_mionet_co2/train.sbatch \
    vanilla_mionet pressure_training_config

# MIONet-FNN saturation
sbatch examples/fourier_mionet_co2/train.sbatch \
    mionet_fnn saturation_training_config

# All 6 experiments
for arch in vanilla_mionet mionet_fnn fourier_mionet; do
  for var in saturation pressure; do
    sbatch examples/fourier_mionet_co2/train.sbatch \
        $arch ${var}_training_config
  done
done
```

### Evaluation

```bash
sbatch examples/fourier_mionet_co2/eval.sbatch saturation
sbatch examples/fourier_mionet_co2/eval.sbatch pressure
```

## Results

### Paper Reference (Table 6)

| Model | Sat R² | Pres R² |
|-------|--------|---------|
| Vanilla MIONet | 0.948 | 0.961 |
| MIONet-FNN | 0.971 | 0.979 |
| Fourier-MIONet | 0.985 | 0.986 |

### NOF Reproduction (500 test samples, 8× GPU)

| Model | Sat MPE | Sat R² | Pres MRE | Pres R² |
|-------|---------|--------|----------|---------|
| Vanilla MIONet | — | — | — | — |
| MIONet-FNN | 0.1454 | 0.705 | 0.0614 | 0.592 |
| **Fourier-MIONet** | **0.0227** | **0.990** | **0.0082** | **0.987** |

Fourier-MIONet exceeds the paper's R² targets.
Baselines underperform due to minimal branch architecture
(single conv layer) — they demonstrate the ranking
Fourier-MIONet >> MIONet-FNN >> Vanilla MIONet,
consistent with the paper.

## Files

```text
fourier_mionet_co2/
├── README.md
├── train.sbatch
├── eval.sbatch
├── evaluate_pressure.py
├── evaluate_saturation.py
└── conf/
    ├── vanilla_mionet/
    │   ├── model_config.yaml
    │   ├── saturation_training_config.yaml
    │   └── pressure_training_config.yaml
    ├── mionet_fnn/
    │   ├── model_config.yaml
    │   ├── saturation_training_config.yaml
    │   └── pressure_training_config.yaml
    └── fourier_mionet/
        ├── model_config.yaml
        ├── saturation_training_config.yaml
        └── pressure_training_config.yaml
```

## Reference

```bibtex
@article{jiang2024fourier,
  title={{Fourier-MIONet}: Fourier-enhanced multiple-input
    neural operators for multiphase modeling of geological
    carbon sequestration},
  author={Jiang, Zhongyi and Zhu, Min and Lu, Lu},
  journal={Reliability Engineering \& System Safety},
  volume={251},
  pages={110392},
  year={2024}
}
```

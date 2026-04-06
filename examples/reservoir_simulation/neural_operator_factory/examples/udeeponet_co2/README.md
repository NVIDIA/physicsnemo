# U-DeepONet for CO2 Sequestration

Reproduces the main results from:

> **U-DeepONet: U-Net enhanced deep operator network for**
> **geologic carbon sequestration**
> Diab, W. & Al Kobaisi, M. (2024).
> *Scientific Reports*, 14, 21298.
> [doi:10.1038/s41598-024-72393-0](https://doi.org/10.1038/s41598-024-72393-0)

## Overview

The U-DeepONet uses a DeepONet architecture with 3 U-Net
blocks in the branch network and a sinusoidal trunk for time
encoding. No Fourier layers are used, making it significantly
faster to train than U-FNO while matching or exceeding accuracy.

| Setting | Saturation | Pressure |
|---------|-----------|----------|
| Width (f) | 64 | 96 |
| Branch | 3 × UNet2D + ReLU | Same |
| Trunk | 10-layer sin FNN | Same |
| Decoder | FC (f → 1) | Same |
| Learning rate | 0.0007 | 0.0006 |

## Dataset

Same CO2 sequestration dataset as the U-FNO example:

- **Grid**: 96 × 200 × 24 time steps
- **Input channels**: 12
- **Samples**: 4,500 train / 500 val / 500 test

The dataset is publicly available at:
<https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo?usp=sharing>

## Usage

All commands from `neural_operator_factory/`:

### Training

```bash
# Gas saturation (width=64)
sbatch examples/udeeponet_co2/train.sbatch saturation_training_config

# Pressure buildup (width=96)
sbatch examples/udeeponet_co2/train.sbatch pressure_training_config
```

### Evaluation

```bash
sbatch examples/udeeponet_co2/eval.sbatch saturation
sbatch examples/udeeponet_co2/eval.sbatch pressure
```

## Results

### Paper Reference (Table 4)

| Variable | MPE/MRE | MAE | R² |
|----------|---------|-----|-----|
| Saturation | 0.0158 | 0.0146 | 0.985 |
| Pressure | 0.0072 | 0.64 | 0.994 |

### NOF Reproduction (500 test samples, 8× GPU)

| Variable | NOF MPE/MRE | Paper | NOF R² | Paper R² |
|----------|------------|-------|--------|----------|
| Saturation | 0.0195 | 0.0158 | 0.991 | 0.985 |
| Pressure | 0.0082 | 0.0072 | 0.992 | 0.994 |

### Training Time (8× GPU DDP)

| Variable | Epochs | Time |
|----------|--------|------|
| Saturation | 100 | ~14 min |
| Pressure | 140 | ~19 min |

## Files

```text
udeeponet_co2/
├── README.md
├── train.sbatch
├── eval.sbatch
├── evaluate_pressure.py
├── evaluate_saturation.py
└── conf/
    ├── model_config.yaml                  # U-DeepONet (width=64)
    ├── saturation_training_config.yaml    # lr=0.0007
    └── pressure_training_config.yaml      # lr=0.0006, width=96
```

## Reference

```bibtex
@article{diab2024udeeponet,
  title={{U-DeepONet}: {U-Net} enhanced deep operator
    network for geologic carbon sequestration},
  author={Diab, Waleed and Al Kobaisi, Mohammed},
  journal={Scientific Reports},
  volume={14},
  pages={21298},
  year={2024},
  publisher={Nature Publishing Group}
}
```

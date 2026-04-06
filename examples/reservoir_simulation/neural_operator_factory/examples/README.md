# Neural Operator Factory — Examples

Each subdirectory is a self-contained example with its own Hydra config files
and SLURM batch scripts. All examples use the shared training script at
`training/train.py` and per-example evaluation scripts.

## Running an Example

All jobs are submitted from the `neural_operator_factory/` directory:

```bash
cd examples/reservoir_simulation/neural_operator_factory

# Training (8-GPU via SLURM)
sbatch examples/ufno_co2/train.sbatch saturation_ufno

# Evaluation (single GPU via SLURM)
sbatch examples/ufno_co2/eval.sbatch saturation
```

## Available Examples

| Example | Description | Dataset | Architectures |
|---------|-------------|---------|---------------|
| [ufno_co2](ufno_co2/) | Reproduce U-FNO paper (Wen et al. 2022) | CO2 sequestration (3D) | FNO, Conv-FNO, U-FNO |
| [udeeponet_co2](udeeponet_co2/) | Reproduce U-DeepONet paper (Diab & Al Kobaisi 2024) | CO2 sequestration (3D) | U-DeepONet |
| [fourier_mionet_co2](fourier_mionet_co2/) | Reproduce Fourier-MIONet paper (Jiang et al. 2024) | CO2 sequestration (3D) | MIONet, MIONet-FNN, Fourier-MIONet |
| [tno_co2](tno_co2/) | Reproduce TNO paper (Diab & Al Kobaisi 2025) | CO2 sequestration (3D) | TNO (autoregressive) |

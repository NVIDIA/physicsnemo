# Geometry Guardrails Example

This example demonstrates how to use PhysicsNeMo's geometry guardrails for
validating CAD/STL files against a distribution of geometries using
real-world automotive datasets.

## Overview

Geometry guardrails provide out-of-distribution (OOD) detection for 3D
geometric data. They learn the distribution of geometries from
training data and flag out-of-distribuution shapes at inference time.

This example uses two datasets:

- **DrivAerML**: 500 parametrically morphed variants of the DrivAer notchback vehicle
- **AhmedML**: 500 geometric variations of the Ahmed body

## Prerequisites

Install the required dependencies:

```bash
pip install pyvista
```

## Dataset Setup

### 1. Data Split Strategy

For DrivAerML, we use a train/test split:

- **Training**: 400 STL files (runs 1-400)
- **Test/Validation**: 100 STL files (runs 401-500)

The script will automatically organize these into separate directories.

### 2. Download Datasets from HuggingFace

**DrivAerML** (<https://huggingface.co/datasets/neashton/drivaerml>):

Download training files (400 files):

```bash
# Create directory for DrivAerML training data
mkdir -p data/drivaerml_train

# Download training STL files (runs 1-400)
BASE_URL="https://huggingface.co/datasets/neashton/drivaerml/resolve/main"
for i in $(seq 1 400); do
    wget "${BASE_URL}/run_$i/drivaer_$i.stl" \
        -O "data/drivaerml_train/drivaer_$i.stl"
done
```

Download test/validation files (100 files):

```bash
# Create directory for DrivAerML validation data
mkdir -p data/drivaerml_val

# Download validation STL files (runs 401-500)
BASE_URL="https://huggingface.co/datasets/neashton/drivaerml/resolve/main"
for i in $(seq 401 500); do
    wget "${BASE_URL}/run_$i/drivaer_$i.stl" \
        -O "data/drivaerml_val/drivaer_$i.stl"
done
```

**AhmedML** (<https://huggingface.co/datasets/neashton/ahmedml>):

Download all files for cross-dataset evaluation:

```bash
# Create directory for AhmedML data
mkdir -p data/ahmedml

# Download STL files (example for runs 1-500)
BASE_URL="https://huggingface.co/datasets/neashton/ahmedml/resolve/main"
for i in $(seq 1 500); do
    wget "${BASE_URL}/run_$i/ahmed_$i.stl" \
        -O "data/ahmedml/ahmed_$i.stl"
done
```

### 3. Directory Structure

After downloading, your directory structure should look like:

```text
examples/minimal/guardrails/
├── geometry_validation.py     # Main script
├── README.md                   # This file
└── data/                       # Dataset directory
    ├── drivaerml_train/        # DrivAerML training STL files (400 files)
    │   ├── drivaer_1.stl
    │   ├── drivaer_2.stl
    │   └── ... (up to drivaer_400.stl)
    ├── drivaerml_val/          # DrivAerML validation STL files (100 files)
    │   ├── drivaer_401.stl
    │   ├── drivaer_402.stl
    │   └── ... (up to drivaer_500.stl)
    └── ahmedml/                # AhmedML STL files (500 files)
        ├── ahmed_1.stl
        ├── ahmed_2.stl
        └── ... (up to ahmed_500.stl)
```

## Getting Started

### Run the Example

```bash
python geometry_validation.py
```

The script will:

1. Verify that datasets are properly downloaded and organized
2. Run three experiments on GPU:
   - **Experiment 1**: GMM (Gaussian Mixture Model) trained on DrivAerML train,
   tested on DrivAerML validation
   - **Experiment 2**: PCE (Polynomial Chaos Expansion) trained on DrivAerML train,
   tested on DrivAerML validation
   - **Experiment 3**: GMM trained on DrivAerML train, tested on AhmedML (cross-dataset)
3. Report results with OK/WARN/REJECT classifications for each experiment

### Interpret Results

**Output Example:**

```text
============================================================
Experiment 1: GMM - DrivAerML Train → DrivAerML Validation
============================================================
Results: 98 geometries validated
  OK:      97 (99.0%)
  WARN:      1 (1.0%)
  REJECT:    0 (0.0%)

============================================================
Experiment 2: PCE - DrivAerML Train → DrivAerML Validation
============================================================
Results: 98 geometries validated
  OK:      96 (98.0%)
  WARN:      2 (2.0%)
  REJECT:    0 (0.0%)

============================================================
Experiment 3: GMM - DrivAerML Train → AhmedML (Cross-Dataset)
============================================================
Results: 500 geometries validated
  OK:       0 (0.0%)
  WARN:    0 (0.0%)
  REJECT:  500 (100.0%)
```

- **OK**: Geometry is within the expected distribution (safe for inference)
- **WARN**: Geometry is unusual but may be acceptable (investigate)
- **REJECT**: Geometry is highly anomalous (likely invalid or OOD)

**Note on Validation Counts**:

In the example output above, Experiments 1 and 2 show 98 geometries validated
instead of the expected 100. This is because 2 STL files in the DrivAerML
validation set are corrupted and are automatically skipped during mesh validation.

**Expected Behavior**:

- Experiments 1 & 2 (same dataset): Most geometries should be OK since they're
  from the same distribution
- Experiment 3 (cross-dataset): Most geometries should be WARN/REJECT since
  AhmedML is a different vehicle shape

## Troubleshooting

### Issue: "No valid STL files found"

*Solution:* Verify STL files are downloaded correctly and paths are correct.

### Issue: Too many false positives

*Solution:* Lower thresholds (`warn_pct`, `reject_pct`).

## Support

For questions or issues, file an issue on the PhysicsNeMo GitHub repository.

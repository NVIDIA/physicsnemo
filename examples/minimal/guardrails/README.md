# Geometry Guardrails Example

This recipe composes optional geometry descriptors with the generic
`DensityScorer`, then applies an application-owned OK/WARN/REJECT policy.
It uses automotive STL datasets; mesh loading, directory processing, thresholds,
and persistence metadata are visible in `geometry_validation.py`.

## Overview

The scorer fits GMM or PCE density models to reference feature vectors. Larger
scores indicate greater novelty. The geometry descriptors preserve absolute
position, orientation, and size, so use consistent coordinate units and
preprocessing across reference and query meshes.

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

Organize the files into the directories below; the script verifies them.

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

### Composition and saved state

The central operations are:

```python
from physicsnemo.experimental.uq.ood import DensityScorer

names, reference = directory_features(train_dir, n_workers=4)  # Recipe helper
scorer = DensityScorer(method="gmm", gmm_components=1, device="cuda")
scorer.fit(reference)
names, query = directory_features(test_dir, n_workers=4)
percentiles = scorer.percentiles(scorer.score(query))
statuses = [
    "REJECT" if p >= 99.9 else "WARN" if p >= 99.0 else "OK" for p in percentiles
]
```

The directory helper uses `physicsnemo.mesh.io.from_pyvista` and
`physicsnemo.experimental.uq.ood.geometry.extract_features`. Invalid meshes are
reported and skipped by this recipe; an entirely invalid directory raises an
error. Descriptor extraction stays on CPU; the generic scorer uses its configured
device. `n_workers=1` runs extraction in the calling process.

Each fitted model writes an NPZ file for array state and a companion JSON file
for scoring configuration, descriptor identity, units, preprocessing, and policy.
The example reloads these before scoring. Loading rejects a mismatched feature
identity. Set `length_unit` to the actual source coordinate unit; it records the
recipe's choice and does not infer or convert units. These application artifacts
are distinct from the removed `GeometryGuardrail.save()` wrapper format.

### Interpret results

- **OK**: Score below the configured warning threshold.
- **WARN**: Score between the warning and rejection thresholds.
- **REJECT**: Score at or above the rejection threshold.

These labels express the recipe's policy. They do not certify prediction
accuracy or mesh validity. Choose thresholds using appropriate held-out data;
raising a percentile threshold makes a decision less sensitive, reducing flags.
The example reports the actual counts from the supplied files rather than
assuming a fixed number of valid geometries.

For a self-contained example without downloaded datasets, run
`examples/minimal/uq/geometry_scoring.py` from the repository root.

## Support

For questions or issues, file an issue on the PhysicsNeMo GitHub repository.

Geometry schema v1.1 corrects PCA column labels to match the existing interleaved
values. Descriptor values and their order are unchanged. The version and name
hash change, so v1.0 recipe metadata fails the compatibility check. After verifying
that an artifact used the original extractor, update its feature version, names
and hash together; fitted density parameters do not need reordering or refitting
for this label correction.

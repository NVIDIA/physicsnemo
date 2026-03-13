# Darcy GeoTransolver multi-dataset

Darcy flow solver example using the GeoTransolver model (when 2D support is available) with a **multi-dataset** pipeline: one Numpy (`.npz`) source and one HDF5 (`.h5`) source composed via `physicsnemo.datapipes.MultiDataset`.

## Config layout

- **conf/config.yaml** — Main config; composes dataloader, model, and training. Override `data.numpy_path` and `data.hdf5_path`.
- **conf/dataloader/** — DataLoader and MultiDataset.
- **conf/model/geotransolver.yaml** — GeoTransolver model (functional_dim=1, out_dim=1, geometry_dim=2).
- **conf/training/default.yaml** — Epochs, optimizer, scheduler, loss.

## Data

Both sources must expose the same TensorDict keys after transforms (for `output_strict=True`). Typical Darcy schema: `permeability`, `darcy` (2D fields per sample).

- **Numpy**: path to a single `.npz` (samples along first dimension) or a directory of `.npz` files (`file_pattern: "*.npz"`).
- **HDF5**: path to a single `.h5` (samples along first dimension) or a directory of `.h5` files (`file_pattern: "*.h5"`).

(Support for MATLAB-derived data for the numpy side may be added later under the same dataloader/numpy layout.)

## Run visualization

From this directory:

```bash
python load_and_visualize_data.py \
  data.numpy_path=/path/to/numpy_data \
  data.hdf5_path=/path/to/hdf5_data
```

Figures are written to the Hydra output directory (e.g. `outputs/.../`).

## Benchmark each dataset

To time load throughput for the numpy and hdf5 sources separately:

```bash
python benchmark_datasets.py \
  data.numpy_path=/path/to/numpy_data \
  data.hdf5_path=/path/to/hdf5_data
```

Optional: `bench_n_samples=100` limits the number of samples per pass (for quick runs on large datasets).

## Geometry for GeoTransolver

On a **regular 2D grid** we use **normalized (x, y) coordinates** as the geometry input: each grid point gets a 2D position in `[0, 1]`. The training script builds a single grid of shape `(1, H*W, 2)` and expands it to `(B, H*W, 2)` per batch, with `geometry_dim=2` in the model config. This gives GeoTransolver a consistent notion of spatial location without requiring an unstructured mesh.

## Training

From this directory:

```bash
python train.py data.numpy_path=/path/to/npz data.hdf5_path=/path/to/h5
```

Optional overrides: `training.max_epochs=50`, `training.batch_size=8`, `training.optimizer.lr=5e-4`. Checkpoints are saved under `./checkpoints/`.

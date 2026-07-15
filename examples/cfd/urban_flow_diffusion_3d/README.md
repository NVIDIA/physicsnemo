<!-- markdownlint-disable -->
# 3D Diffusion Prior for Unconditional Sampling in a Simplified Turbulent Urban Environment

## Problem overview

This example trains a 3D denoising diffusion model as a generative prior over
turbulent flow fields (streamwise/wall-normal/spanwise velocity volumes) in a
simplified urban environment (flow past a single obstacle), then draws
unconditional samples from it, using
[`physicsnemo.experimental.models.diffusion_unets.DiffusionUNet3D`](../../../physicsnemo/experimental/models/diffusion_unets/diffusion_unet_3d.py)
as the denoising backbone, EDM preconditioning
(`physicsnemo.diffusion.preconditioners.EDMPreconditioner`), and the
`physicsnemo.diffusion` training/sampling utilities (noise schedulers,
`MSEDSMLoss`, `sample`).

The denoising backbone architecture is based on
[Diff-SPORT: Diffusion-based Sensor Placement Optimization and Reconstruction
of Turbulent flows in urban environments](https://arxiv.org/abs/2506.00214).
This example implements only the unconditional generative-prior training and
sampling stage of that pipeline -- not Diff-SPORT's sensor-placement
optimization or its conditional/posterior reconstruction from sparse
observations.

## Getting started

### Prerequisites

Install the example's Python dependencies (this includes `huggingface_hub`,
which provides the CLI used below to download the dataset):

```bash
pip install -r requirements.txt
```

This example targets the **full-resolution** configuration: `288x88x88`
volumes (`D,H,W`), 3 channels (U, V, W), 4-level U-Net (`channel_mult:
[1,2,2,2]`, `model_channels: 64`).

### Download the dataset

The dataset used to develop and smoke-test this example is available at
[`abvish/UrbanFlow-oneObstacle-NoTrip`](https://huggingface.co/datasets/abvish/UrbanFlow-oneObstacle-NoTrip)
on Hugging Face (3D and 2D velocity fields from a Nek5000 spectral-element
CFD simulation of urban flow past a single obstacle, CC-BY-4.0, public). This
example uses the **3D** velocity-fluctuation fields under the repo's `3d/`
folder.

**A note on size.** The 3D shards are large: roughly **335 GB each** (one shard
holds 6,250 snapshots at `288x88x88`), and all currently-published 3D shards
together are about **1.3 TB**. You do not need the whole set: a single shard is
enough to train and smoke-test. Make sure you have the disk space before
downloading.

Download with the Hugging Face CLI (`hf`, provided by the `huggingface_hub`
package installed in the prerequisites above). The dataset is public, so no
login or token is required:

```bash
# Option A: a single 3D shard (~335 GB), downloaded into ./data
hf download abvish/UrbanFlow-oneObstacle-NoTrip \
  3d/combined-3d-ds1-fluctuations-snap-00000-06249.h5 \
  --repo-type dataset --local-dir ./data

# Option B: all currently-published 3D shards (~1.3 TB), into ./data
hf download abvish/UrbanFlow-oneObstacle-NoTrip \
  --include "3d/*.h5" --repo-type dataset --local-dir ./data
```

With `--local-dir ./data`, files keep their repo-relative path, so Option A
lands at `./data/3d/combined-3d-ds1-fluctuations-snap-00000-06249.h5`. (The
legacy `huggingface-cli download ...` command accepts the same arguments if
that is the version you already have installed.)

Each 3D shard is an HDF5 file with separate `U`/`V`/`W` velocity-fluctuation
datasets of shape `(N, 288, 88, 88)`, plus `x`/`y`/`z` grid coordinates and `t`
timestamps. `UflowDataset3D` auto-detects this layout; it also accepts a single
combined `data` dataset of shape `(N, 3, 288, 88, 88)` (the raw per-component
layout is what the shards ship with, the combined layout is what an
`-optimized` repacking pass produces). Point the config at the file you
downloaded:

```bash
python train.py paths.dataset=./data/3d/combined-3d-ds1-fluctuations-snap-00000-06249.h5
```

Generate unconditional samples from a checkpoint:

```bash
python generate.py paths.dataset=/path/to/your/data.h5 generate.io.inf_ckpt=<epoch>
```

### Configuration basics

Configuration is managed through [Hydra](https://hydra.cc/docs/intro/), with
config groups under `conf/`: `dataset/`, `model/`, `train/`, `generate/`,
`evaluate/`, `visualize/`. Override any value on the command line, e.g.
`python train.py train.hp.epochs=100 model.model_args.model_channels=64`.

## Smoke test

Trained for real against the full dataset (10,000 snapshots, no
subsampling) on a single L40S GPU, fp32, `batch_size_per_gpu=1`:

Loss decreases across both completed epochs, confirming the migrated
pipeline (`DiffusionUNet3D` + `EDMPreconditioner` + `EDMNoiseScheduler` +
`MSEDSMLoss`) trains correctly end-to-end on real data. 


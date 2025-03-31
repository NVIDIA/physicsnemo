# Unified Recipe for Training Global Weather Forecasting Models

This example demonstrates how to train a neural global weather forecast model.
The recipe is set up so that modifying the model architecture, data, or the
training procedure is straightforward.

## Configs

The `conf` directory contains the configuration files for the model, data,
training, etc. The configs are given in YAML format and use the `omegaconf`
library to manage them. Several example configs are given for generating
different datasets, models, and training procedures. For example, AFNO and
GraphCast are given with corresponding training procedure and datasets configs.
The default configs are set to only download and train a tiny dataset and can be
run on an 8GB GPU. To train larger models please adjust `conf/config.yaml`
according to the comments.

## Getting the ERA5 dataset

In this example we provide scripts to obtain the ERA5 dataset from [ARCO
ERA5](https://github.com/google-research/arco-era5) and perform needed curation
and remapping steps. ARCO ERA5 contains a complete lat lon gridded dataset of
the ERA5 reanalysis including single and pressure level data. Often when
training a model on ERA5, a temporal and channel subset is used. For example,
FourCast Net is trained on a 20-channel subset of ERA5 at 6 hour temporal
resolution [(AFNO)](https://openreview.net/pdf?id=EXHG-A3jlM). There can also be
the need for remapping from lat lon grids as is the case with the [DLWP
model](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002502).
Given these requirements we provide the following workflow for generating needed
datasets that works for most applications.

### Download temporally subsampled ERA5 dataset from ARCO ERA5

We recommend first downloading a temporally and single leveled subset of ERA5
from ARCO ERA5. This can be done using the `download_era5.py` script and configs
for this can be found in `./conf/dataset/`. This script will require ~40TB of
storage for non-tiny configs but can be adjusted to download a smaller subset of
the data. Given a 2.5 Gb/s connection the download will take ~1.5 days. The
default configs will only download ~100 GBs.

```python download_era5.py```

### Generate Curated Dataset for Training

Once the ERA5 dataset is downloaded you can generate a curated dataset for
training. This can be done using the `curate_era5.py` script and configs for
this can be found in `./conf/curated_dataset/`. This script will generate the
zarr dataset needed for training including needed transformations such as
regridding.

```python curate_era5.py```

### NOTE

In theory one should be able perform curation directly from ARCO ERA5. This will
work however there is a significant penalty in doing so due to the pressure
levels being chunked together in ARCO ERA5. This means that if you want to
extract a single pressure level you will need to download all 37 levels. If you
are planning to test multiple transforms or channel subsamplings then this will
become prohibitively expensive. Because of this we recommend following our
described workflow. We have also raised an issue on ARCO ERA5 to fix this
chunking [issue](https://github.com/google-research/arco-era5/issues/69) and if
resolved we will update instructions.

## Training the model

### Prerequisites

Install the required dependencies by running below:

```bash
pip install -r requirements.txt
```

Apart from the dataset configs the main configs for training are `model`,
`training`, and `validation`. These can be adjusted accordingly and to train the
model, run

```python train.py```

Progress can be monitored using MLFlow. Open a new terminal and navigate to the
training directory, then run:

```mlflow ui -p 2458```

View progress in a browser at <http://127.0.0.1:2458>

Data parallelism is also supported with multi-GPU runs. To launch a multi-GPU
training, run

```mpirun -np <num_GPUs> python train.py```

If running inside a docker container, you may need to include the
`--allow-run-as-root` in the multi-GPU run command.

## SFNO Training

One of the showcased models available in the configs is [Spherical Fourier Neural Operators:
Learning Stable Dynamics on the Sphere](https://arxiv.org/pdf/2306.03838.pdf). In order to
train the SFNO model, [PhysicsNeMo Makani](https://github.com/NVIDIA/modulus-makani)
needs to be installed. This allows the model to be added to physicsnemo's model registry.
For more information on this process, please refer to [PhysicsNeMo model registry](
https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.models.html#physicsnemo-model-registry-and-entry-points).

```bash
git clone git@github.com:NVIDIA/makani.git
cd makani
pip install -e .
```

The config file can be modified to train the SFNO model by uncommenting all SFNO configs.
Following the prior dataset fetching and curation steps, the model can be trained by running:

```python train.py```

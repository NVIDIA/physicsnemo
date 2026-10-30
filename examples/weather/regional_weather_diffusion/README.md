<!-- markdownlint-disable -->
# Regional Weather Diffusion: High-Resolution Weather Models Using Generative Diffusion

## Problem Overview

The resolution of global weather and climate models is typically too coarse to resolve localized weather phenomena, such as severe thunderstorms and
mesoscale convective systems, at sufficient detail. Therefore, high-resolution regional models are commonly used to model the area of interest in more detail. Modeling the atmosphere at this scale is both challenging and expensive with traditional numerical weather prediction. Simple loss functions also often limit the ability of AI-based atmospheric models to reproduce extreme events and characterize uncertainty. High-resolution diffusion models offer a solid foundation for realistic and probabilistic AI-based modeling of local weather.

This recipe provides a toolkit for training and simple inference for regional AI high-resolution models
using diffusion. Examples of the types of models that can be trained with it include:

- [CorrDiff](https://arxiv.org/html/2309.15214),
a _downscaling_ model that generates local high-resolution weather samples probabilistically from a lower-resolution input at the same time. The regression+diffusion setup used in the [original CorrDiff](https://nvidia.github.io/earth2studio/main/modules/generated/models/dx/CorrDiffTaiwan/) is supported, while purely diffusion-based downscaling can also be trained. This recipe is now better supported and recommended over the [original CorrDiff recipe](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/corrdiff).

- [StormCast](https://arxiv.org/abs/2408.10958),
a generative diffusion model designed to emulate NOAA’s High-Resolution Rapid Refresh (HRRR) model, a 3 km
operational convection-allowing model. StormCast autoregressively predicts multiple atmospheric state variables.
The [original StormCast](https://nvidia.github.io/earth2studio/main/modules/generated/models/px/StormCast/) uses a regression+diffusion U-Net configuration similar to CorrDiff, while the [full-CONUS
version](https://nvidia.github.io/earth2studio/main/modules/generated/models/px/StormCastCONUS/) uses a diffusion transformer without a regression stage.

<p align="center">
<img src="../../../docs/img/stormcast_rollout.gif"/>
</p>

- [StormScope](https://arxiv.org/abs/2601.17268),
a _nowcasting/nearcasting_ model for clouds and precipitation, using satellite radiances and radar measurements as training data. StormScope's generative model architecture allows the uncertainty of the predicted variables to be accurately quantified.

All of the above models can be formulated in terms of a high-resolution state update guided by a conditioning. In the code, we refer to the high-resolution training target in generic terms as `state` and to everything the model conditions on as `background`, including any past state history a forecasting model needs as input. See [Model Inputs and Outputs](#model-inputs-and-outputs) and [Adding Custom Datasets](datasets/CUSTOM_DATASET.md) for details.

## Getting Started

### Preliminaries

1. Install PhysicsNeMo (if not already installed) with the `datapipes-extras` and `nn-extras` optional dependency groups, along with the packages in `requirements.txt`. 
2. Copy this folder (`examples/weather/regional_weather_diffusion`) to a system with a GPU available.

To install the requirements not covered by PhysicsNeMo with the above-mentioned optional dependency groups, you can run:

```bash
pip install -r requirements.txt
```

`natten` (used by the DiT architecture) is built from source unless a pre-built wheel matching your PyTorch and CUDA versions is available. To install a pre-built wheel, use the `natten-cu12` or `natten-cu13` PhysicsNeMo extra, or see the [NATTEN installation instructions](https://natten.org/install/).

### Quick Start

To check that everything works before preparing your own data, generate a small example dataset filled with random data and run a short training on it:

```bash
python -c "from datasets.xarray import write_example_files; write_example_files('data', image_shape=(128, 128))"
python train.py --config-name lite
```

The first command writes data files in the format read by the default `XarrayDataset` to `data/`. The second trains the default `reference_downscaling` model (a diffusion transformer) for 20 steps with a batch size of 2, using the `lite` config. The outputs, including checkpoints and validation plots, are saved in `rundir/stormcast-training/0`. Rerunning the same command continues from the latest checkpoint in that directory; delete it to start from scratch (see [Checkpoints and Resuming](#checkpoints-and-resuming)).

To train a model on your own data:

1. Convert your data to the format described in [datasets/XARRAY_DATASET.md](datasets/XARRAY_DATASET.md), or [implement a custom dataset](datasets/CUSTOM_DATASET.md).
2. Copy the `reference_*` config closest to your problem (see [Configuration Basics](#configuration-basics)), e.g. `cp config/reference_downscaling.yaml config/my_model.yaml`, and edit its `dataset` section to point to your data and select your variables.
3. Test the config by setting it in `config/lite.yaml` in place of `reference_downscaling` and running `python train.py --config-name lite`.
4. Launch the training with `python train.py --config-name my_model`, or with `torchrun` to use multiple GPUs (see [Distributed Training](#distributed-training)).

### Testing

To test the functionality of the training code, you can run the `pytest` test suite in `test_training.py` in a single-GPU or multi-GPU environment. 

To run the single-GPU tests, run:

```bash
pytest test_training.py
```

If no tests fail, that indicates that the training environment is functional. It is normal for many tests to be skipped because not all parameter combinations are supported and some tests are meant to be run with multiple GPUs. Multi-GPU tests need to be run using `torchrun`, with tests of 1, 2, and 4 GPUs included. For instance, to run all 2-GPU tests:

```bash
torchrun --standalone --nproc_per_node=2 --no-python pytest test_training.py
```

The multi-GPU tests might produce some clutter because all processes write to the terminal, but the output should be readable.

The tests take a while to run (around 15 min on an H100 for the single-GPU tests). Potentially useful `pytest` options include `-k "test_training"` (run only the tests from the `test_training` function) and `-x` (stop at first failure).

### Configuration Basics

Training is handled by `train.py`, configured using [Hydra](https://hydra.cc/docs/intro/) based on the contents of the `config` directory and validated using `pydantic`. Hydra allows for YAML-based modular and hierarchical configuration management and supports command-line overrides for quick testing and experimentation. The `config` directory includes the following subdirectories:

 - `dataset`: specifies the dataset used for training and the parameters of the dataset, such as the location of the data files
 - `model`: specifies the model type and model-specific hyperparameters
 - `sampler`: specifies hyperparameters used in the sampling process for diffusion models
 - `training`: specifies training-specific hyperparameters and settings like checkpoint/log frequency and where to save training outputs
 - `inference`: specifies settings for `inference.py`, such as which model checkpoints and dataset samples to use
 - `hydra`: specifies basic Hydra settings, like where to store outputs (based on the training outputs directory)

Also in the `config` directory are several top-level configs:

 - `reference_*`: recommended starting points for new projects:
   - `reference_downscaling`: diffusion-only downscaling with a DiT (the default used by `train.py`)
   - `reference_corrdiff`: CorrDiff-style regression+diffusion downscaling with a U-Net
   - `reference_stormcast`: StormCast-style forecasting guided by low-resolution conditioning, with a DiT
   - `reference_stormscope`: StormScope-style nowcasting with a DiT
 - `replication_*`: configs for reproducing the training of published models
 - `lite`: a minimal config with a small batch size and frequent printouts and checkpoints, for quickly testing that training runs without errors
 - `test_*`: configs for testing training with synthetic data, without needing any data files

You can select any of these by specifying it as a config name at the command line using `--config-name`. Optionally, you can also override any specific items of interest using the command line args. For example:

```bash
python train.py --config-name reference_stormcast training.batch_size=4
```

More extensive configuration modifications can be made by creating a new top-level configuration file. The `reference_*` configs show how to specify a top-level config that uses default configuration settings with additional custom modifications added on top.

At runtime, Hydra parses the config subdirectory and command-line overrides into a runtime configuration object `cfg`, which has all settings accessible through both attribute and dictionary-like interfaces. For example, the total training batch size can be accessed either as `cfg.training.batch_size` or `cfg['training']['batch_size']`. The actual content of the config is validated using `pydantic` based on the guidelines specified in [`utils/config.py`](./utils/config.py).

The training script `train.py` initializes the training experiment and launches the main training loop, which is defined in `utils/trainer.py`. Outputs (training logs, checkpoints) are saved to a directory specified by the following `training` config items:

```yaml
training.outdir: 'rundir' # Root path under which to save training outputs
training.experiment_name: 'stormcast-training' # Name for the training experiment
training.run_id: '0' # Unique ID to use for this training run
training.rundir: ./${training.outdir}/${training.experiment_name}/${training.run_id} # Path where experiment outputs will be saved
```
The `training.run_id` setting can be used for distinguishing between different runs of the same configuration. The final training output directory is constructed by composing together the `training.outdir` root path (defaults to `rundir`), the `training.experiment_name`, and the `training.run_id`. By default, the outputs of [`inference.py`](#inference) are saved in the `inference` subdirectory of this directory.

## Training Models

This recipe can be used to train different configurations of diffusion models: nowcasting and downscaling models, hybrid nowcasting/downscaling models that use both a past state and a forcing condition, as well as unconditional diffusion models. The type of model is not set explicitly; instead, it depends on how the dataset provides input (conditioning) and target data.

### Model Inputs and Outputs

The input and output data are contained in the dict returned by the `__getitem__` function of the dataset. The expected model output (i.e. the training target) is always contained in the `"state"` key of the dict.

The types of input the model uses are configured in the `model.diffusion_conditions` (if training a diffusion model) or `model.regression_conditions` (if training a regression model) configuration settings. These are lists where the valid items are `"background"`, `"invariant"` and (for `model.diffusion_conditions` only) `"regression"`.

`"background"` is the condition tensor returned by your dataset and contains all variables that the model is conditioned on, typically past state and low-resolution conditioning, depending on your model type. It must be returned in the `"background"` key of the dict returned by the `__getitem__` function of your dataset. For unconditional diffusion models, `"background"` may be omitted from the `model.*_conditions` list and from the `__getitem__` dict. The following table shows typical settings for each type of model:

| Model type | Example model | `"background"` contains | `"state"` contains |
| --- | --- | --- | --- |
| Hybrid | StormCast | `cat(past_state, low_res_conditioning)` | `future_state` |
| Nowcasting | StormScope | `past_state` | `future_state` |
| Downscaling | CorrDiff | `low_res_conditioning` | `high_res_state` |
| Unconditional | | Nothing (omit `"background"` key) | `state_tensor` |

If you want to use invariants (that is, conditions that are the same for each sample; a surface elevation map is a typical example), return the invariants as a NumPy array from the `get_invariants` function of your dataset and add `"invariant"` to your condition list.

If training a regression-diffusion model like CorrDiff, add `"regression"` to `model.diffusion_conditions`. The training loop will then automatically run the regression model supplied in `model.regression_weights`. Conversely, `model.regression_conditions` is not used and setting it is not necessary if training a pure diffusion model.

DiT-type models additionally support scalar conditions that are single numbers that apply to the entire grid. To add these, return the conditions as a 1D NumPy array in the `"scalar_conditions"` key of the dictionary returned by `__getitem__`. Additionally, implement the `scalar_condition_channels` method for the dataset.

### Model Architecture

#### Diffusion-Only vs Regression+Diffusion

- **Diffusion-only** (`reference_downscaling`, `reference_stormcast`, `reference_stormscope`): a single diffusion model generates `"state"` directly. This is simpler to train and run and has worked well in StormCast-CONUS and StormScope, but may require tuning the noise distribution to your dataset.
- **Regression+diffusion** (`reference_corrdiff`, as in CorrDiff and the original StormCast): a regression model trained with a mean squared error loss predicts the expected `"state"`, and a diffusion model generates the residual on top of it. This can make the diffusion task easier, at the cost of training two models and running both at inference. Train the [regression model](#training-regression-models) first, then the [diffusion model](#training-diffusion-models).

We recommend starting with diffusion-only training for most new projects.

#### U-Net vs DiT

The architecture is selected with `model.architecture`:

- **`dit`** (recommended for new projects): a diffusion transformer with [NATTEN](https://natten.org/) neighborhood attention. Supports scalar conditions and [mask tokens](datasets/CUSTOM_DATASET.md#spatial-mask), but currently only diffusion model training.
- **`unet`**: a SongUNet-family U-Net, as used in CorrDiff and the original StormCast. Required for regression models and lead time labels.

`model.hyperparameters` is passed to the [`DiT`](../../../physicsnemo/models/dit/dit.py) or [SongUNet](../../../physicsnemo/models/diffusion_unets/) class via `utils/nn.py`; see their docstrings for all options. The main ones are:

- DiT: `hidden_size`, `depth`, `num_heads`, `patch_size` and `attn_kernel_size` (neighborhood size in tokens). The grid dimensions must be divisible by `patch_size`.
- U-Net: `model_channels`, `channel_mult` (per-level channel multipliers; its length sets the depth), `num_blocks` and `attn_resolutions`. The grid dimensions must be divisible by `2**(len(channel_mult) - 1)`.

See `config/model/reference_*.yaml` for starting points and `config/model/replication_*.yaml` for the settings of published models.

### Training Regression Models

You can ignore this section if you plan to train a diffusion-only model such as StormScope.

Regression models are trained by setting `training.loss.type` to `regression`. For example, to train the regression model of the original StormCast, you can use the `replication_stormcast_v1` config and an optional name for the training experiment:

```bash
python train.py --config-name replication_stormcast_v1 training.loss.type=regression training.experiment_name=regression
```

### Training Diffusion Models

Diffusion model training is launched by setting `training.loss.type` to `edm` (the default).

To train a diffusion model that uses the regression model output (unnecessary for diffusion-only models), there are two config items that must be defined:
  1. `'regression'` included in `model.diffusion_conditions`.
  2. `model.regression_weights` set to the path of a PhysicsNeMo (`.mdlus`) checkpoint with model weights for the pre-trained regression model. These are saved in the checkpoint directory during training (see [Checkpoints and Resuming](#checkpoints-and-resuming)).
     
The `reference_corrdiff` and `replication_stormcast_v1` top-level configs show examples of how to specify these settings.

With that, launching diffusion training for the original StormCast configuration looks something like:

```bash
python train.py --config-name replication_stormcast_v1 training.experiment_name=diffusion
```

### Testing Training

To run a quick end-to-end test of training, copy the `config/lite.yaml` config file and swap `reference_downscaling` for your training config file. Then launch training using your modified `lite.yaml` config file. It applies overrides that make the training run for a few steps and go through all essential steps (initialization, training, validation and checkpoint saving). This verifies that training runs without obvious problems, though it is not expected to provide a useful model and will likely not reveal problems that only appear after longer training, such as training instabilities.

### Checkpoints and Resuming

Checkpoints are saved every `training.checkpoint_freq` steps and at the end of training, in `{training.rundir}/checkpoints_regression` or `{training.rundir}/checkpoints_diffusion`. Each consists of two files:

- `<ModelClass>.0.<step>.mdlus` (e.g. `StormCastUNet.0.16000.mdlus` or `EDMPreconditioner.0.16000.mdlus`): the model weights and architecture, loadable with `physicsnemo.Module.from_checkpoint(path)`. Use the best one (typically the one with the lowest validation loss) for inference, `model.regression_weights` or `training.initial_weights`.
- `checkpoint.0.<step>.pt`: the optimizer and scheduler states, needed to resume training but not for inference.

Training automatically resumes from the latest checkpoint in `training.rundir`, which makes it easy to continue interrupted runs (`training.max_run_steps` can be used to stop a job after a given number of steps). To start a new run instead, change `training.experiment_name` or `training.run_id`, or delete the old run directory. Set `training.resume_checkpoint` to a step number to resume from a specific checkpoint.

To initialize a new run from an earlier model, e.g. for fine-tuning, set `training.initial_weights` to its `.mdlus` file. This loads only the weights and is ignored when resuming.

### Distributed Training

Both regression and diffusion training can be distributed easily with data parallelism through `torchrun` or other launchers (for example, SLURM `srun`). As long as GPU memory is sufficient, the same configuration file can be used regardless of the number of GPUs. Each training batch is distributed evenly between the GPUs; you just need to ensure the configuration being run has a batch size that is divisible by the number of available GPUs/processes. For example, distributed training over 8 GPUs on one node would look something like:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --config-name <your_training_config>
```

#### Domain Parallelism

As an advanced form of distributed training, the training code also supports domain parallelism through PhysicsNeMo `ShardTensor`. It distributes individual training samples across multiple GPUs, and therefore can be used to train large models and/or large domains when even a single sample will not fit in the memory of one GPU. Domain parallelism is controlled by the `training.domain_parallel_size` config setting that specifies how many GPUs each sample should be distributed to. For example, if you have at least two GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --config-name <your_training_config> training.domain_parallel_size=2
```

will train using a domain parallel size of 2 (use `test_diffusion` as `your_training_config` to run a quick test of domain parallel training). When domain parallelism is used, the local batch size should be 1, so the global batch size will be smaller than the number of GPUs: `batch_size * domain_parallel_size == number_of_gpus`.

Domain parallelism introduces some additional communication overhead and therefore should only be used if even a local batch size of 1 cannot fit on a single GPU.

### Memory Management

The default configuration uses a batch size of 64 (controlled by `training.batch_size`), as used in the StormCast paper. Many datasets can be trained with a smaller batch size; for instance, StormCast-CONUS and StormScope use a batch size of 16, as set in the `reference_stormcast`, `reference_stormscope` and corresponding `replication_*` config files. If your data grid is large, you may be able to train with a smaller batch size since the larger grid already provides more training data per batch.

If you have few GPUs and/or GPUs with limited memory, the default setting of `training.batch_size_per_gpu: 'auto'` may cause you to run out of memory. In that case, you can reduce the per-GPU memory utilization by manually setting `training.batch_size_per_gpu` to `training.batch_size // (number_of_gpus * n)` where `n` is an integer. The training code will automatically employ gradient accumulation to maintain the desired effective batch size specified by `training.batch_size` while using less memory, at the cost of longer training time.

Another way to reduce memory usage is to enable 16-bit training by setting `training.perf.fp_optimizations` to `amp-bf16`. While the default is `fp32`, our experience indicates that BF16 training works as well as 32-bit training and therefore it is set by the reference configuration files. To train DiT models in BF16, it is necessary to set `model.hyperparameters.attn_kwargs.qk_norm=true`, likewise already enabled in the reference configs.

If even a single sample is too big to fit on one GPU, consider using [domain parallelism](#domain-parallelism).

### Sigma-Bin Loss Tracking

For diffusion models, you can enable per-sigma-bin loss logging to diagnose which noise levels the model struggles with. This logs the mean training loss for equal-probability bins of the sampled sigma values to TensorBoard and/or WandB (under `loss/train_sigma_bin/`).

To enable it, add the following to your diffusion training config:

```yaml
training:
  loss:
    track_sigma_bin_loss: true
    sigma_bin_count: 8  # number of equal-probability bins (default: 8)
    # sigma_bin_edges: []  # optional: explicit strictly-increasing bin edges
```

When `sigma_bin_edges` is not provided, bins are computed automatically so that each bin has equal probability mass under the configured sigma distribution (`lognormal` or `loguniform`). This feature is ignored for regression models.

## Datasets

The dataset used by the training process is set by the `dataset.name` configuration option, in the form `module.DatasetClass` where the training will use the class `DatasetClass` found in `module.py` within the `datasets` directory. For example, `dataset.name=xarray.XarrayDataset` will use `XarrayDataset` from `datasets/xarray.py`. You can either use one of the provided datasets or implement your own.

### Included Datasets

- `xarray.XarrayDataset`: a flexible Xarray-based dataset that can be configured to train many types of models. For more complex use cases or for loading data from custom data formats, it can be used as a starting point and modified or subclassed. See [datasets/XARRAY_DATASET.md](datasets/XARRAY_DATASET.md).
- `mock.MockDataset`: supplies random data in different configurations; useful for testing training loops.
- `data_loader_hrrr_era5.HrrrEra5Dataset`: legacy dataloader for StormCast; reads HRRR and ERA5 data from zarr-formatted files. See [datasets/HRRR_ERA5_DATASET.md](datasets/HRRR_ERA5_DATASET.md).

### Adding Custom Datasets

When `xarray.XarrayDataset` is not sufficient, you can define a custom dataset class. See [datasets/CUSTOM_DATASET.md](datasets/CUSTOM_DATASET.md) for the interface it must implement, how to configure it, and the optional spatial mask and loading strategies.

## Experiment Tracking

Every `training.print_progress_freq` steps, the trainer prints a few training statistics such as the training loss.

This recipe also supports both TensorBoard and Weights & Biases (WandB), which can be enabled by setting `training.log_to_tensorboard=True` and `training.log_to_wandb=True`, respectively. TensorBoard is free to use; WandB requires a license, but academic accounts are free to create at [wandb.ai](https://wandb.ai/). After you set up an account, you can adjust `entity` and `project` in `utils/logging.py` to the appropriate names for your WandB workspace. TensorBoard logs are written to `{training.rundir}/tensorboard` while WandB logs are in `{training.rundir}/wandb`.

Validation runs every `training.validation_freq` steps and evaluates `training.validation_steps` batches (16 by default) of the validation set, using the same samples and noise every time so that the validation loss is comparable between steps. Increase `training.validation_steps` if these samples don't represent the validation set well. Images of generated samples and power spectra are also saved in the logs and in `{training.rundir}/images` for variables listed in `training.validation_plot_variables`.

Diffusion losses typically drop quickly at first and then slowly. Their late improvements are small even while sample quality still improves. The validation images and spectra are therefore usually more informative than the loss alone: a a ratio of generated to target spectral power close to 1 means that the generated fields have realistic small-scale detail. If the loss increases or spikes, see [Training Is Unstable](#training-is-unstable--not-converging).

## Inference

`inference.py` generates samples from a trained diffusion model with any dataset, using the training config to build the validation dataset and the model inputs. The inference options are added to the training config with `+inference=default` (see [`config/inference/default.yaml`](config/inference/default.yaml)). For example:

```bash
python inference.py --config-name reference_downscaling +inference=default \
    inference.diffusion_checkpoint=<checkpoint_dir>/EDMPreconditioner.0.<step>.mdlus \
    inference.indices=[0,10,20] inference.ensemble_size=4 inference.plot_variables=[t2m_t+0]
```

This generates an ensemble of 4 samples for each of the validation dataset samples 0, 10 and 20, and saves them along with the corresponding targets, in physical units, to `samples.zarr` in `inference.output_dir` (by default `{training.rundir}/inference`). It also saves a plot of `t2m` for each sample, comparing the first ensemble member to the target. For regression+diffusion models, the regression model is loaded from `model.regression_weights` unless `inference.regression_checkpoint` is given, and its prediction is also saved if `inference.save_regression=true`. Other parts of the config can be overridden as in training, e.g. `sampler.args.num_steps=32`.

**This is not autoregressive rollout.** `inference.py` does not feed its predictions back into the model as input for a following step; each sample uses its own input from the dataset. (Prior versions of this recipe carried predictions forward through the two-element `state` convention; that convention has been removed -- see [Adding Custom Datasets](datasets/CUSTOM_DATASET.md).) For autoregressive rollout or more complex inference workflows, `inference.py` can be used as a starting point, or you can bring your checkpoints to [Earth2Studio](https://github.com/NVIDIA/earth2studio) for further analysis and visualizations. The [Earth2Studio wrapper for StormCast](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/stormcast.py) can be used as a starting point for custom implementations.

## Troubleshooting / Frequently Asked Questions

### I'm Out of GPU Memory

Possible solutions include decreasing the batch size, training with more GPUs and using domain parallelism. See the section [Memory Management](#memory-management).

### How Do I Train a Model Without Low-Resolution Conditioning (For Example, a Nowcasting Model)?

Supply the past state from the dataset in `"background"` and the future state in `"state"`. Refer to [Model Inputs and Outputs](#model-inputs-and-outputs).

### How Do I Train a Pure Downscaling Model Without State Update?

Supply the low-resolution conditioning from the dataset in `"background"` and the high-resolution target in `"state"`. Refer to [Model Inputs and Outputs](#model-inputs-and-outputs).

### How Do I Train an Unconditional Diffusion Model?

Ensure that the `model.diffusion_conditions` and `model.regression_conditions` settings for your model do not include `"background"`. Your dataset may omit the `"background"` key entirely from the dict returned by `__getitem__`; if it provides one anyway, it will be ignored. See [Model Inputs and Outputs](#model-inputs-and-outputs).

### Training Is Slow

If training seems slow, check your GPU utilization. If it's low, the problem is likely a bottleneck with the data loading. Another way to identify a data bottleneck is to test your model using `MockDataset` (set `dataset.name=mock.MockDataset`); if training with `MockDataset` is much faster than with real data, it likely indicates a data loading bottleneck. Many datasets used in regional high-resolution models contain lots of data. Depending on the dataset size and format, consider the following optimizations:
  * Increasing `training.num_data_workers` (or `dataset.loader.num_workers`, which overrides it)
  * If your dataset implements one, switching to the PhysicsNeMo `datapipes` loading strategy (`dataset.loader.backend: "datapipes"`) -- see [Loading Strategy: `torch` vs `datapipes`](datasets/CUSTOM_DATASET.md#loading-strategy-torch-vs-datapipes)
  * Parallel loading of data in multiple threads within the dataset; controlled by `dataset.num_threads` if using the reference `XarrayDataset`
  * Optimized data preprocessing using e.g. [Numba](https://numba.pydata.org/)
  * Data compression, if bandwidth from storage is the limiting factor
  * Caching data, particularly if your dataset is small enough to be preloaded to memory (see the `cache_location` option of `XarrayDataset`)

Enabling BF16 training (`training.perf.fp_optimizations=amp-bf16`) speeds up training considerably; if using a DiT model, `model.hyperparameters.attn_kwargs.qk_norm=true` should be set to ensure stability.

Compiling the training loss with ``torch.compile`` (`training.perf.torch_compile=True`), which will also compile the model forward pass, also typically improves performance. This recipe disables it under domain parallelism, as compilation is not yet fully supported by PhysicsNeMo ShardTensor.

### Training Is Unstable / Not Converging

Consider the following solutions:
  * Ensure that your data is normalized to near zero mean and unit variance.
  * Skewed and heavy-tailed data distributions may introduce instability. Consider applying a nonlinear transformation in preprocessing, or in the dataset (see [Normalization Statistics](datasets/XARRAY_DATASET.md#normalization-statistics) for `XarrayDataset`), to bring the variable distribution closer to a normal distribution. For instance, some variables become better behaved when log-transformed. Ensure that the distribution of the transformed variable is near zero mean and unit variance.
  * The default lognormal sampling of noise levels may introduce instability in some cases. Consider setting `training.loss.sigma_distribution` to `loguniform`, and `training.loss.sigma_min` and `training.loss.sigma_max` to the ends of the noise level range (for example, 0.01 and 200, respectively).
  * If using DiT, we recommend setting `model.hyperparameters.attn_kwargs.qk_norm=true`; without this option, DiT has proven to be unstable in the current implementation. It is already set in the reference configurations.
  * If you have many state variables, the diffusion model may not have the capacity to denoise them all. Increasing the model width (`model.hyperparameters.hidden_size` for DiT, `model.hyperparameters.model_channels` for U-Net) may help, at the cost of increased compute and memory requirements.

## References

- [Kilometer-Scale Convection Allowing Model Emulation using Generative Diffusion Modeling](https://arxiv.org/abs/2408.10958)
- [Elucidating the design space of diffusion-based generative models](https://openreview.net/pdf?id=k7FuTOWMOc7)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Learning Accurate Storm-Scale Evolution from Observations](https://arxiv.org/abs/2601.17268)

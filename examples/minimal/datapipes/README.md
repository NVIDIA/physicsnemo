# PhysicsNeMo DataPipes

Dataloading is critical to SciML applications, both for training and inference,
and the physicsnemo datapipe infrastructure aims to deliver a flexible and configurable
set of tools to enable your application.

> [!NOTE]
> What is a **datapipe**?  We consider a datapipe the process that loads data from
> persistent storage, performs any online preprocessing (augmentation, normalization,
> etc) that can't be done in advance, and prepares the loaded data to be ingested
> into the model and training or inference loop.

There are plenty of tools in the `Python` ecosystem for loading, preprocessing, and
preparing your data for training or inference.  To compare / contrast some of these
tools with the ecosystem available, and see if the `PhysicsNeMo` datapipe interface
might be valuable to your workload, consider the following design principles
we followed when building the `PhysicsNeMo` datapipes:

1. **GPU-first** - Many scientific datasets are *large* for even a single example:
the data is high resolution and the preprocessing needs benefit from GPU acceleration.
Compare this to other methods where the data preprocessing is predominantly CPU-based,
such as the `PyTorch` DataLoader: whereas CPU-based preprocessing may introduce GPU
pipeline stalls on high resolution data, GPU-based preprocessing will maximize
throughput.

2. **Threading over Multiprocessing** - In `Python`, true concurrency is difficult
due to the global interpreter lock (GIL), and is typically only achieved by
spawning subprocesses (via multiprocessing) or when offloading
to compiled libraries or GPU kernels.
For this reason, many data loaders leverage multiprocessing for data concurrency:
load images in separate processes, and collate a batch on the main thread.
For simplicity, with a GPU-first paradigm, the `PhysicsNeMo` datapipe focuses on GPU
concurrency via asynchronous execution and stream-based parallelism.  IO is coordinated
in multiple threads, instead of multiple processes, and streams enable multiple
preprocessing pipelines to execute concurrently on the GPU.

3. **Unambiguous Configuration and Serialization** - Datapipes can be a particularly
frustrating component in reproducibility of AI results - the preprocessing, sampling,
batching and other parameters can be hard to infer from training scripts.  Here,
we make a deliberate design choice to enable datapipe configuration serialization
as a first-class citizen.  `PhysicsNeMo` Datapipes can be built directly in `Python`,
but also instantiated from `hydra` YAML files for version control and distribution.

4. **Familiar Interfaces** - We built our tools from scratch, but they are meant
to look familiar and inter-operate with the tools you already know.  Use
`PhysicsNeMo` DataLoaders as a replacement for `PyTorch`'s DataLoader; tools like
`DistributedSampler` will still work. Users of `torchvision` will be familiar
with the concept of chaining transformations together.

5. **Extensibility out of the box** - We want to provide a data pipeline that gives
great performance and usability immediately - but it will never be the case that
one codebase covers all possible data needs out of the box.  Therefore, the
`PhysicsNeMo` datapipe is extensible: you can build custom data readers for
new dataformats, and plug them in to datasets; you can build new transforms
for your data that we might not have, and simply plug them into a transformation
pipeline. You can even package all of this up as a pip-installable extension: Using
the built in registry enables you to still instantiate and version control datapipes,
when the components are not even part of PhysicsNeMo.

## When should I use `PhysicsNeMo` datapipes over X/Y/Z data utility?

In general, the `PhysicsNeMo` datapipe utility is built to deliver good performance
on data that is large, per example, like most scientific data is.  If you want a
batch size of 512 small images, it may be more performant to use a CPU-centric
tool.

Another advantage of the PhysicsNeMo datapipe is the ability to build datapipes
directly from configuration files, allowing serializable and version-controlled
data configuration.  This isn't the only tool that can do this, of course.

## Core Datapipe Design

Think of datasets as a hierarchy of data: at the highest level, an entire **dataset**
consists of independent **examples**.  Each example has one or more **tensor components**:
image data may have input images and target labels; CFD data may have positions,
target pressures, a mesh object, boundary conditions, etc.; weather data may contain
sensor readings as a function of time.  Each example may be the same size as the others,
or each example may be a unique size.  Even the components of an example can be variable,
though this can require extra care in reading and using the dataset.

The `PhysicsNeMo` datapipe consists of the following components:

- `reader`s contain the logic to understand a **dataset** on disk, and
  load examples into CPU memory.  

- The `dataset`, which contains a `reader`, orchestrates threads that preload
  data **examples** from disk and move it to GPU.  On the GPU, a `dataset` can apply a
  series of transformations to each **example**.  Each example is stored in `tensordict`
  format.  The dataset will also track metadata, for understanding where each **example**
  came from (index, filepath, etc.).

- A `transform` is a callable class that accepts a tensordict as input, and returns
  a `tensordict` as output.  Chaining transformations together is the core way to
  manipulate data examples on the fly in a datapipe.

- The `dataloader` is a drop-in replacement for the `PyTorch` DataLoader, with additional
  optimizations for the GPU-centric processing here.  The `dataloader` handles
  stream concurrency, batch collation, and triggering preloading of datasets.

---

## Tutorials

This directory contains progressive tutorials that teach you how to use the
`PhysicsNeMo` datapipe infrastructure effectively.  Note that some of the tutorials
are repetitive and verbose, to highlight different features of the datapipe
ecosystem.  We'll give some overview of what you can learn in each tutorial,
but they are meant to be run interactively and explored.

### Data Prerequisites

You do not need to have any specific data in hand for the tutorials.  You can
generate synthetic data with the scripts `generate_regular_data.py` and
`generate_variable_points_data.py`.

### Tutorial 1: Getting Started with DataPipes

**File:** `tutorial_01_getting_started.py`

Learn the core concepts of data loading from disk:

- Creating a Reader to load data from files
- Understanding the `(TensorDict, metadata)` return format
- Wrapping a reader in a Dataset
- Iterating with a DataLoader
- Accessing batch data via TensorDict keys

```bash
# Generate tutorial data first
python generate_regular_data.py -n 100 \
-s "velocity:128,128,128,3 pressure:128,128,128,1 position:128,128,128,3" \
-b zarr -o output/tutorial_data/

# Run the tutorial
python tutorial_01_getting_started.py
```

### Tutorial 2: Transforms and Data Preprocessing

**File:** `tutorial_02_transforms.py`

Build preprocessing pipelines with transforms:

- Apply a single transform (Normalize)
- Compose multiple transforms together
- Subsample point clouds with SubsamplePoints
- Use geometric transforms (Translate, Scale)
- Save/load normalization statistics from files
- Denormalize data with the `inverse()` method

```bash
# Generate regular grid data (for most sections)
# Note: Tutorial 2 can reuse the data from Tutorial 1
python generate_regular_data.py -n 100 \
-s "velocity:128,128,128,3 pressure:128,128,128,1 position:128,128,128,3" \
-b zarr -o output/tutorial_data/

# Generate point cloud data (for subsampling sections)
python generate_variable_points_data.py -n 100 \
-s "coords:3 features:8" --min-points 50000 \
--max-points 100000 -b zarr -o output/pointcloud_data/

# Run the tutorial
python tutorial_02_transforms.py
```

### Tutorial 3: Custom Collation for GNNs

**File:** `tutorial_03_custom_gnn_datapipe.py`

Build a GNN-ready data pipeline with custom collation:

- Build a custom Transform for computing KNN graph edges
- Implement a custom Collator for PyG-style graph batching
- Understand how PyG batches graphs (offset edges, concatenate features, batch tensor)
- Put it all together in a complete GNN training pipeline

```bash
# Generate point cloud data with coordinates and features (can be reused from tutorial 2)
python generate_variable_points_data.py -n 100 \
-s "coords:3 features:8" --min-points 50000 \
--max-points 100000 -b zarr -o output/pointcloud_data/

# Run the tutorial
python tutorial_03_custom_gnn_datapipe.py
```

### Tutorial 4: Hydra Configuration for DataPipes

**File:** `tutorial_04_hydra_config.py`

Build entire datapipes from YAML configuration with minimal Python code:

- Define reader, transforms, dataset, and dataloader in YAML
- Use `hydra.utils.instantiate()` to build components
- Override any parameter from the command line
- Switch between configurations easily

```bash
# Generate tutorial data (from tutorials 2 and 3)
python generate_variable_points_data.py -n 100 -s \
"coords:3 features:8" --min-points 50000 \
--max-points 100000 -b zarr -o output/pointcloud_data/

# Run with default config
python tutorial_04_hydra_config.py

# Override from command line
python tutorial_04_hydra_config.py dataloader.batch_size=8 dataloader.dataset.device=cuda

# Use point cloud configuration (this is the default)
python tutorial_04_hydra_config.py --config-name tutorial_04_pointcloud

# Override transform parameters
python tutorial_04_hydra_config.py --config-name tutorial_04_pointcloud \
    subsample.n_points=5000
```

PhysicsNeMo Datapipes
======================

.. automodule:: physicsnemo.datapipes
.. currentmodule:: physicsnemo.datapipes

The PhysicsNeMo Datapipes consists largely of two separate components.

Prior to version 2.0 of PhysicsNeMo, each datapipe was largely
independent from all others, targeted for very specific datasets and applications,
and broadly not extensible. Those datapipes, preserved in v2.0 for compatibility,
are described in the climate, cae, gnn, and
benchmark subsections.

In PhysicsNeMo v2.0, the datapipes API has been redesigned from scratch to focus
on key factors to enable scientific machine learning training and inference.
This document describes the architecture and design philosophy.

Refer to the PhysicsNeMo examples for runnable datapipe tutorials to
get started.


Datapipes Philosophy
--------------------

The PhysicsNeMo datapipe structure is built on several key design decisions
that are specifically made to enable diverse scientific machine learning datasets:

- GPU First: data preprocessing runs on the GPU, not the CPU.
- Isolation of roles: reading data is separate from transforming data, which is 
  separate from pipelining data for training, which is separate from threading
  and stream management. Changing data sources, or preprocessing pipelines,
  should require no intervention in other areas.
- Composability and Extensibility: Use the toolkit and examples to build what you
  need if a component is not included.
- Datapipes as configuration: Changing a pipeline should not require source code
  modification. The registry system in PhysicsNeMo datapipes enables Hydra instantiation
  of datapipes at runtime for version-controlled, runtime-configured datapipes.
  You can also register and instantiate custom components.

Data flows through a PhysicsNeMo datapipe in a consistent path:

1. A ``reader`` will bring the data from storage to CPU memory.
2. An optional series of one or more transformations will apply on-the-fly
   manipulations of that data, per instance of data.
3. Several instances of data will be collated into a batch (customizable,
   like in PyTorch).
4. The batched data is ready for use in a model.

At the highest level, ``physicsnemo.datapipes.DataLoader`` has a similar API and
model as ``pytorch.utils.data.DataLoader``, which enables a drop-in replacement for many
cases. However, PhysicsNeMo has a very different computation orchestration.

Quick Start
-----------

.. code-block:: python

    from physicsnemo.datapipes import (
        Dataset,
        DataLoader,
        HDF5Reader,
        Normalize,
        SubsamplePoints,
    )

    # 1. Choose a Reader for your data format
    reader = HDF5Reader(
        "simulation_data.h5",
        fields=["pressure", "velocity", "coordinates"],
    )

    # 2. Define a transform pipeline
    transforms = [
        Normalize(
            input_keys=["pressure"],
            method="mean_std",
            means={"pressure": 101325.0},
            stds={"pressure": 5000.0},
        ),
        SubsamplePoints(
            input_keys=["coordinates", "pressure", "velocity"],
            n_points=2048,
        ),
    ]

    # 3. Create a Dataset (Reader + Transforms + device transfer)
    dataset = Dataset(reader, transforms=transforms, device="cuda")

    # 4. Wrap in a DataLoader for batched iteration
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in loader:
        predictions = model(batch["pressure"], batch["coordinates"])


Refer to the examples in the
`examples directory <https://github.com/NVIDIA/physicsnemo/tree/main/examples/minimal/datapipes>`_ to explore PhysicsNeMo datapipes and get started.


Architecture
------------

The diagram below gives a high-level overview of how the ``physicsnemo``
datapipe tools interact.

.. code-block:: text

    ┌──────────────┐       ┌──────────────────┐       ┌──────────────────────┐
    │    Reader    │──────▶│     Dataset      │──────▶│     DataLoader       │
    │              │       │                  │       │                      │
    │  _load_sample│       │  device transfer │       │  batches indices     │
    │  __len__     │       │  + transforms    │       │  + collation         │
    │              │       │  through Compose │       │  + stream prefetch   │
    │  Returns:    │       │                  │       │                      │
    │  (TensorDict,│       │  Returns:        │       │  Yields:             │
    │   metadata)  │       │  (TensorDict,    │       │  batched TensorDict  │
    │  on CPU      │       │   metadata)      │       │                      │
    └──────────────┘       └──────────────────┘       └──────────────────────┘
                                    │
                           ┌────────┴────────┐
                           │   Transforms    │
                           │                 │
                           │  Compose chains │
                           │  multiple into  │
                           │  a pipeline     │
                           └─────────────────┘

Core API
--------

DataLoader
^^^^^^^^^^

The ``DataLoader`` serves as a nearly drop-in
replacement for the PyTorch DataLoader. A notable difference is the movement
of ``pin_memory`` from the ``DataLoader`` class to the ``Reader`` classes.
This is because of the much earlier GPU data transfer in the PhysicsNeMo
datapipe compared to PyTorch.

The ``DataLoader`` drives one of two mutually-exclusive paths, selected by
dataset type:

- **Map-style preload path** (:class:`~physicsnemo.datapipes.DatasetBase`,
  for example ``Dataset``, ``MeshDataset``): a dedicated dispatcher thread keeps a
  *bounded* number of samples in flight by pulling the index stream
  *lazily* under backpressure and submitting host-only loads to a worker
  pool. The main thread consumes the resulting samples in order
  (host-to-device transfer plus transforms on a preprocessing stream) and
  reassembles batches from boundary markers, so the full epoch is never
  materialized up front and irregular batch sizes are supported.
- **Iterable generator path** (:class:`~physicsnemo.datapipes.IterableDatasetBase`):
  a generator dataset driven entirely on the main thread (no sampler, no
  worker pool). Refer to `Iterable Datasets`_ below.

In both paths, *all device-kernel launches happen on a single main
thread*. This is the real constraint for Warp-based transforms. Warp may
launch on any CUDA stream when the launch comes from the main thread
and Warp's current stream matches the active PyTorch stream.
Preprocessing runs on a separate preprocessing stream. A CUDA event orders
that stream against the compute stream so preprocessing overlaps training
without blocking the host.

.. autoclass:: physicsnemo.datapipes.dataloader.DataLoader
    :members:
    :show-inheritance:

Dataset
^^^^^^^

The ``Dataset`` is the core IO + Transformation coordinator of the datapipe 
infrastructure. Whereas the ``DataLoader`` will orchestrate the pipeline,
the ``Dataset`` is responsible for the threaded execution of ``Reader``s and
``Transform`` pipelines to execute it.

.. autoclass:: physicsnemo.datapipes.dataset.Dataset
    :members:
    :show-inheritance:

MultiDataset
^^^^^^^^^^^^

The ``MultiDataset`` combines two or more ``Dataset`` instances into a single
index space through concatenation. Each sub-dataset can have its own ``Reader``
and transforms. ``MultiDataset`` maps global indices to the owning sub-dataset
and local index, and adds ``dataset_index`` to sample metadata so batches can
identify the source.
Use ``MultiDataset`` when you train on multiple datasets with the same
``DataLoader``. You can optionally enforce matching ``TensorDict`` keys across
all outputs for collation. Refer to
:const:`physicsnemo.datapipes.multi_dataset.DATASET_INDEX_METADATA_KEY`
for the metadata key added to each sample.

To collate and stack outputs from different datasets, set
``output_strict=True`` when you construct a ``MultiDataset``. After
construction, ``MultiDataset`` loads the first batch from each sub-dataset
and verifies that the ``TensorDict`` from the ``Reader`` and ``Transform``
pipeline has consistent keys. Collation details differ by dataset, so
``MultiDataset`` validates output key consistency only.

.. autoclass:: physicsnemo.datapipes.multi_dataset.MultiDataset
    :members:
    :show-inheritance:


Iterable Datasets
^^^^^^^^^^^^^^^^^

Map-style datasets (``Dataset``, ``MeshDataset``) assume a fixed length and a
sampler that yields indices. Some workloads lack both a fixed length and a
sampler. These include online simulations, procedural generators, and sources
that produce samples during iteration without a meaningful ``__len__``. For those cases, subclass
:class:`~physicsnemo.datapipes.IterableDatasetBase` and yield samples from
``__iter__``. The ``DataLoader`` detects iterable datasets automatically and
switches to the main-thread-only generator path. It ignores ``shuffle`` and
``sampler`` and issues a warning. ``len(loader)`` raises because the length is
unknown.

An iterable dataset chooses one of two emission modes through the
``yields_batches`` attribute:

- ``yields_batches = False`` (default): ``__iter__`` yields individual
  samples and the ``DataLoader`` collates them into batches of
  ``batch_size`` (honoring ``drop_last``).
- ``yields_batches = True``: ``__iter__`` yields fully-formed batches and the
  ``DataLoader`` passes them through without further collation, which is the
  natural fit for a generator that already produces a batch per step.

Reproducibility seeds from ``(epoch, position)`` rather than the map-style
``(epoch, index)`` pair. Implement ``set_epoch`` and/or ``set_generator`` to
seed deterministically from the iteration position.
Because the generator runs on the main thread, Warp kernels inside it are
safe on any preprocessing stream the ``DataLoader`` binds. Refer to the online
simulation tutorial in the
`examples directory <https://github.com/NVIDIA/physicsnemo/tree/main/examples/minimal/datapipes>`_
for a runnable Warp ``Darcy2D`` generator wired through this path.

.. autoclass:: physicsnemo.datapipes.IterableDatasetBase
    :members:
    :show-inheritance:


Readers
^^^^^^^

Readers are the data-ingestion layer. Each one loads individual samples from a
specific storage format (HDF5, Zarr, NumPy, VTK) and returns CPU tensors
in a uniform dict interface. Refer to :doc:`physicsnemo.datapipes.readers` for the
base class API and all built-in readers.

Transforms
^^^^^^^^^^

Transforms are composable, device-agnostic operations applied to each sample
after it is loaded and transferred to the target device. The ``Compose``
container chains multiple transforms into a single callable. Refer to
:doc:`physicsnemo.datapipes.transforms` for the base class API, ``Compose``,
and all built-in transforms.

Collation
^^^^^^^^^

Combining a set of TensorDict objects into a batch of data can, at times, 
require special care. For example, collating graph datasets for Graph Neural 
Networks requires different merging of batches than concatenation along a batch
dimension. For this reason, PhysicsNeMo datapipes offer custom collation functions
as well as an interface to write your own collator. If the dataset you are
trying to collate cannot be accommodated here, open an issue on GitHub.

For an example of a custom collation function that produces a batch of PyG graph data,
refer to the examples on GitHub for the datapipes.

.. autoclass:: physicsnemo.datapipes.collate.Collator
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.collate.DefaultCollator
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.collate.ConcatCollator
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.collate.FunctionCollator
    :members:
    :show-inheritance:

Extending the Pipeline
----------------------

Custom Reader example:

.. code-block:: python

    import torch
    from physicsnemo.datapipes import Reader

    class CSVReader(Reader):
        def __init__(self, path, **kwargs):
            super().__init__(**kwargs)
            import pandas as pd
            self.df = pd.read_csv(path)

        def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
            row = self.df.iloc[index]
            return {
                "features": torch.tensor(row[:-1].values, dtype=torch.float32),
                "target": torch.tensor([row.iloc[-1]], dtype=torch.float32),
            }

        def __len__(self) -> int:
            return len(self.df)

Custom Transform example:

.. code-block:: python

    from tensordict import TensorDict
    from physicsnemo.datapipes import Transform

    class LogScale(Transform):
        def __init__(self, keys: list[str], epsilon: float = 1e-8):
            super().__init__()
            self.keys = keys
            self.epsilon = epsilon

        def __call__(self, data: TensorDict) -> TensorDict:
            for key in self.keys:
                data[key] = torch.log(data[key] + self.epsilon)
            return data

.. toctree::
   :maxdepth: 1
   :caption: Built-in Readers and Transforms

   physicsnemo.datapipes.readers
   physicsnemo.datapipes.transforms

Legacy Datapipes
----------------

The following datapipe modules predate the v2.0 redesign and are preserved for
backward compatibility. They are domain-specific, self-contained pipelines
originally written for particular datasets and workflows.

.. toctree::
   :maxdepth: 1
   :caption: Legacy Datapipes

   physicsnemo.datapipes.benchmarks
   physicsnemo.datapipes.climate
   physicsnemo.datapipes.gnn
   physicsnemo.datapipes.cae

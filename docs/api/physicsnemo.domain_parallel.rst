
PhysicsNeMo ``domain_parallel``
================================

In scientific AI applications, the parallelization techniques to enable state of the art
models are different from those used in training large language models.  PhysicsNeMo
introduces a parallelization primitive called a ``ShardTensor`` that is designed for
large-input AI applications to enable domain parallelization.

``ShardTensor`` provides a distributed tensor implementation that supports uneven sharding across devices.
It is a subclass of ``torch.Tensor`` that interoperates with PyTorch's DTensor and plain tensors,
while adding flexibility for cases where different ranks may have different local tensor sizes.

A key feature of ``ShardTensor`` is automatic promotion. When a plain ``nn.Module`` weight meets a
sharded activation, ``ShardTensor`` promotes the weight to a replicated distributed tensor and
reduces its gradient over the domain mesh in the backward pass. Standard ``nn.Module`` models
``nn.Module`` models work unmodified on sharded inputs - ``distribute_module`` is no longer needed
therefore work unmodified on sharded inputs. ``distribute_module`` is no longer needed or
recommended. For an additional data parallel axis, use Distributed Data Parallel (DDP) when
parameters are plain tensors, or Fully Sharded Data Parallel 2 (FSDP2)
(``torch.distributed.fsdp.fully_shard``) when parameters are sharded. ``torch.compile``
is supported, with the caveat that sequence-sharded ring attention must stay outside compiled regions.

.. note::

    **PhysicsNeMo 2.3 changes the ``ShardTensor`` input contract.** Prior to version 2.3, 
    ShardTensor expected all inputs to be either ``ShardTensor`` or ``DTensor``.
    Those inputs are still supported, but plain ``torch.Tensor`` inputs enables
    ``DDP`` and ``FSDP2`` support now as well.  We recommend this newer way of
    using ``ShardTensor``.

``ShardTensor``
---------------

.. autoclass:: physicsnemo.domain_parallel.ShardTensor
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.domain_parallel.ShardTensorSpec
    :members:
    :show-inheritance:

For debugging purposes, you can modify the promotion behavior of ``ShardTensor``.

.. autoclass:: physicsnemo.domain_parallel.TensorPromotionMode
    :members:
    :show-inheritance:

Utility Functions
-----------------

.. autofunction:: physicsnemo.domain_parallel.scatter_tensor

.. autofunction:: physicsnemo.domain_parallel.sync_module_over_mesh

Synchronization Responsibilities
--------------------------------

In a training script, ``DDP`` synchronizes weights across its entire process
group. With a 2D mesh, that process group does not include the domain axis of
the mesh.

Call ``sync_module_over_mesh`` after creating the model and before converting
its weights to distributed tensors.  It copies the plain parameters and
buffers from one process to the others in the mesh, so every model copy starts
with the same values.

This function does not synchronize gradients.  ``ShardTensor`` automatically
sums the gradient contributions produced by domain-parallel operations.

``sync_module_over_mesh`` skips ``DTensor`` and ``ShardTensor`` values.
``scatter_tensor`` and ``distribute_tensor`` create consistent distributed
tensors from one source tensor.  In contrast, ``from_local`` uses the value
already present on each process.  When using ``from_local``, you are
responsible for providing the correct local piece on every process.

The startup synchronization happens only once.  It does not keep buffers in
sync if they change during training.  Synchronize changing buffers separately
when your model requires it.  Checkpoint loading is also separate. Use the
distributed checkpoint utilities for distributed tensors, and synchronize
plain model state if only one process loaded it.

Example: FSDP2 with Domain Parallelism
--------------------------------------

Create a two-dimensional mesh.  The ``"ddp"`` dimension holds different
training samples and shards the model weights.  The ``"domain"`` dimension
splits one sample across multiple GPUs.  The product of the two dimensions
must equal the number of distributed processes.

For example, with eight GPUs, ``data_parallel_size=4`` and ``domain_size=2``
creates a :math:`(4, 2)` mesh:

.. code-block:: python

    import torch
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from physicsnemo.domain_parallel import sync_module_over_mesh

    data_parallel_size = 4
    domain_size = 2
    mesh = init_device_mesh(
        "cuda",
        (data_parallel_size, domain_size),
        mesh_dim_names=("ddp", "domain"),
    )
    data_mesh = mesh["ddp"]
    domain_mesh = mesh["domain"]

    model = MyModel().cuda()

    # All domain ranks must start with the same plain weights and buffers.
    sync_module_over_mesh(model, domain_mesh)

    # FSDP2 shards weights only across the data-parallel dimension.
    fully_shard(model, mesh=data_mesh)

Call ``sync_module_over_mesh`` before ``fully_shard``.  If the model has
parameters that must themselves be sharded over the domain, synchronize the
plain model first, then convert those selected parameters with
``distribute_tensor`` before calling ``fully_shard``.

Example: DDP with Domain Parallelism
------------------------------------

When the weights fit on each GPU, use DDP on the ``"ddp"`` dimension.  The
domain mesh is still used for ``ShardTensor`` inputs:

.. code-block:: python

    from torch.nn.parallel import DistributedDataParallel as DDP

    model = MyModel().cuda()
    sync_module_over_mesh(model, domain_mesh)
    model = DDP(
        model,
        device_ids=[torch.cuda.current_device()],
        process_group=data_mesh.get_group(),
    )

For detailed information on ``ShardTensor`` and domain parallelism, please refer to the :doc:`Domain Parallelism <../../user-guide/domain_parallelism_entry_point>` tutorial.


PhysicsNeMo ``ShardTensor``
===========================

In scientific AI applications, the parallelization techniques to enable state of the art 
models are different from those used in training large language models.  PhysicsNeMo 
introduces a new parallelization primitive called a ``ShardTensor`` that is designed for 
large-input AI applications to enable domain parallelization.

``ShardTensor`` provides a distributed tensor implementation that supports uneven sharding across devices. 
It builds on PyTorch's DTensor while adding flexibility for cases where different ranks may have 
different local tensor sizes.

The example below shows how to create and work with ``ShardTensor``:

.. code:: python

    import torch
    from torch.distributed.device_mesh import DeviceMesh 
    from torch.distributed.tensor.placement_types import Shard
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.distributed.shard_tensor import ShardTensor, scatter_tensor

    def main():
        # Initialize distributed environment
        DistributedManager.initialize()
        dm = DistributedManager()

        # Create a 1D device mesh - by default, a -1 will use all devices
        # (For a 2D mesh, -1 will work to infer a single dimension in a mesh tensor)
        mesh = dm.initialize_mesh((-1,), mesh_dim_names=["spatial"])

        # Create a tensor on rank 0
        if dist.rank == 0:
            tensor = torch.randn(100, 64)
        else:
            tensor = None

        # Scatter the tensor across devices with uneven sharding
        # This will automatically determine appropriate local sizes
        sharded = scatter_tensor(
            tensor,
            global_src=0, 
            mesh=mesh,
            placements=(Shard(0),)  # Shard along first dimension
        )

        # Work with local portions
        local_tensor = sharded.to_local()
        
        # Redistribute to different sharding scheme
        new_sharded = sharded.redistribute(
            placements=(Shard(1),)  # Change to shard along second dimension
        )

How does this work?
""""""""""""""""""""

``ShardTensor`` extends PyTorch's ``DTensor`` to support uneven sharding where different ranks can have different 
local tensor sizes. It tracks shard size information and handles redistribution between different 
sharding schemes while maintaining gradient flow.

Key differences from ``DTensor`` include:
- Support for uneven sharding where ranks have different local sizes
- Tracking and propagation of shard size information
- Custom collective operations optimized for uneven sharding
- Flexible redistribution between different sharding schemes

Operations work by:

1. Converting inputs to local tensors

2. Performing operations locally 

3. Constructing new ``ShardTensor`` with appropriate sharding

4. Handling any needed communication between ranks

.. autosummary::
   :toctree: generated

``ShardTensor``
----------------

.. autoclass:: physicsnemo.distributed.shard_tensor.ShardTensor
    :members:
    :show-inheritance:

Utility Functions
------------------

.. autofunction:: physicsnemo.distributed.shard_tensor.scatter_tensor


Why do we need this?
""""""""""""""""""""

During deep learning training, memory usage can grow significantly when working with large input data, even if the model itself is relatively small. This is because many operations create intermediate tensors that temporarily consume memory.

For example, consider a 2D convolution operation on a high-resolution image. If we have a batch of 1024x1024 images, even a simple 3x3 convolution needs to save the entire input image in memory for computing the gradients in the backward pass.

For high resolution images, this can easily lead to out of memory errors as model depth grows, even if the number of parameters is small - this is a significant contrast from LLM model training, where the memory usage is dominated by the number of parameters and the corresponding optimizer states.  In software solutions like DeepSpeed and ZeRO, this is handled by partitioning the model across GPUs, but this is not a solution for large-input applications.

``ShardTensor`` helps address this by:
- Distributing the input data across multiple devices
- Performing operations on smaller local portions
- Coordinating the necessary communication between devices in the forward and backward passes

``ShardTensor`` is built as an extension of PyTorch's DTensor, and gains substantial functionality by leveraging the utilities already implemented in the PyTorch distributed package.  However, some operations on sharded input data are not trivial to implement correctly, nor relevant to the model sharding problem.  In PhysicsNeMo, we have implemented parallelized versions of several key operations, including (so far):

- Convolution (1D, 2D, 3D)
- Neighborhood Attention (2D)

These operations are implemented in the ``physicsnemo.distributed.shard_utils`` module, and are enabled by dynamically intercepting calls to (for example) ``torch.nn.functional.conv2d``.  When the function is called with ShardTensor inputs, the operation is automatically parallelized across the mesh associated with the input.  When the function is called with non-ShardTensor inputs, the operation is executed in a non-parallelized manner, exactly as expected.

To enable these operations, you must import ``patch_operations`` from ``physicsnemo.distributed.shard_utils``.  This will patch the relevant functions in the distributed package to support ``ShardTensor`` inputs.

We are continuing to add more operations, and contributions are welcome!




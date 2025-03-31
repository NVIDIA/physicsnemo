Domain Decomposition, ``ShardTensor`` and ``FSDP`` Tutorial
============================================================

This tutorial demonstrates how to use PhysicsNeMo's ``ShardTensor`` functionality alongside PyTorch's ``FSDP``   (Fully Sharded Data Parallel) to train a simple convolutional neural network. We'll show how to:

1. Create a simple CNN model
2. Set up input data sharding across multiple GPUs
3. Combine FSDP with domain decomposition
4. Train the model

Simple CNN Model
----------------

The preamble to the training script has an important patch to make sure that the conv2d operation works with ``ShardTensor``:

.. code-block:: python

    import torch

    # This is necessary to patch Conv2d to work with ShardTensor
    from physicsnemo.distributed.shard_utils import patch_operations

    import torch.nn as nn

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.distributed.shard_tensor import ShardTensor
    from torch.distributed.tensor import distribute_module, distribute_tensor
    from torch.distributed.tensor.placement_types import Shard, Replicate
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

Next, setup the distributed environment including the device mesh.  Here we do it globally, 
but you can do it locally as well and pass device_mesh objects around.

Setting Up the Environment
--------------------------

.. code-block:: python

    # Initialize distributed environment
    DistributedManager.initialize()
    dm = DistributedManager()

    # Create a 2D mesh for hybrid parallelism
    # First dimension for data parallel, second for spatial decomposition
    mesh = dm.initialize_mesh((-1, 2), mesh_dim_names=["data", "spatial"])

    # Get submeshes for different parallel strategies
    data_mesh = mesh["data"]      # For FSDP
    spatial_mesh = mesh["spatial"] # For spatial decomposition

First, let's create a simple one-layer CNN model:

.. code-block:: python

    import torch
    import torch.nn as nn
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.distributed.shard_tensor import ShardTensor
    from torch.distributed.tensor.placement_types import Shard
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, 10)
            
        def forward(self, x):
            # This is automatically parallel:
            x = self.conv(x)
            x = self.relu(x)
            # This operation reduces on the parallel dimension.
            # This will leave x as a Partial placement, meaning
            # it isn't really sharded anymore but the results on the domain
            # pieces haven't been computed yet.
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    

Preparing Data with ``ShardTensor``
------------------------------------

Create a simple dataset and shard it across devices:

.. code-block:: python

    def create_sample_data(batch_size=32, height=32, width=64):
        # Create random data
        data = torch.randn(batch_size, 3, height, width, device=f"cuda:{dm.device}")
        labels = torch.randint(0, 10, (batch_size,), device=f"cuda:{dm.device}")
        
        # Convert to ShardTensor for spatial decomposition
        placements = (Shard(2),)  # Shard H dimensions
        data = ShardTensor.from_local(
            data,
            device_mesh=spatial_mesh,
            placements=placements
        )

        # For the labels, we can leverage DTensor to distribute them:
        labels = ShardTensor.from_dtensor(
            distribute_tensor(labels,
                device_mesh=spatial_mesh,
                placements=(Replicate(),)
            )
        )
        
        return data, labels

Combining FSDP with Domain Decomposition
----------------------------------------

Set up the model with both FSDP and spatial decomposition:

.. code-block:: python

    def setup_model():
        # Create base model
        model = SimpleCNN().to(f"cuda:{dm.device}")
        
        # Take the module and distributed it over the spatial mesh
        # This will replicate the model over the spatial mesh
        # You can, if you want FSDP, get more fancy than this.
        model = distribute_module(
            model,
            device_mesh=spatial_mesh,
        )

        # Wrap with FSDP
        # Since the model is replicated, this will mimic DDP behavior.
        model = FSDP(
            model,
            device_mesh=data_mesh,
            use_orig_params=True
        )

        
        return model

Note that, above, we manually distribute the model over the spatial mesh, then setup FSDP over the data parallel mesh.


Training Loop
-------------

Implement a basic training loop:

.. code-block:: python

    def train_epoch(model, optimizer, criterion):
        model.train()
        
        for i in range(10):  # 10 training steps
            # Get sharded data
            inputs, targets = create_sample_data()
            
            # Forward pass
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if dm.rank == 0 and i % 2 == 0:
                print(f"Step {i}, Loss: {loss.item():.4f}")

Main Training Script
--------------------

Put it all together:

.. code-block:: python


    def main():



        # Create model and optimizer
        model = setup_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train for 5 epochs
        for epoch in range(5):
            if dm.rank == 0:
                print(f"Epoch {epoch+1}")
            train_epoch(model, optimizer, criterion)
            
        # Cleanup
        DistributedManager.cleanup()

    if __name__ == "__main__":
        main()


Running the Code
----------------

To run this example with 4 GPUs (2x2 mesh):

.. code-block:: bash

    torchrun --nproc_per_node=4 train_cnn.py

This will train the model using both data parallelism (``FSDP``) and spatial decomposition (``ShardTensor``) across 4 GPUs in a 2x2 configuration.

Key Points
----------

1. The device mesh is split into two dimensions: one for data parallelism (``FSDP``) and one for spatial decomposition (``ShardTensor``).  We get that in one line using torch DeviceMesh: ``mesh = dm.initialize_mesh((-1, 2), mesh_dim_names=["data", "spatial"])``.  And in fact, for multilevel parallelism, you can extend your mesh further.  Think of DeviceMesh like a tensor of arbitrary rank, and each element is one GPU.
2. Input data is sharded across the spatial dimension using ``ShardTensor``
3. ``FSDP`` handles parameter sharding and optimization across the data parallel dimension
4. The model can process larger spatial dimensions efficiently by distributing the computation

This example demonstrates basic usage - for production use cases, you'll want to add:

- Proper data loading and preprocessing
- Model checkpointing
- Validation loop
- Learning rate scheduling
- Error handling
- Logging and metrics

For more advanced usage and configuration options, refer to the PhysicsNeMo documentation on ``ShardTensor`` and the PyTorch FSDP documentation.

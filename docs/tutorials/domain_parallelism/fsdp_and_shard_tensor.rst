Domain Decomposition, ShardTensor and FSDP Tutorial
===================================================

In this tutorial, we will see how to combine domain parallelism, ``ShardTensor``, and a training or inference recipe.  Before diving too deeply into this tutorial, we recommend you read the other domain parallelism tutorial:

- :ref:`Domain Parallelism and Shard Tensor`
- :ref:`Implementing new layers for ShardTensor`


This tutorial demonstrates how to use PhysicsNeMo's ``ShardTensor`` functionality alongside PyTorch's ``FSDP``   (Fully Sharded Data Parallel) to train or evaluate a simple ViT.  Here's what's in the tutorial:

1. ViT Model Overview
2. Benchmarking the ViT on a single GPU
3. Enabling domain parallelism with ``ShardTensor``
4. Training and evaluating the model with domain parallelism

Simple ViT Model
----------------

The model we'll use for this tutorial is a really straightforward and simple ViT. It's very similar to the spirit of the original vision transformer from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" `Dosovitskiy et al. <https://arxiv.org/abs/2010.11929>`_. The model consists of two main conceptual pieces.  First, there is a convolutional tokenizer: it is a convolution with stride==kernel_size (so, non-overlapping image pieces) followed by a reshape to a sequence-like tensor with channels last.  Second, there is a transformer block with residual attention and a residual MLP.

The overall model architecture is straightforward, as well: the input image is tokenized using the convolutional tokenizer, a positional embedding is added, and then a series of transformer blocks are applied.  At the end of the transformer layers, all of the tokens are averaged together.  The entire architecture has one final layer to project the embedding dimension onto the output dimension.

.. note::

   This isn't really how you might implement a transformer for a vision classification task in practice - there are better, more sophisticated techniques.  Since the original ViT publication, technical advances such as `Convolution Transformers <https://arxiv.org/abs/2103.15808>`_, `Shifted Windows <https://arxiv.org/abs/2103.14030>`_, `Neighborhood Attention <https://arxiv.org/abs/2204.07143>`_, and others have outperformed vanilla ViTs like this for classification.  We encourage you to pick the model architecture most suitable for your task; for simplicitly to demonstrate the domain parallel techniques, we've pick a "Standard" vision transformer here.

Here's the core of the model:

.. dropdown:: Model Implementation
   :open:

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/model/ViT.py
      :caption: Example 1: ViT Model
      :language: python

If you want to get deeper into the components, you should expand the following sections to see the code:


.. dropdown:: Patch Embedding Implementations

   .. tab-set::

       .. tab-item:: 2D

           .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/model/PatchEmbed2D.py
              :caption: Convolutional Patch Embedding in 2D
              :language: python

       .. tab-item:: 3D

           .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/model/PatchEmbed3D.py
              :caption: Convolutional Patch Embedding in 3D
              :language: python

.. dropdown:: Transformer Block

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/model/TransformerBlock.py
      :caption: Transformer Block
      :language: python

.. dropdown:: MLP

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/model/MLP.py
      :caption: Simple Multi-layer Perceptron
      :language: python

.. dropdown:: Multi-head Attention

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/model/MultiHeadAttention.py
      :caption: Multi-head Attention with pytorch scaled dot product attention
      :language: python



Running the ViT
---------------

The training script for this tutorial is very simple: there is no data or labels, only synthetic data.  We loop over image sizes, initialize the ViT model, and then evaluate it's performance (computational performance, not model accuracy!) in a simple loop.  We measure both inference as well as training performance using ``torch.cuda.Event`` objects to capture timing information, and average over a few iterations.  For simplicity, we've package each of those pieces into simple functions to make it easier to run and reproduce this code:

.. dropdown:: Transformer Block

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/model/TransformerBlock.py
      :caption: Transformer Block
      :language: python

.. dropdown:: MLP

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/model/MLP.py
      :caption: Simple Multi-layer Perceptron
      :language: python

.. dropdown:: Multi-head Attention

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/model/MultiHeadAttention.py
      :caption: Multi-head Attention with pytorch scaled dot product attention
      :language: python





Next, setup the distributed environment including the device mesh.  Here we do it globally, 
but you can do it locally as well and pass device_mesh objects around.

Setting Up the Environment
--------------------------

.. tab-set::

    .. tab-item:: Single GPU

        .. code-block:: python

            import torch
            import torch.nn as nn

    .. tab-item:: DDP 

        .. code-block:: python

            import torch
            import torch.nn as nn

            from physicsnemo.distributed import DistributedManager

            # Add DDP import
            from torch.nn.parallel import DistributedDataParallel as DDP

    .. tab-item:: ShardTensor + FSDP

        .. code-block:: python

            import torch
            import torch.nn as nn

            # Imports for Domain Parallelism
            from physicsnemo.distributed import DistributedManager, scatter_tensor
            from torch.distributed.tensor import distribute_module, distribute_tensor

            # FSDP instead of DDP
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.tensor.placement_types import (  # noqa: E402
                Replicate,
                Shard,
            )


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

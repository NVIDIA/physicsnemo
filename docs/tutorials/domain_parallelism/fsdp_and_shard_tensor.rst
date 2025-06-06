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

The model we'll use for this tutorial is a straightforward and simple ViT. It's very similar to the spirit of the original vision transformer from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" `Dosovitskiy et al. <https://arxiv.org/abs/2010.11929>`_. The model consists of two main conceptual pieces.  First, there is a convolutional tokenizer: it is a convolution with stride==kernel_size (so, non-overlapping image pieces) followed by a reshape to a sequence-like tensor with channels last.  Second, there is a transformer block with residual attention and a residual MLP.

The overall model architecture is straightforward, as well: the input image is tokenized using the convolutional tokenizer, a positional embedding is added, and then a series of transformer blocks are applied.  At the end of the transformer layers, all of the tokens are averaged together.  The entire architecture has one final layer to project the embedding dimension onto the output dimension.

.. note::

   This isn't really how you might implement a transformer for a vision classification task in practice - there are better, more sophisticated techniques.  Since the original ViT publication, technical advances such as `Convolution Transformers <https://arxiv.org/abs/2103.15808>`_, `Shifted Windows <https://arxiv.org/abs/2103.14030>`_, `Neighborhood Attention <https://arxiv.org/abs/2204.07143>`_, and others have outperformed vanilla ViTs like this for classification.  We encourage you to pick the model architecture most suitable for your task; for simplicity to demonstrate the domain parallel techniques, we've picked a "Standard" vision transformer here.

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

The training script for this tutorial is very simple: there is no data or labels, only synthetic data.  We loop over image sizes, initialize the ViT model, and then evaluate its performance (computational performance, not model accuracy!) in a simple loop.  We measure both inference as well as training performance using ``torch.cuda.Event`` objects to capture timing information, and average over a few iterations.  For simplicity, we've package each of those pieces into simple functions to make it easier to run and reproduce this code:

.. dropdown:: How to measure model performance

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/utils/measure_perf.py
      :caption: After a few warmup steps, use ``torch.cuda.Event`` objects to capture timing information, and average over a few iterations.
      :language: python

.. dropdown:: Measuring memory usage

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/utils/measure_memory.py
      :caption: Use ``torch.cuda.reset_peak_memory_stats()`` and ``torch.cuda.max_memory_allocated()`` to measure memory usage.
      :language: python

.. dropdown:: End to End Benchmarking

   .. literalinclude:: ../../test_scripts/domain_parallelism/st_and_fsdp/utils/benchmark.py
      :caption: Combine the above two functions to measure performance and memory usage for a given model and inputs.
      :language: python

Up to here, we actually haven't written **any** code specific to domain parallelism.  That's deliberate - when using ``DDP``, users don't expect to have to modify model code significantly, and we've followed that same philosophy with ``ShardTensor``.  Below, we'll walk through the main script to highlight how components of the script change to enable domain parallelism.

Setting Up the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

We need an extra import for ``DDP``, and a few extra for ``ShardTensor`` and ``FSDP``:

.. tab-set::

    .. tab-item:: Single GPU

        .. code-block:: python

            import torch
            import torch.nn as nn

    .. tab-item:: DDP 

        .. code-block:: python

            import torch
            import torch.nn as nn

            # Use PhyscicsNeMo's distributed manager to simplify initialization
            from physicsnemo.distributed import DistributedManager

            # Add DDP import
            from torch.nn.parallel import DistributedDataParallel as DDP

    .. tab-item:: ShardTensor + FSDP

        .. code-block:: python

            import torch
            import torch.nn as nn

            # Use PhyscicsNeMo's distributed manager to simplify initialization
            from physicsnemo.distributed import DistributedManager
            
            # Upstream imports for FSDP:
            from torch.distributed.tensor import distribute_module, distribute_tensor

            # FSDP instead of DDP
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.tensor.placement_types import (  # noqa: E402
                Replicate,
                Shard,
            )

            # PhysicsNeMo imports to turn your inputs into ShardTensors
            from physicsnemo.distributed import scatter_tensor
            

Run Configuration
^^^^^^^^^^^^^^^^^

The configuration is the same for all three cases:

.. tab-set::

    .. tab-item:: Single GPU  

        .. code-block:: python

            args = parse_args()

            image_sizes = list(range(args.image_size_start, args.image_size_stop + 1, args.image_size_step))
            device = torch.device('cuda')
            
            # Generate image sizes based on start, stop, and step
            if args.dimension == 2:
                image_sizes = list(range(args.image_size_start, args.image_size_stop + 1, args.image_size_step))
            elif args.dimension == 3:
                image_sizes = list(range(args.image_size_start, min(args.image_size_stop + 1, 513), args.image_size_step))
            
            # Should we use mixed precision?
            precision_mode = "FP16" if args.use_mixed_precision and torch.cuda.is_available() else "FP32"
        
    .. tab-item:: DDP 

        .. code-block:: python

            args = parse_args()

            image_sizes = list(range(args.image_size_start, args.image_size_stop + 1, args.image_size_step))
            device = torch.device('cuda')
            
            # Generate image sizes based on start, stop, and step
            if args.dimension == 2:
                image_sizes = list(range(args.image_size_start, args.image_size_stop + 1, args.image_size_step))
            elif args.dimension == 3:
                image_sizes = list(range(args.image_size_start, min(args.image_size_stop + 1, 513), args.image_size_step))
        
            # Should we use mixed precision?
            precision_mode = "FP16" if args.use_mixed_precision and torch.cuda.is_available() else "FP32"

    .. tab-item:: ShardTensor + FSDP 

        .. code-block:: python

            args = parse_args()

            image_sizes = list(range(args.image_size_start, args.image_size_stop + 1, args.image_size_step))
            device = torch.device('cuda')
            
            # Generate image sizes based on start, stop, and step
            if args.dimension == 2:
                image_sizes = list(range(args.image_size_start, args.image_size_stop + 1, args.image_size_step))
            elif args.dimension == 3:
                image_sizes = list(range(args.image_size_start, min(args.image_size_stop + 1, 513), args.image_size_step))
        
            # Should we use mixed precision?
            precision_mode = "FP16" if args.use_mixed_precision and torch.cuda.is_available() else "FP32"
            

Distributed Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``physicsnemo.distributed.DistributedManager``, setting up the 1D or 2D parallelization is straightforward:


.. tab-set::

    .. tab-item:: Single GPU  

        .. code-block:: python

            # Initialize distributed manager first
            DistributedManager.initialize()
            dm = DistributedManager()





            # Set device based on local rank
            device = dm.device
            torch.cuda.set_device(device)

    .. tab-item:: DDP 

        .. code-block:: python

            # Initialize distributed manager first
            DistributedManager.initialize()
            dm = DistributedManager()

            # Set via commandline and argparse:
            ddp_size = args.ddp_size
            domain_size = args.domain_size

            # Set device based on local rank
            device = dm.device
            torch.cuda.set_device(device)

            

    .. tab-item:: ShardTensor + FSDP 

        .. code-block:: python

            # Initialize distributed manager first
            DistributedManager.initialize()
            dm = DistributedManager()

            # Set via commandline and argparse:
            ddp_size = args.ddp_size
            domain_size = args.domain_size

            # Set device based on local rank
            device = dm.device
            torch.cuda.set_device(device)
            
            # Use the physics nemo distribute manager to quickly and easily set up a pytorch DeviceMesh:
            mesh = dm.initialize_mesh(
                mesh_shape=(ddp_size, domain_size,), # -1 works the same way as reshaping
                mesh_dim_names = ["ddp","domain"]
            )
            ddp_mesh = mesh["ddp"]
            domain_mesh = mesh["domain"]
            

Preparing the Inputs
^^^^^^^^^^^^^^^^^^^^

We use synthetic inputs for this tutorial.  The global batch size is assumed to be configured on the command line - and when using domain parallelism, divide the global batch size by the number of model replications.  For 2D parallelism, we *still* divide the global batch size by the replicate count, but also apply a scatter.  Since we're parallelizing over the batch + domain, this means scattering one batch of data over an image axis (``Shard(2)`` below - remember the ``BCHW(D)`` format in pytorch, we're targeting ``H``):

.. tab-set::


    .. tab-item:: Single GPU 

        .. code-block:: python

            if args.dimension == 2:
                full_img_size = (img_size, img_size)
            elif args.dimension == 3:
                full_img_size = (img_size, img_size, img_size)
            


    .. tab-item:: DDP 

        .. code-block:: python

            if args.dimension == 2:
                full_img_size = (img_size, img_size)
            elif args.dimension == 3:
                full_img_size = (img_size, img_size, img_size)
            
            # Create synthetic data - scale the batch size down by DDP size.
            x = torch.randn(args.batch_size // ddp_size, 3, * full_img_size, device=device)
            target = torch.randint(0, num_classes, (args.batch_size // ddp_size,), device=device)


    .. tab-item:: ShardTensor + FSDP 

        .. code-block:: python

            if args.dimension == 2:
                full_img_size = (img_size, img_size)
            elif args.dimension == 3:
                full_img_size = (img_size, img_size, img_size)
            
            # Create synthetic data - scale the batch size down by DDP size.
            x = torch.randn(args.batch_size // ddp_size, 3, * full_img_size, device=device)
            target = torch.randint(0, num_classes, (args.batch_size // ddp_size,), device=device)
            
            # Domain Parallel NOTE: we're generating data once per GPU but only keeping the data once per domain.
            # In a real application, you'd do this properly - each GPU would read its own shard of the data.
            
            if args.domain_size > 1:
                
                # When scattering the data, we need to know the global rank of the source
                # But by definition, we use the domain_rank == 0 as the source.  Convert:
                global_rank_of_source = torch.distributed.get_global_rank(domain_mesh.get_group(), 0)
        
                # Scatter the input data across the domain:
                x = scatter_tensor(
                    x, 
                    global_rank_of_source, 
                    domain_mesh, 
                    placements=(Shard(2),), # Shard along the 2nd dimension (B C **H** W) which is the Height
                    global_shape = x.shape, # This will be inferred if not provided!
                    dtype = x.dtype, # This will be inferred if not provided!
                )

                target = scatter_tensor(
                    target, 
                    global_rank_of_source, 
                    domain_mesh, 
                    placements=(Replicate(),),  # REPLICATE the target
                    global_shape = target.shape, # This will be inferred if not provided!
                    dtype = target.dtype, # This will be inferred if not provided!
                )

Configure the Model
^^^^^^^^^^^^^^^^^^^

Configuring the model is easy - we build it as usual, and then use some ``torch`` functionality to distribute it across 1D or 2D parallelism:


.. tab-set::

    .. tab-item:: Single GPU 

        .. code-block:: python

            # Base model
            model = HybridViT(img_size = full_img_size, in_channels=3, num_classes=num_classes)
            model = model.to(device)
            


    .. tab-item:: DDP 

        .. code-block:: python

            # Base model
            model = HybridViT(img_size = full_img_size, in_channels=3, num_classes=num_classes)
            model = model.to(device)

            # Wrap model with DDP
            model = DDP(model, device_ids=[dm.local_rank], output_device=dm.local_rank)
        
    .. tab-item:: ShardTensor + FSDP

        .. code-block:: python

            # Base model
            model = HybridViT(img_size = full_img_size, in_channels=3, num_classes=num_classes)
            model = model.to(device)

            # This step syncs across the domain only
            model = distribute_module(
                model,
                device_mesh=domain_mesh,
                partition_fn = partition_model, # See below to understand what this is!
            )
            # This step goes in the other axis on the mesh: every rank "i" of
            # each domain will sync up here.
            model = FSDP(model, device_mesh=ddp_mesh, use_orig_params=False)


Above, in the *ShardTensor + FSDP* column, you may have noticed the presence of the ``partition_fn`` argument in ``distribute_module``.  You can read about it in more detail on the `PyTorch docs <https://docs.pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.distribute_module>`_, but to summarize: it lets you have full control over the way your model's parameters are sharded across the domain mesh.  Here, we're letting most parameters get replicated, but because this ViT includes a learnable position encoding that is the same size as the tokenized data (which we're sharding, of course), we can use the partition function to shard the embedding in the same way:

.. code-block:: python

    def partition_model(name, submodule, device_mesh):
    
        for key, param in submodule._parameters.items():
            if "pos_embed" in key:
                # Replace the pos_embed with a scattered ShardTensor
                # Global source is the global rank of local rank 0:
                scattered_pos_embed = distribute_tensor(
                    submodule.pos_embed,
                    device_mesh=device_mesh,
                    placements=[
                        Shard(1),
                    ],
                )
                submodule.register_parameter(key, torch.nn.Parameter(scattered_pos_embed))

The partition function is applied recursively to your module - this implementation doesn't do anything fancy except spot the parameter named ``pos_embed``, shard it, and replace it in the original model.  By default, all parameters that aren't converted here will get cast to ``DTensor`` (`docs <https://docs.pytorch.org/docs/stable/distributed.tensor.html>`_) - which is good, that's exactly what we want to happen for 2D parallelism!

**That's it, by the way** - after adding a few extra imports, setting up a ``DeviceMesh``, sharding the inputs, and distributing the model, everything else proceeds as usual.  You can run the benchmark with the same code across all three implementations:


.. tab-set::

    .. tab-item:: Single GPU  

        .. code-block:: python

            results = end_to_end_benchmark(args, model, (x, target), full_img_size, device, num_classes)
    
            if dm.rank == 0:
                print_and_save_results(results, args, precision_mode, dm.world_size)

    .. tab-item:: DDP 

        .. code-block:: python
    
            results = end_to_end_benchmark(args, model, (x, target), full_img_size, device, num_classes)
    
            if dm.rank == 0:
                print_and_save_results(results, args, precision_mode, dm.world_size)

    .. tab-item:: ShardTensor + FSDP 

        .. code-block:: python
        
            results = end_to_end_benchmark(args, model, (x, target), full_img_size, device, num_classes)
    
            if dm.rank == 0:
                print_and_save_results(results, args, precision_mode, dm.world_size)


.. note:: 
    Don't want to copy+paste?  Find the full training script and all worker functions, configurable by domain size and ddp size, on `GitHub <>`_

TODO - update this link!!

Benchmark results
-----------------

It wouldn't be a fun tutorial if we didn't show some of the results here, of course.  This also can be useful for deciding when you should use ``ShardTensor`` and when it's just fine to stick to DDP.  (Spoiler - use ShardTensor when you can't even fit ``batch_size==1`` on a single GPU!)

1024x1024 2D Image
^^^^^^^^^^^^^^^^^^

At a resolution of 1024 pixels on a side, our baseline ViT shows reasonable performance on a single GPU.  We can keep the per-GPU batch size fixed, and scale out with DDP, and get very good scaling - DDP is great at that.  We can also scale in two directions and see that latency, at fixed global batch size, decreases; however, ``ShardTensor`` isn't ideal in this regime:

.. tab-set::

   .. tab-item:: Training - Throughput

      **Training Throughput (Images / second)** - at 1024 pixels per side, decreases with more GPUs per image (aka, with ShardTensor) but total throughput is highest with each GPU responsible for a full image.

      +----------------+------+------+------+------+
      | GPUS / Image   | B=1  | B=2  | B=4  | B=8  |
      +================+======+======+======+======+
      | 1              | 0.46 | 0.91 | 1.8  | 3.6  |
      +----------------+------+------+------+------+
      | 2              | 0.76 | 1.6  | 3.1  | -    |
      +----------------+------+------+------+------+
      | 4              | 1.3  | 2.7  | -    | -    |
      +----------------+------+------+------+------+
      | 8              | 1.9  | -    | -    | -    |
      +----------------+------+------+------+------+

   .. tab-item:: Training Memory Usage

      **Training Memory Usage (GB)** - at this resolution, the model uses only 14Gb of GPU memory per image - out of 80Gb total.

      +----------------+------+------+------+------+
      | GPUS / Image   | B=1  | B=2  | B=4  | B=8  |
      +================+======+======+======+======+
      | 1              | 13.9 | 14.4 | 14.4 | 14.4 |
      +----------------+------+------+------+------+
      | 2              | 7.6  | 7.4  | 7.1  | -    |
      +----------------+------+------+------+------+
      | 4              | 4.5  | 4.2  | -    | -    |
      +----------------+------+------+------+------+
      | 8              | 2.9  | -    | -    | -    |
      +----------------+------+------+------+------+


``ShardTensor``, in most operations, does add a little overhead: most of the kernels that benefit from domain parallelism require communication between GPUs, and efficiency increases as the computational size increases from 1024 squared to 2048 squared:

.. tab-set::

   .. tab-item:: Latency

      **Latency per step (s)** - processing time increases linearly with the number of tokens in each layer - but tokens scale as the resolution *squared*.

      +----------------+----------------+----------------+----------------+----------------+
      | GPUs           | Inference 1024 | Train 1024     | Inference 1024 | Train 2048     |
      +================+================+================+================+================+
      | 1              | 0.55           | 7.96           | 2.2            | 31.4           |
      +----------------+----------------+----------------+----------------+----------------+
      | 2              | 0.32           | 4.13           | 1.32           | 16.4           |
      +----------------+----------------+----------------+----------------+----------------+
      | 4              | 0.19           | 2.23           | 0.76           | 8.78           |
      +----------------+----------------+----------------+----------------+----------------+
      | 8              | 0.13           | 1.33           | 0.54           | 5.02           |
      +----------------+----------------+----------------+----------------+----------------+

   .. tab-item:: Speedup

      **Speedup** - After a certain data size, ``ShardTensor`` is always faster with more GPUs.  But, larger images show bigger benefits.

      +----------------+----------------+----------------+----------------+----------------+
      | GPUs           | Inference 1024 | Train 1024     | Inference 1024 | Train 2048     |
      +================+================+================+================+================+
      | 1              | 1.0            | 1.0            | 1.0            | 1.0            |
      +----------------+----------------+----------------+----------------+----------------+
      | 2              | 1.7            | 1.9            | 1.7            | 1.9            |
      +----------------+----------------+----------------+----------------+----------------+
      | 4              | 2.9            | 3.6            | 2.9            | 3.6            |
      +----------------+----------------+----------------+----------------+----------------+
      | 8              | 4.2            | 6.0            | 4.1            | 6.3            |
      +----------------+----------------+----------------+----------------+----------------+

   .. tab-item:: Memory Usage

      **Memory Usage (GB)** - Like latency, memory usage in training scales roughly like the number of tokens.  For inference, it's driven mostly by model size.

      +----------------+----------------+----------------+----------------+----------------+
      | GPUs           | Inference 1024 | Train 1024     | Inference 1024 | Train 2048     |
      +================+================+================+================+================+
      | 1              | 2.5            | 13.9           | 4.6            | 51.4           |
      +----------------+----------------+----------------+----------------+----------------+
      | 2              | 2.2            | 7.6            | 3.8            | 26.5           |
      +----------------+----------------+----------------+----------------+----------------+
      | 4              | 2.0            | 4.5            | 2.8            | 13.9           |
      +----------------+----------------+----------------+----------------+----------------+
      | 8              | 1.9            | 2.9            | 2.3            | 7.6            |
      +----------------+----------------+----------------+----------------+----------------+

   .. tab-item:: Memory Reduction

      **Memory Reduction (%)** - For highest resolution data, we see close to linear reduction in memory with more GPUs.

      +----------------+----------------+----------------+----------------+----------------+
      | GPUs           | Inference 1024 | Train 1024     | Inference 1024 | Train 2048     |
      +================+================+================+================+================+
      | 1              | 100%           | 100%           | 100%           | 100%           |
      +----------------+----------------+----------------+----------------+----------------+
      | 2              | 88%            | 55%            | 83%            | 52%            |
      +----------------+----------------+----------------+----------------+----------------+
      | 4              | 80%            | 32%            | 61%            | 27%            |
      +----------------+----------------+----------------+----------------+----------------+
      | 8              | 76%            | 21%            | 50%            | 15%            |
      +----------------+----------------+----------------+----------------+----------------+

If you're tracking the memory scaling performance of this model, you'll see that the training memory at higher resolution is roughly proportional to the total number of pixels in the image.  At 51.4 Gb of training memory for 2048x2048 sized images, we expect the next step up (4096x4096 pixels) to require more than 200GB of memory per GPU.  Of course, with ``ShardTensor``, we can run it out of the box on 8 GPUs - and we see about 26Gb of memory used, per GPU, as expected.  You can even run large scale 3D vision models like this - since the memory usage scales with the resolution **cubed** instead of squared, memory issues appear even faster.

Key Points
----------

Hopefully, by this point, you understand the key steps to enable ``ShardTensor`` in your model.  Know that it's still a work in progress for both performance and broad layer support.  Many key models will work out of the box; some will have operations that aren't yet fully supported.  If you have specific requests for support, please open an issue on `GitHub <https://github.com/NVIDIA/physicsnemo/tree/main>`_ - and check out the tutorial for :ref:`supporting layers yourself <implementing_new_layers_in_shard_tensor>` if you don't want to wait!

To recap the workflow for 2D, domain parallelism: 
1. The device mesh is split into two dimensions: one for data parallelism (``FSDP``) and one for spatial decomposition (``ShardTensor``).  We get that in one line using torch DeviceMesh: ``mesh = dm.initialize_mesh((-1, 2), mesh_dim_names=["data", "spatial"])``.  And in fact, for multilevel parallelism, you can extend your mesh further.  Think of DeviceMesh like a tensor of arbitrary rank, and each element is one GPU.
2. Input data is sharded across the spatial dimension using ``ShardTensor``
3. ``FSDP`` handles parameter sharding and optimization across the data parallel dimension
4. The model can process larger spatial dimensions efficiently by distributing the computation

We hope this tutorial helps you scale your models and data to massive resolutions!

Domain Parallelism and Shard Tensor
===================================

In scientific AI, one of most challenging aspects in training a model is dealing with extremely high resolution data.  In this
tutorial, we'll explore what makes high resolution data so challenging to handle, for both training and inference, and why that's different from the scaling challenges in other domains (like NLP, image processing, etc.).  We'll also take a technical look at how we're working to streamline high-resolution model training in ``PhysicsNeMo``, and how you can leverage our tools for your own scientific workloads as well.

What makes scientific AI challenging?
------------------------------------

To understand why scientific AI hits unique challenges in training and inference on high resolution data, let's take a look at the computational and memory cost of training models and subsequently running inference.  "Cost" here refers to two fundamental, high level concepts: computational cost is how much computing power is needed to complete an operation (and is, in general, a complicated interplay of GPU FLOPs, memory bandwidth, cache sizes, algorithm efficiencies, and more); memory costs refer to the amount of GPU memory required to perform the computations.

For all AI models, the memory cost of inference is dominated by just two categories of use:

1. **Model parameters** (weights, biases, encodings, etc.) all are required to be loaded into GPU memory for fast access during inference.  For a model with N total parameters, each parameter requires 4 bytes in float32 precision, or 2 in float16/bfloat16.  A rough approximation is that a 100M parameter model requires 400MB of memory  in float32 precision.  For Large Language Models with billions of parameters, even at inference time this is a large amount of memory. 

2. **Active Data** (the inputs and outputs!) represent the memory required to actually compute the layers and outputs of the model.  For inference, the available memory has to be enough to hold the input data, output data, and model parameters as well as temporariliy accommodate memory of intermediate activations.  As one layer's output is consumed by the next layer, the total memory needed typically never exceeds the requirements of the most memory-intensive layer.

For scientific AI with high resolution data, the memory cost at inference can be dominated not by the model parameters but by the data - though it's not always a clear cut winner.

During training (for a standard training loop), the high resolution of the data is even more challenging.  There are two additional memory consumers during a model training, in most cases:

3. **Optimizer states** (gradients, moments) are needed to accumulate and update the model's parameters during training.  This can be as little memory usage as the model's parameters, again, for SGD.  For more complicated optimizers, like ``adam``, the optimizer must store moments and running gradient averages and the usage increases.

4. **Activations** For each layer during training, pytorch will typically save the some version of the layer's input, output, or other component as the "intermediate activation" for that layer.  In practice, this is a computational optimization to enable the backwards pass to compute and propagate gradients more efficiently.  Each layer, however, requires extra memory storage during training that is proportional to the resolution of the input data.  

As a cumulative effect, as models continue to stack up layers and save intermediate activations, the activation-related memory required training a model grows with both the depth of the model and the resolution of the input data.  In contrast to Large Language Models, where the memory usage during training is dominated by the parameters, gradients, and optimizer states, for high resolution scientific AI models with modest parameter counts the memory usage is dominated by actications!

To address this challenge, in PhysicsNeMo we have developed a domain-parallelism framework specifically designed to parallelize the high compute and memory costs of training and inferencing models on high resolution data.  Named ``ShardTensor``, and built on top of PyTorch's ``DTensor`` framework, ``ShardTensor`` allows models to divide expensive operations across multiple GPUs - parallelizing both the compute required as well as the storage of the intermediate activations.

The remainder of this tutorial will focus on the high level concepts of ``ShardTensor`` and domain parallelism, and :ref:`Implementing new layers for ShardTensor`  will be covered in a separate tutorial.

Starting with an Example
----------------------

As a high level example, let's consider a simple 2D convolution operation.  There have been many tutorials on the mathematics and efficient computation of convolutions; let's not focus on that here.  Instead, consider if the input data to the convolution is spread across two GPUs, and we want to correctly compute the ouput of the convolution but without ever coalescing the input data on a single GPU.

Just applying the convolution to each half provides incorrect results.  We can simulate this, actually, in pytorch on one device:

.. code-block:: python

    import torch

    full_image = torch.randn(1, 8, 1024, 1024)

    left_image = full_image[:,:,:512,:]
    right_image = full_image[:,:,512:,:]

    convolution_operator = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)

    full_output = convolution_operator(full_image)

    left_output = convolution_operator(left_image)
    right_output = convolution_operator(right_image)

    recombined_output = torch.cat([left_output, right_output], dim=2)

    # Do the shapes agree?
    print(full_output.shape)
    print(recombined_output.shape)
    # (they should!)

    # Do the values agree?
    torch.allclose(full_output, recombined_output)
    # (they do not!)

To understand why they don't agree, we can look at the location of the disagreement:

.. code-block:: python

    diff = full_output - recombined_output
    b_locs, c_locs, h_locs, w_locs = torch.where( torch.abs(diff) > 1e-6)
    print(torch.unique(b_locs))
    print(torch.unique(c_locs))
    print(torch.unique(h_locs))
    print(torch.unique(w_locs))

This will produce the following output:

.. code-block:: text

    tensor([0])
    tensor([0, 1, 2, 3, 4, 5, 6, 7])
    tensor([511, 512])
    tensor([   0,    1,    2,  ..., 1021, 1022, 1023])

We see in particular that along the height dimension (dim=2), the output is incorrect only along the pixels 511 and 512 - right where we split the data!  The problem is that the convolution operator is a local operation, but splitting the data prevents it from seeing the correct neighboring pixels right at the border.  You could fix this directly:

.. code-block:: python

    # Slice off the data needed on the other image (around the center of the original image)
    missing_left_data = right_image[:,:,0:1,:]
    missing_right_data = left_image[:,:,-1:,:]

    # Add it to the correct image
    padded_left_image = torch.cat([left_image, missing_left_data], 2)
    padded_right_image = torch.cat([missing_right_data, right_image], 2)

    # Recompute convolutions
    right_output = convolution_operator(padded_right_image)[:,:,1:,:]
    left_output = convolution_operator(padded_left_image)[:,:,:-1,:]
    # ^ Need to drop the extra pixels in the output here

    recombined_output = torch.cat([left_output, right_output], dim=2)

    # Now, the output works correctly:
    torch.allclose(recombined_output, full_output)
    # True

In the example above, for a simple convolution, we saw that just splitting the data and applying the base operation didn't give the results we needed. In general, this is true of many operations we see in AI models: splitting the data across GPUs requires extra operations or communication, depending on the operation, to get everything right.  We also haven't even mentioned the gradients yet - to call ``backward()`` through this split operation across devices also requires extra operations and communication.  But, in order to get the memory and potential computational benefits of domain parallelism, it's necessary.

How does ``ShardTensor`` help?
-----------------------------

PyTorch's ``DTensor`` interface already has an interface for a distributed tensor mechanism, and it's great - great enough, in fact, that ``ShardTensor`` is built upon it.  However, ``DTensor`` is built with a different paradigm of parallelism in mind, including model parallelisms from `DeepSpeed <https://www.deepspeed.ai/getting-started/>`_ and `MegaTron <https://developer.nvidia.com/megatron-core>`_ - which is supported in pytorch via `Fully Sharded Data Parallelism <https://pytorch.org/docs/stable/fsdp.html>`_.  It has several shortcomings: notably, it can not accommodate data that isn't distributed uniformly or according to ``torch.chunk`` syntax.  For scientific data, such as mesh data, point clouds, or anything else irregular, this is a nearly-immediate dead end for deploying domain parallelism.  Further, ``DTensor``'s mechanism for implementing parallelism is largely restricted to lower level ``torch`` operations - great for broad support in PyTorch, but not as accesible for most developers.

With ``ShardTensor``, we extend the functionality of ``DTensor`` in the ways needed to make domain parallelism simpler and easier to apply.  In practice, this looks like the following, if we reuse the convolution example from before:

.. literalinclude:: ../../test_scripts/domain_parallelism/sharded_conv_example.py
    :caption: Example of domain parallel convolution with ``ShardTensor``
    :language: python


If you run this (``torchrun --nproc-per-node 4 conv_example.py``), you'll see the checks on output and gradients both pass.  Further, the last line will print:

.. code-block:: text

    Distributed grad sharding and local shape: (Shard(dim=2),), torch.Size([1, 8, 256, 1024])

Note that when running this, there was no need to perform manual communication or padding, in either the forward or backward pass.  And, though we used a convolution, the details of the operation didn't need to be explicitly specified.  In this case, it just worked.

How does ``ShardTensor`` work?
-----------------------------

At a high level, ``DTensor`` from pytorch is a concept of a local chunk of a tensor (stored as a ``torch.Tensor``), and a ``DTensorSpec`` object which combines a ``DeviceMesh`` object representing the group of GPUs the tensor is on, and a description of how that global tensor is distributed (or replicated).  ``ShardTensor`` extends this API with an addition to the specification to track the shape of each local tensor along sharding axes.  This becomes important when the input data is something like a point cloud, rather than an evenly-distributed tensor.

At run time, when an operation in ``torch`` has ``DTensor`` as input, pytorch will use a custom dispatcher in ``DTensor`` to route perform operations correctly on the inputs.  ``ShardTensor`` extends this by intercepting a little higher than ``DTensor``: operations can be intercepted at the functional level, or at the dispatch level, and if ``ShardTensor`` has no registered implementation it will fall back to DTensor.

ShardTensor also has dedicated implementations of common reduction operations ``sum`` and ``mean``, in order to properly intercept and distribute gradients correctly.  This is why, in the example above, you can seamlessly call ``mean().backward()`` on a ``ShardTensor`` and the gradients will arrive to their proper sharding.  No need to do anything special - reducing a ``ShardTensor`` will handle this automatically.

There is a substantial amount of care needed to implement layers in ``ShardTensor`` (or ``DTensor``!).  If you're interested in doing so for your custom model, please check out a full tutorial on this subject: :ref:`Implementing new layers for ShardTensor`

When Should You Use ``ShardTensor``?
==================================

``ShardTensor`` and domain parallelism solve a very specific problem in Scientific AI: input data is such high resolution that models can't train, even at Batch Size of 1, due to memory limitations.  And while that challenge can be partially surmounted with reduced precision and input spatial downsampling, not all models can tolerate those techniques without sacrificing accuracy.  In this case, you should view ``ShardTensor`` as a solution to that problem: it will enable you to run training and inference on higher resolution data than a single GPU can accommodate.  It is not the only technique for this, and in some cases it isn't the best choice.  In this section we'll compare and contrast ``ShardTensor`` to some other techniques for high resolution data, which can highlight some strengths and weaknesses of ``ShardTensor.``

One other technique for high resolution data is `Pipeline Parallelism <https://docs.pytorch.org/docs/stable/distributed.pipelining.html#>`_.  In pipeline parallelism, the model is divided across 2 or more devices, and each device contains full layers and activations, but to run the entire model the data is "pipelined": input data on GPU 0 is propagated through the local layers, and the outputs of the last layer on GPU 0 become the inputs to the first layer on GPU 1, and so on.  Gradients can be computed by running the pipeline in reverse, as well.

For some use cases, pipeline parallelism can be very powerful.  But it also has some weaknesses that ``ShardTensor`` can avoid.  Pipeline parallelism enables scaling of GPU memory resources but does not take much advantage of scaling up GPU compute resources without modifying the training loop.  While GPU 0 is active, all other GPUs are waiting on input.  And once GPU 0 passes data to GPU 1, GPU 0 sits idly until the backward pass or the next batch of data arrives.  For large minibatch data, a good strategy could be to feed each batch of data sequentially: when data passes from GPU 0 to GPU 1, the next example can start processing on GPU 0.  For inference on large datasets, this is quite efficient, but during training this may cause a computational "bubble" or stall everytime gradients are computed and the model is updated.

With just one, or at most a few, point(s) in the model where pipeline parallelism divides your model, it is conceptually simple and each GPU has minimal communication overhead.  However, not all models are well supported with pipeline parallelism (consider a UNet architecture).  On the other hand, ``ShardTensor`` enables you to slice your model by dividing each and every layer over sharded inputs.  In terms of model support, this makes more complicated architectures like UNet simple: the concatenation of features across the down/up sampling paths is unmodified in user space (and in fact it's pretty simple in sharded implementations too: it becomes a concat of the local tensor objects).  On the other hand, because each layer introduces additional overhead of communication or coordination, a sharded layer can be less efficient than a purely-local layer.

As a general rule, ``ShardTensor`` performs efficiently when the input data is large, and when the ratio of communication time to computation time is small. For some operations, like sequence-parallel attention via a Ring Mechanism (`Ring Attention <https://arxiv.org/pdf/2310.01889>`_), the benefits become clear, as shown below: the sharded model is faster after a certain input data size. More importantly, the sharded model is still **functional** after a massive input size—something pipeline parallelism could not achieve for a simple one-layer model.

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. image:: ../../img/domain_parallelism/training_latency_vs_sequence_length_8_heads_256_dim_backward.png
          :width: 100%
          :alt: Training latency vs sequence length

     - .. image:: ../../img/domain_parallelism/inference_latency_vs_sequence_length_8_heads_256_dim.png
          :width: 100%
          :alt: Inference latency vs sequence length

.. centered:: **Figure:** Left: The latency of a single forward/backward pass, over multiple GPUs with ``ShardTensor``, as compared to a baseline implementation. At larger sequence lengths, scaling efficiency exceeds 95% on 8 GPUs. Right: Inference performance showing how domain parallelism provides reduced latency for high resolution data processing.


Of course, a one-layer model isn't a good representation of actual user code.  Instead, use this as a guiding principle: when the GPU kernels are long because the input data is large, ``ShardTensor`` will scale very efficiently.  When GPU kernels are small, and a model launches many small kernels, ``ShardTensor`` will be functional but not as efficient.  In these cases you may have slightly better scaling with pipeline or other parallelism.  Note, however, that ``ShardTensor`` is still in development and performance optimizations for small kernels are ongoing.

Another technique for dealing with high resolution input data during training is activation checkpointing.  In this technique, during the forward pass, activations are moved from GPU memory to CPU memory to make more space available.  They are restored during the backward pass when needed, and the rest of the backward pass continues.  Compared to pipeline parallelism, this technique can better leverage parallelization across GPUs with standard Data-Parallel scaling.  However, it can be limited by GPU/CPU transfer speeds and possible blocking operations.  On NVIDIA GPUs with NCCL enabled, the peer-to-peer bandwidth can be significantly higher than CPU-GPU bandwidth (though not all - GraceHopper systems, for example, can efficiently and effectively take advantage of CPU memory offloading).  Unlike ``ShardTensor``, the offloading of activations may need hand tuning and optimization based on GPU system architecture.  ``ShardTensor`` is designed to work with your model ``as-is`` to the greatest possible extent.

In general, if your model meets all of these conditions, you should consider using ``ShardTensor`` for domain parallelism during training:

- Your model has relatively large input size even at batch size of 1 - so large, in fact, that you run out of GPU memory trying to train the model with batch size 1.
    - If your model comfortably fits batch_size=1 training, you will have a simpler and more efficient training using PyTorch's `DistributedDataParallel <https://pytorch.org/docs/stable/ddp.html>`_
- Your model is composed of supported domain-parallel layers (convolutions, normalizations, upsampling/pooling/reductions, attention layers, etc.)
    - Not every layer has a domain-parallel implementation in PhysicsNeMo.  You can add it to your code yourself if it's simple (consider a P.R. if you do!) or ask for support on github.
    - How do you know if a layer is supported?  Pass a ``ShardTensor`` in like above and test it!
- You have multiple GPUs available (ideally connected with high-performance peer to peer path such as NCCL).

For the best efficiency training with ``ShardTensor``, look for:

- Your model is mostly composed of large, compute- or bandwidth-bound kernels rather than very small, latency-bound kernels.

- Your model is composed of mostly non-blocking CUDA kernels, allowing the slightly higher overhead of domain parallelism to still fill the GPU queue efficiently.

For inference, on the other hand, ``ShardTensor`` can still be useful for lower latency inference on extremely high resolution data.  Especially if the model is primarly composed of compute- or bandwidth-bound kernels, and the commmunication overhead is small, ``ShardTensor`` can provide reductions of inference latency.

Summary
=======

In this tutorial, we saw details about PhysicsNeMo's ``ShardTensor`` object, and how it can be used to enable domain parallelism.  For more behind-the-scenes details of how layers are enabled, see :ref:`Implementing new layers for ShardTensor`.  For an example of combining domain parallelism with other parallelisms through FSDP, see `fsdp_and_shard_tensor :ref:`Domain Decomposition, ShardTensor and FSDP Tutorial`.

Glossary
========

- **DeviceMesh**: A pytorch abstraction that represents a set of connected GPUs.  See `DeviceMesh <https://docs.pytorch.org/docs/stable/distributed.html#devicemesh>`_ for more details.  ``DeviceMesh`` is particularly useful for multilevel parallelism (data parallel training + domain parallelism, for example).

- **DTensor**: PyTorch's distributed tensor object.  See `DTensor <https://docs.pytorch.org/docs/stable/distributed.tensor.html>`_ for more details.

- **ShardTensor**: PhysicsNeMo's distributed extension to ``DTensor``.  In particular, ``ShardTensor`` removes requirements for even data distribution (though it's still optimal for computational load balancing) and implements domain parallel paths for many operations.

- **NCCL**: NVIDIA's collective communication library for high speed GPU-GPU communication.  See `NCCL <https://developer.nvidia.com/nccl>`_ for more details.

- **DDP**: PyTorch's distributed data parallel training system.  See `DDP <https://pytorch.org/docs/stable/ddp.html>`_ for more details.

- **FSDP**: PyTorch's fully sharded data parallel training system.  ``FSDP`` is an superset of ``DDP``, and to use ``ShardTensor`` domain parallelism you must use ``FSDP``, not ``DDP``.  See `FSDP <https://pytorch.org/docs/stable/fsdp.html>`_ for more details.

- **DeepSpeed**: A distributed training and inference framework for large language models, built fully sharding weights, gradients, and optimizer states.  See `DeepSpeed <https://www.deepspeed.ai/>`_ for more details.

- **MegaTron**: Another distributed training and inference framework for large language models, built on sharding weights along the channel dimension with optimized Attention collectives.  See `MegaTron <https://developer.nvidia.com/megatron-core>`_ for more details.
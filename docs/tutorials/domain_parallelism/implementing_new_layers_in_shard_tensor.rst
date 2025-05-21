Implementing new layers for ShardTensor
=========================================

This tutorial is a walkthrough of how to extend domain parallel functionality via ``ShardTensor``.  We'll first discuss at a high level some parallelism techniques, and then look at exactly how to implement a domain parallel layer with a few examples.  For some background on what `ShardTensor` is and when to use it, check out the tutorial :ref:`domain_parallelism.rst`.

When is extending ``ShardTensor`` needed?
---------------------------------------

``ShardTensor`` is designed to support domain-parallel operations, or operations that can be performed on a tensor that resides across multiple devices. Many operations are supported already - out of the box - by the upstream ``DTensor`` class that ``ShardTensor`` inherits from.  Some operations - many convolutions, interpolations, poolings, normalizations, and attention - are supported through ``PhysicsNeMo``.  In this tutorial, we'll look at a few increasingly-complicated situations and see how ``ShardTensor`` handles them - or doesn't - and how to fix cases that aren't supported or aren't performant.


Example 0: Vector Addition
--------------------------

As a basic example (and note that this is a built-in operation from ``DTensor``), let's implement a shard tensor version of ``torch.add()``.  Here's a single-device implementation:

.. literalinclude:: ../../test_scripts/domain_parallelism/new_layers/vector_add_baseline.py
    :caption: Example 0: Vector Addition, single device
    :language: python


To perform this with ``ShardTensor``, we first need to convert these tensors to ``ShardTensor`` objects.  The easiest way to do this is with the ``scatter_tensor`` method:

.. literalinclude:: ../../test_scripts/domain_parallelism/new_layers/vector_add_sharded.py
    :caption: Example 0: Vector Addition, distributed computation
    :language: python

This will run, out of the box (and in fact doesn't even need ``ShardTensor``, ``DTensor`` implements distributed vector addition).  If you have a multi-GPU system, execute the code with a command like ``torchrun --nproc-per-node 8 example_0_sharded.py``.  You ought to see pretty good scaling efficiency - with no communication overhead, the distributed operation can work at approximately weak scaling speeds.  For small tensors, though, where the addition operation is bound by launch latency: you will see a slightly higher overhead with distributed operations because there is slightly more organization and bookkeeping required.
    

Example 1: Vector Dot Product
-----------------------------

Let's look now at a slightly more complicated example: the dot product of two vectors.  In this case, because the output is a single scalar, we'll find that there *is* communication required and see how to implement that seamlessly with ``ShardTensor``.

Here's the single-device implementation.  Note that the **only** difference here is in the definition of ``f``:

.. code-block:: python

    def f(x, y):
        return torch.dot(x, y)

For reference, here's the full code:

.. literalinclude:: ../../test_scripts/domain_parallelism/new_layers/vector_dot_baseline.py
    :caption: Example 1: Vector Dot Product, single device
    :language: python


If we make the same changes to the distributed version, we get an error when we run it (as of torch 2.6!):

.. code-block:: bash

    NotImplementedError: Operator aten.dot.default does not have a sharding strategy registered.

This is a good time to talk about how PyTorch decides what to do each time it's called for an operation on ``torch.Tensor(s)``, which will lead into how we fix this error.

PyTorch, as you likely already know, implements operations on multiple backends and with multiple paths for execution.  How does it decide which path to use, when you call an operation on a tensor?  The answer lies in the PyTorch ``__torch_function__`` and ``__torch_dispatch__`` interface.

There are many resources, more detailed and more correct than this (for example, see `this blog post <https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557>`_ or `this one <https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ and especially the `official walkthrough <https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough>`_), but here is a high level overview: function routing is built on input types, rather than functions themselves. So when you call a function with an object (like ShardTensor) that extends the PyTorch ``torch.Tensor`` interface, you can use ``__torch_function__`` and ``__torch_dispatch__`` to capture and reroute operations to custom implementations.

For built in functions to PyTorch, this is simply a matter of registering a pair of functions with ``ShardTensor``: the function you want to intercept, and the function you want to route data to instead (as long as at least one argument is a ``ShardTensor``).  We'll see this in action below, but in the case of functions that ``torch`` does not know about (external functions, user functions, etc.), we can tap into this system manually. 


With all that in mind, let's add a handler for ``torch.dot`` that works on PhysicsNeMo's ``ShardTensor``:

.. code-block:: python

    from physicsnemo.distributed import DistributedManager, scatter_tensor, ShardTensor
    from torch.distributed.tensor.placement_types import Shard, Replicate

    def sharded_dot_product(func, types, args, kwargs):
        # NOTE: all functions overloaded and used by __torch_function__ will have 
        # the same input signature.  You can use python argument unpacking to 
        # extract what you need:
        def extract_args(x, y, *args, **kwargs):
            return x, y
        x, y = extract_args(*args, **kwargs)
        
        # Each tensor has a _spec attribute, which contains information about the tensor's placement
        # and the devices it lives on:
        x_spec = x._spec
        y_spec = y._spec
        
        # It's usually good to ensure the tensor placements work:
        if not x_spec.placements == y_spec.placements:
            raise NotImplementedError("Tensors must be sharded on the same device")
        
        if not x_spec.mesh == y_spec.mesh:
            raise NotImplementedError("Tensors must be sharded on the same mesh")
        
        # And, you might want to check placements are valid in more complex cases
        
        # Extract the mesh - we'll want it for the all reduce:
        mesh = x_spec.mesh
        
        # This is a straightforward implementation, for clarity
        # Get the local values of each tensor:
        local_x = x.to_local()
        local_y = y.to_local()
        
        # This is a purely single-gpu operation:
        local_dot_product = torch.dot(local_x, local_y)
        # If you wanted to write a generic sharding handler for this type of operation, 
        # you could do:
        # local_dot_product = func(local_x, local_y)
        # But it's over kill here...
        
        # SUM_Reduce the local result across all ranks:
        dist.all_reduce(local_dot_product, op=dist.ReduceOp.SUM, group=mesh.get_group())

        # We do want to return the result as a ShardTensor, for consistency.
        # We can easily create one on the same mesh as a "Replicated" tensor:

        output = ShardTensor.from_local(
            local_tensor = local_dot_product, 
            device_mesh =  mesh, 
            placements = (Replicate(),)
        )

        return output

    # Don't forget to register it with ShardTensor:
    ShardTensor.register_function_handler(torch.dot, sharded_dot_product)

Once you have registered a path for ShardTensor to do this computation, you can run the same code as before and it should work out of the box.  For completeness, here's the full code:

.. literalinclude:: ../../test_scripts/domain_parallelism/new_layers/vector_dot_sharded.py
    :caption: Example 1: Vector Dot Product, distributed computation
    :language: python


You should now be able to run this code with ``torchrun --nproc-per-node 8 example_1_sharded.py``.  You should see a significant nearly linear scaling efficiency over NVLink-connected devices.

Example 2: Nearest Neighbors 
----------------------------

With some basics out of the way, let's look at something a little more interesting and useful.  In many scientific AI workloads, we need to do a query of the nearest neighbors of a point cloud to build a GNN.  PyTorch doesn't really have an efficient implementation of a nearest neighbor operation - let's write one here (poorly!) just to show how it can be parallelized.

.. note:: 
    There are much better ways to write a kNN to operate on PyTorch tensors - don't use this in production code!  This is a brute force implementation. Most times when you need the nearest neighbors of a point cloud, some sort of KDTree or hash mapping structure is significantly more efficient. We're not using that in this tutorial for clarity, but when we need these operations in ``physicsnemo`` we use optimized implementations backed by libraries like ``cuml`` (see the `cuml documentation <https://docs.rapids.ai/api/cuml/stable/>`_) and ``warp``. (`documentation <https://developer.nvidia.com/warp-python>_`).


.. literalinclude:: ../../test_scripts/domain_parallelism/new_layers/knn_brute_force_baseline.py
    :caption: Example 2: Nearest Neighbors, single device
    :language: python


If you run this (``python example_2_baseline.py``), you'll see that it's not quite as quick as the other examples - it's also really memory intensive.  At the time of this tutorial's publication, we saw about 1.544 seconds for 10 runs on a single A100, or 150ms per call.  Additionally, this line will allocate memory of ``N_points_to_search  * N_target_points * 3 * 4 `` bytes (=32 GB in fp32!):

.. code-block:: python

    displacement_vec = x[None, :, :] - y[:, None, :]

As written, you can't really scale this up larger on a single device.  There are - as noted - better ways to do a kNN but let's parallelize this one since it's a good way to learn how you might parallelize custom functions.  In fact, a functional parallelization couldn't be easier - do nothing but cast the inputs to ``ShardTensor`` and let the existing operations take care of it.  The underlying implementations of ``ShardTensor`` and ``DTensor`` enables this out of the box:

.. literalinclude:: ../../test_scripts/domain_parallelism/new_layers/knn_brute_force_sharded.py
    :caption: Example 2: Nearest Neighbors, distributed computation, basic functionality
    :language: python


Go ahead and pause, and run these codes with ``torchrun --nproc-per-node 8 example_2_sharded.py``.  You should see a good speedup - we saw about 33 ms per call on a single A100.  Compared to 150ms, that's a nice improvement, about 4.5x faster ... but we're also using 8 GPUs.  Why isn't it 8x faster?

The issue is once again in this line:

.. code-block:: python

    displacement_vec = x[None, :, :] - y[:, None, :]

Except this time, the ``x`` and ``y`` tensors are being subtracted when their sharded axes disagree.  ``x[None,:,:]`` will shift the placement of the shards of ``x`` from ``Shard(0)`` to ``Shard(1)``, while ``y[:,None,:]`` will not shift the shards of ``y`` from ``Shard(0)``.  When ``DTensor`` does the subtraction (remember - it's the fallback handler for ``ShardTensor`` when we haven't implemented a custom handler), it makes the decision to replicate one of these tensors (the first one, here), and leaves that axis replicated in the output.  Like this:


.. literalinclude:: ../../test_scripts/domain_parallelism/new_layers/reshape_subtract.py
    :caption: Example of automatic resharding by ``DTensor``
    :language: python

You'll see this output:

.. code-block:: text

    a_sharded shape and placement: torch.Size([234567, 3]), (Shard(dim=0),)
    b_sharded shape and placement: torch.Size([12345, 3]), (Shard(dim=0),)
    a_sharded shape and placement: torch.Size([1, 234567, 3]), (Shard(dim=1),)
    b_sharded shape and placement: torch.Size([12345, 1, 3]), (Shard(dim=0),)
    distance_vec shape and placement: torch.Size([12345, 234567, 3]), (Shard(dim=1),)

It's nice, of course, that ``DTensor`` will get this numerically correct out of the box - but it's not the most efficient way we could do something like this.  Instead, we can write the ``knn`` function to use a ring-based computation: compute the knn on local chunks, and then shift the slices of the point cloud along the mesh to compute the next iteration.  It requires more collectives, but because we can overlap the communication and computation - and never have to construct the entire distance matrix - it's more efficient.

.. literalinclude:: ../../test_scripts/domain_parallelism/new_layers/knn_brute_force_ring_sharded.py
    :caption: Example 2: Nearest Neighbors, distributed computation, ring-based computation
    :language: python

Run this (``torchrun --nproc-per-node 8 example_2_sharded.py``), and you'll see the time per iteration is more like 20.7ms.  That's an 8x speed up over the original, single device implementation - much better!

.. note:: 
    There is an important piece of that previous example, in case you overlooked it.  The ``knn`` function has a few extra lines registering it with PyTorch's overrides system (`torch.overrides <https://docs.pytorch.org/docs/stable/torch.overrides.html>`_). This step lets PyTorch track the ``knn`` function, and registering it with ``ShardTensor`` sends execution to the ``knn_ring`` function instead.  When that function in turn calls the ``knn`` function on standard ``torch.Tensor`` objects, it is executed normally on the local objects.

What collectives do I need for my operation?
--------------------------------------------

If you're looking to extend ``ShardTensor`` to support a new domain parallelism operation, it can fall into one of several - not exhaustive - categories.  Use this to guide your thinking about performant domain-parallel implementations.

- **Fully Local** operations can be computed locally at every value of a tensor, with a one-to-one mapping between input and output tensors.  Activations are an obvious example of this, but tensor-wise math can be too: ``c = a + b``, where ``a`` and ``b`` are both tensors, can follow this pattern (absent reshaping/broadcasting, as we saw above, which complicates things).  In these cases, the "domain parallel" component of an operation is really just a purely local operation + making sure the output tensor ``c`` is represented properly as a distributed object.  No communication is needed.

- **Semi-Local** operations depend on neighboring values, but not on _every_ value.  Depending on the details of operation, and the pattern of distributing a tensor across devices, to correctly perform this operation some information at the edges of each local tensor may need to be exchanged.  One example of this is `convolution` operations, where information must be exchanged across the domain decomposition boundary for most cases.  A more complicated example is a distributed graph, where some graph nodes share an edge that spans a domain boundary.  In many cases, this type of information exchange across a boundary is referred to as a 'halo'.  As long as the halo is small compared to the computation, these operations scale well through domain decomposition.

- **Reduction** based operations that require a large scale reduction of data, such as ``sum`` but also normalization layers, can usually be implemented in two or less passes: first, compute local reductions on the local piece of a tensors, and ``allreduce`` it across the domain.  Then, update the output by applying a correction factor based on the local to global statistics calculated.  

- **Global**  operations require a global view of tensors to compute correctly: each output point depends on information from all possible input locations.  Two examples that appear very different but are both global for domain decomposition are _Attention_ mechanisms, and distance-based queries on point clouds such as the _kNN_ we implemented earlier.  In both cases, one particular value of output can depend on any or all values of input tensors.  There are multiple ways to proceed for these operations, but in this case a ``ring`` collective can be quite efficient: tensors will perform the computation on local chunks, and then part of the input (KV, for attention, or part of the point cloud) will be passed to the next rank in a ring topology.  With overlapping communication and computation, these algorithms can achieve excellent scaling properties.  A challenge may be that the outputs of each iteration of the ring may need to be combined in non-intuitive ways.  `Ring Attention <https://arxiv.org/pdf/2310.01889>`_, which inspired the implementation in ``PhysicsNeMo``, necessitates log- and sign-based accumulation of outputs.  The ``kNN`` layer has two ``topk`` calls per iteration - one for the real operation, and one to combine the output.

It isn't necessarily true that all operations fall in to these categories for domain parallelism.  However, thinking about the way input data is decomposed, how output data must be decomposed, and what communication patterns are needed will often be enough to guide you to a correct, efficient implementation of a domain parellel function.

.. note::
    ``ShardTensor``, as implemented in ``torch``, follows the execution model of ``torch``: in general, no knowledge of previous or subsequent operations is assumed at each layer.  So, while there are certainly more optimized ways to support *specific* models (imagine 2 convolutions back to back, for example: you could perform a halo exchange just once if you sized it properly) in general we have traded absolute peak performance for the ability to support a flexible set of layers and models with minimal to no user-space operations.  So we do the halo exchange twice in two back-to-back convolutions, but the benefit is an increase in usability and flexibility.  When the data size is large, the overhead is small in comparison.



Supporting Your Model
---------------------

``ShardTensor``, like ``DTensor`` upstream in PyTorch, is designed to drop in and replace ``torch.Tensor`` objects.  As such, you rarely have to modify your model code directly to have multiple execution paths for distributed vs. single-device tensors.  Instead, ensure support for all the torch functions in your model and let PyTorch's dispatch techniques route everything appropriately.


Going Backwards
---------------

One thing we haven't covered in this tutorial is the backwards pass of sharded operations.  There are two components to this:

1. When you call ``backward()`` on a ``ShardTensor`` object ... what happens?  We have designed ``ShardTensor`` to smoothly handle the most common cases: you compute a loss via a reduction (``outputs.mean()``) and then call backward on the output of the reduction.  ``ShardTensor`` will then ensure the loss moves backwards correctly through the reduction and the gradients are also sharded - just like their inputs.
2. When you have implemented a custom operation and registered it with ``ShardTensor.register_function_handler``, what do the gradients do? If you use the ``to_local`` and ``from_local`` operations on ``ShardTensor`` objects, which are differentiable, and the in-between operations are also differentiable, it will work correctly.  Everything between ``to_local`` and ``from_local`` will use standard autograd operations from upstream PyTorch.  If you need something more complex (like our ring computation, above), you can implement a custom autograd layer in PyTorch that performs the collectives directly.  See the excellent PyTorch documentation on `Defining new autograd functions <https://docs.pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html>`_ for many more details.


Summary
-------


Up to here, we've seen a couple examples of distributed computation with ``ShardTensor``.  Let's recap:

- ``ShardTensor`` is built on top of ``DTensor``, enabling it to fall back to ``DTensor`` computations whenever it doesn't have a custom implementation.  In nearly all simple operations, this is functional.
- When necessary, ``ShardTensor`` can have a dedicated execution path for sharded operations via ``ShardTensor.register_function_handler(target, handler)``.  This technique will route calls to ``target`` to ``handler`` when ``target`` is called on ``ShardTensor`` objects, as long as the function is a torch function.
- Not every function you want to use, of course, is part of ``torch``.  In this case, you can use PyTorch's overrides system to inform torch about the function and then route calls appropriately.
- Even though many operations are functional out of the box from ``DTensor``, it does not mean they are efficient.  ``DTensor`` is optimized for Large Language Model applications.  In ``PhysicsNeMo``, we are providing a number of efficient distributed operations for common scientific AI needs - and if want you need isn't supported, feel free to reach out on `GitHub <https://github.com/NVIDIA/physicsnemo/tree/main>`_ for support!

``ShardTensor`` is still under active development, and we're working to add more model support.  To see how to use it with an end-to-end training example, see the :ref:`fsdp_and_shard_tensor` tutorial.  In particular, ``ShardTensor`` is fully compatible with ``torch.distributed.fsdp.FullyShardedDataParallel`` enabling you to even deploy multiple levels of parallelism: domain parallelism + batch parallelism (+ model parallelism, if needed!).

In general, ``ShardTensor`` is meant to be a seamless, nearly drop in replacement to ``torch.Tensor`` that will parallelize your model - see :ref:`fsdp_and_shard_tensor` for more info.

``ShardTensor`` is especially useful when memory constraints limit the ability to run a model during training or inference on a single GPU.  See :ref:`domain_parallelism.rst` for more discussion of this topic.  With extra computation and bookkeeping needed, we can never expect ``ShardTensor`` to outperform single-device computations when run on very small data and very small models.  However, as the data grows, the extra overhead becomes a very small portion of the computational cost.  And, datasets that don't fit into memory even with batch size 1 can be enabled.



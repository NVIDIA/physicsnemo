Implementing new layers for `ShardTensor`
=========================================

This tutorial is a walkthrough of how to extend domain parallel functionality via `ShardTensor`.  We'll first discuss at a high level some parallelism techniques, and then look at exactly how to implement a domain parallel layer with a few examples.  For some background on what `ShardTensor` is and when to use it, check out the tutorial :ref:`domain_parallelism.rst`.

When is extending `ShardTensor` needed?
---------------------------------------

`ShardTensor` is designed to support domain-parallel operations, or operations that can be performed on a tensor that resides across multiple devices. Many operations are supported already - out of the box - by the upstream ``DTensor`` class that ``ShardTensor`` inherits from.  Some operations - many convolutions, interpolations, poolings, normalizations, and attention - are supported through ``PhysicsNeMo``.  In this tutorial, we'll look at a few increasingly-complicated situations and see how ``ShardTensor`` handles them - or doesn't - and how to fix cases that aren't supported or aren't performant.


Example 0: Vector Addition
--------------------------

As a basic example (and note that this is a built-in operation from ``DTensor``), let's implement a shard tensor version of ``torch.add()``.  Here's a single-device implementation:

.. code-block:: python
    :caption: Example 0: Vector Addition, single device

    import torch
    import time

    # Make a really big tensor:
    N = 1_000_000_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)

    def f(a, b):
        # This is a truly local operation: no communication is needed.
        return a + b

    # run a couple times to warmup:
    for i in range(5):
        c = f(a,b)

    # Optional: Benchmark it if you like:

    # Measure execution time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(10):
        c = f(a,b)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")


To perform this with ShardTensor, we first need to convert these tensors to ``ShardTensor`` objects.  The easiest way to do this is with the ``scatter_tensor`` method:

.. code-block:: python
    :caption: Example 0: Vector Addition, distributed computation

    import torch
    import time

    from physicsnemo.distributed import DistributedManager, scatter_tensor
    from torch.distributed.tensor.placement_types import Shard

    # Another really big tensor:
    N = 1_000_000_000

    DistributedManager.initialize()
    dm = DistributedManager()

    device = dm.device

    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)

    def f(x, y):
        return x + y

    # Get the baseline result
    c_baseline = f(a,b)

    mesh = dm.initialize_mesh([-1,], ["domain"])
    placements = (Shard(0),)
    a_sharded = scatter_tensor(a, 0, mesh, placements)
    b_sharded = scatter_tensor(b, 0, mesh, placements)

    c_sharded = f(a_sharded,b_sharded)

    # Comparison requires that we coalesce the results:
    c_sharded = c_sharded.full_tensor()

    # Now, performance measurement:
    # Warm up:
    for i in range(5):
        c = f(a_sharded,b_sharded)

    # Measure execution time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(10):
        c = f(a_sharded,b_sharded)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    if dm.rank == 0:
        print(f"Rank {dm.rank}, Tensor agreement? {torch.allclose(c_baseline, c_sharded)}")
        print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")

This will run, out of the box (and in fact doesn't even need ``ShardTensor``, ``DTensor`` implements distributed vector addition).  If you have a multiple GPU system, execute the code with a command like ``torchrun --nproc-per-node 8 example_0_sharded.py``.  You ought to see pretty good scaling efficiency - with no communication overhead, the distributed operation can work at approximately weak scaling speeds.  For small tensors, though, where the addition operation is bound by launch latency: you will see a slightly higher overhead with distributed operations because there is slightly more organization and bookkeeping required.
    

Example 1: Vector Dot Product
-----------------------------

Let's look now at a slightly more complicated example: the dot product of two vectors.  In this case, because the output is a single scalar, we'll find that there _is_ communication required and see how to implement that seamlessly with ``ShardTensor``.

Here's the single-device implementation.  Note that the **only** difference here is in the definition of ``f``:

.. code-block:: python

    def f(x, y):
        return torch.dot(x, y)

For reference, here's the full code:

.. code-block:: python
    :caption: Example 1: Vector Dot Product, single device

    import torch
    import time

    # Make a really big tensor:
    N = 1_000_000_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)

    def f(a, b):
        # This is a truly local operation: no communication is needed.
        return torch.dot(a, b)

    # run a couple times to warmup:
    for i in range(5):
        c = f(a,b)

    # Optional: Benchmark it if you like:

    # Measure execution time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(10):
        c = f(a,b)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")

If we make the same changes to the distributed version, we get an error when we run it:

.. code-block:: bash

    NotImplementedError: Operator aten.dot.default does not have a sharding strategy registered.

This is a good time to talk about how pytorch decides what to do each time it's called for an operation on ``torch.Tensor(s)``, which will lead into how we fix this error.

TODO -- THIS SECTION
.. Pytorch, as you likely already know, implements operations on multiple backends and with multiple
TODO -- THIS SECTION


With all that in mind, let's add a handler for ``torch.dot`` that works on physicsnemo's ``ShardTensor``:

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
        
        # IT'S usually good to ensure the tensor placements work:
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

.. code-block:: python
    :caption: Example 1: Vector Dot Product, distributed computation

    import torch
    import torch.distributed as dist
    import time

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
        
        # IT'S usually good to ensure the tensor placements work:
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

    # Register the implementation with ShardTensor's function dispatch:
    ShardTensor.register_function_handler(torch.dot, sharded_dot_product)


    # Another really big tensor:
    N = 1_000_000_000

    DistributedManager.initialize()
    dm = DistributedManager()

    device = dm.device

    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)

    def f(x, y):
        return torch.dot(x , y)

    # Get the baseline result
    c_baseline = f(a,b)

    mesh = dm.initialize_mesh([-1,], ["domain"])
    placements = (Shard(0),)
    a_sharded = scatter_tensor(a, 0, mesh, placements)
    b_sharded = scatter_tensor(b, 0, mesh, placements)


    c_sharded = f(a_sharded,b_sharded)

    # Comparison requires that we coalesce the results:
    c_sharded = c_sharded.full_tensor()


    # Now, performance measurement:

    # Warm up:
    for i in range(5):
        c = f(a_sharded,b_sharded)

    # Measure execution time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(10):
        c = f(a_sharded,b_sharded)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    if dm.rank == 0:
        print(f"Rank {dm.rank}, Tensor agreement? {torch.allclose(c_baseline, c_sharded)}")
        print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")

You should now be able to run this code with ``torchrun --nproc-per-node 8 example_1_sharded.py``.  You should see a significant nearly linear scaling efficiency over NVLink-connected devices.

Example 2: Nearest Neighbors 
----------------------------

With some basics out of the way, let's look at something a little more interesting and useful.  In many scientific AI workloads, we need to do a query of the nearest neighbors of a point cloud to build a GNN.  Pytorch doesn't really have an efficient implementation of a nearest neighbor operation - let's write one here (poorly!) just to show how it can be parallelized.

.. note:: 
    There are better ways to write a kNN to operate on pytorch tensors - don't use this in production code!  Most times when you need the nearest neighbors of a point cloud, some sort of Tree or hash mapping structure is significantly more efficient.  We're not using that in this tutorial for clarity, but when we need these operations in ``physicsnemo`` we use optimized implementations backend by libraries like ``cuml`` and ``warp``. TODO - add links to these libraries.

.. code-block:: python
    :caption: Example 2: Nearest Neighbors, single device

    import torch
    import time

    # This time, let's make two moderately large tensors since we'll have to, at least briefly,
    # construct a tensor of their point-by-point difference.
    N1 = 234_567
    N2 = 12_345
    num_neighbors = 17


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We'll make these 3D tensors to represent 3D points
    a = torch.randn(N1, 3, device=device)
    b = torch.randn(N2, 3, device=device)

    def knn(x, y, n):
        # Return the n nearest neighbors in x for each point in y.
        # Returns the 
        
        # First, compute the pairwise difference between all points in x and y.
        displacement_vec = x[None, :, :] - y[:, None, :]
        
        # Use the norm to compute the distance:
        distance = torch.norm(displacement_vec, dim=2)
        
        distances, indices = torch.topk(distance, k=n, dim=1, largest=False)
        
        x_results = x[indices]
        # distance = distances[indices]
        
        return x_results, distances

    y_neighbors_to_x, neighbor_disances = knn(a,b, num_neighbors)
    print(y_neighbors_to_x.shape) # should be (N2, num_neighbors, 3)
    print(neighbor_disances.shape) # should be (N2, num_neighbors)

    # run a couple times to warmup:
    for i in range(5):
        _ = knn(a,b, num_neighbors)

    # Optional: Benchmark it if you like:

    # Measure execution time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(10):
        _ = knn(a,b, num_neighbors)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")

If you run this (``python example_2_baseline.py``), you'll see that it's not quite as quick as the other examples - it's also really memory intensive.  At the time of this tutorial's publication, we saw about 1.544 seconds for 10 runs on a single A100, or 150ms per call.  Additionally, this line will allocate memory of N1 * N2 * 3 * 4 bytes (=32 GB in fp32!):

.. code-block:: python

    displacement_vec = x[None, :, :] - y[:, None, :]

So, as written you can't really scale this up larger on a single device.  There are, as noted, better ways to do a kNN but let's parallelize this one since it's a good way to learn how you might parallelize custom functions.  In fact, a functional paralellization couldn't be easier - do nothing but cast the inputs to ``ShardTensor`` and let the existing operations take care of it.  The underlying implementation of ``ShardTensor`` and ``DTensor`` enables this out of the box:

.. code-block:: python
    :caption: Example 2: Nearest Neighbors, distributed computation, basic functionality

    import torch
    import torch.distributed as dist
    import time

    from physicsnemo.distributed import DistributedManager, scatter_tensor, ShardTensor
    from torch.distributed.tensor.placement_types import Shard, Replicate

    from physicsnemo.distributed.shard_utils.ring import perform_ring_iteration, RingPassingConfig

    # This time, let's make two moderately large tensors since we'll have to, at least briefly,
    # construct a tensor of their point-by-point difference.
    N1 = 234_567
    N2 = 12_345
    num_neighbors = 17

    DistributedManager.initialize()
    dm = DistributedManager()

    # We'll make these 3D tensors to represent 3D points
    a = torch.randn(N1, 3, device=dm.device)
    b = torch.randn(N2, 3, device=dm.device)

    def knn(x, y, n):
        # Return the n nearest neighbors in x for each point in y.
        
        # First, compute the pairwise difference between all points in x and y.
        displacement_vec = x[None, :, :] - y[:, None, :]
        
        # Use the norm to compute the distance:
        distance = torch.norm(displacement_vec, dim=2)
        
        distances, indices = torch.topk(distance, k=n, dim=1, largest=False)
        
        x_results = x[indices]
        
        return x_results, distances

    # Get the baseline result
    y_neighbors_to_x, neighbor_distances = knn(a, b, num_neighbors)

    if dm.rank == 0:
        print(y_neighbors_to_x.shape)    # should be (N2, num_neighbors, 3)
        print(neighbor_distances.shape)  # should be (N2, num_neighbors)

    mesh = dm.initialize_mesh([-1,], ["domain"])
    placements = (Shard(0),)
    a_sharded = scatter_tensor(a, 0, mesh, placements)
    b_sharded = scatter_tensor(b, 0, mesh, placements)

    # Get the sharded result
    y_neighbors_to_x_sharded, neighbor_distances_sharded = knn(a_sharded, b_sharded, num_neighbors)

    # Check for agreement:
    y_neighbors_to_x_sharded = y_neighbors_to_x_sharded.full_tensor()
    neighbor_distances_sharded = neighbor_distances_sharded.full_tensor()

    if dm.rank == 0:
        # Note - do the ``full_tensor`` call outside this if-block or it will hang!
        print(f"Neighbors agreement? {torch.allclose(y_neighbors_to_x, y_neighbors_to_x_sharded)}")
        print(f"Distances agreement? {torch.allclose(neighbor_distances, neighbor_distances_sharded)}")

    # run a couple times to warmup:
    for i in range(5):
        _ = knn(a_sharded, b_sharded, num_neighbors)

    # Optional: Benchmark it if you like:
    # Measure execution time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(10):
        _ = knn(a_sharded, b_sharded, num_neighbors)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    if dm.rank == 0:
        print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")

Go ahead and pause, here, and run these codes either with ``torchrun --nproc-per-node 8 example_2_sharded.py``.  You should see a significant speedup - we saw about 33 ms per call on a single A100.  Compared to 150ms, that's a nice improvement, about 4.5x faster ... but we're also using 8 GPUs.  Why isn't it 8x faster?

The issue, here, is once again in this line:

.. code-block:: python

    displacement_vec = x[None, :, :] - y[:, None, :]

Except this time, the ``x`` and ``y`` tensors are being subtracted when their sharded axes disagree.  ``x[None,:,:]`` will shift the placement of the shards of ``x`` from ``Shard(0)`` to ``Shard(1)``, while ``y[:,None,:]`` will not shift the shards of ``y`` from ``Shard(0)``.  When ``DTensor`` does the subtraction, it makes the decision to replicate one of these tensors (the first one, here), and leaves that axis replicated in the output.  Like this:


.. code-block:: python

    import torch
    import torch.distributed as dist
    import time

    from physicsnemo.distributed import DistributedManager, scatter_tensor, ShardTensor
    from torch.distributed.tensor.placement_types import Shard, Replicate

    from physicsnemo.distributed.shard_utils.ring import perform_ring_iteration, RingPassingConfig

    # This time, let's make two moderately large tensors since we'll have to, at least briefly,
    # construct a tensor of their point-by-point difference.
    N1 = 234_567
    N2 = 12_345
    num_neighbors = 17

    DistributedManager.initialize()
    dm = DistributedManager()

    # We'll make these 3D tensors to represent 3D points
    a = torch.randn(N1, 3, device=dm.device)
    b = torch.randn(N2, 3, device=dm.device)

    mesh = dm.initialize_mesh([-1,], ["domain"])
    placements = (Shard(0),)
    a_sharded = scatter_tensor(a, 0, mesh, placements)
    b_sharded = scatter_tensor(b, 0, mesh, placements)

    if dm.rank == 0:
        print(f"a_sharded shape and placement: {a_sharded.shape}, {a_sharded.placements}")
        print(f"b_sharded shape and placement: {b_sharded.shape}, {b_sharded.placements}")
    a_sharded = a_sharded[None, :, :]
    b_sharded = b_sharded[:, None, :]

    if dm.rank == 0:
        print(f"a_sharded shape and placement: {a_sharded.shape}, {a_sharded.placements}")
        print(f"b_sharded shape and placement: {b_sharded.shape}, {b_sharded.placements}")

    distance_vec = a_sharded - b_sharded
    if dm.rank == 0:
        print(f"distance_vec shape and placement: {distance_vec.shape}, {distance_vec.placements}")

You'll see this output:

.. code-block:: text

    a_sharded shape and placement: torch.Size([234567, 3]), (Shard(dim=0),)
    b_sharded shape and placement: torch.Size([12345, 3]), (Shard(dim=0),)
    a_sharded shape and placement: torch.Size([1, 234567, 3]), (Shard(dim=1),)
    b_sharded shape and placement: torch.Size([12345, 1, 3]), (Shard(dim=0),)
    distance_vec shape and placement: torch.Size([12345, 234567, 3]), (Shard(dim=1),)

It's nice, of course, that ``DTensor`` will get this correct out of the box - but it's not the most efficient way we could do something like this.  Instead, we can write the ``knn`` function to use a ring-based computation: compute the knn on local chunks, and then shift the slices of the point cloud along the mesh to compute the next iteration.  It requires more collectives, but because we can overlap the communication and computation - and never have to construct the entire distance matrix - it's more efficient.

.. code-block:: python
    :caption: Example 2: Nearest Neighbors, distributed computation, ring-based computation

    import torch
    import torch.distributed as dist
    from torch.overrides import handle_torch_function, has_torch_function
    import time

    from physicsnemo.distributed import DistributedManager, scatter_tensor, ShardTensor
    from torch.distributed.tensor.placement_types import Shard, Replicate

    from physicsnemo.distributed.shard_utils.ring import perform_ring_iteration, RingPassingConfig

    # This time, let's make two moderately large tensors since we'll have to, at least briefly,
    # construct a tensor of their point-by-point difference.
    N1 = 234_567
    N2 = 12_345
    num_neighbors = 17

    DistributedManager.initialize()
    dm = DistributedManager()

    device = dm.device



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We'll make these 3D tensors to represent 3D points
    a = torch.randn(N1, 3, device=device)
    b = torch.randn(N2, 3, device=device)

    def knn(x, y, n):
        # This is to enable torch to track this knn function and route it correctly in ShardTensor:
        if has_torch_function((x, y)):
            return handle_torch_function(
                knn, (x, y), x, y, n
            )

        # Return the n nearest neighbors in x for each point in y.
        
        # First, compute the pairwise difference between all points in x and y.
        displacement_vec = x[None, :, :] - y[:, None, :]
        
        # Use the norm to compute the distance:
        distance = torch.norm(displacement_vec, dim=2)
        
        distances, indices = torch.topk(distance, k=n, dim=1, largest=False)

        x_results = x[indices]
        
        return x_results, distances

    # Get the baseline result
    y_neighbors_to_x, neighbor_disances = knn(a,b, num_neighbors)

    if dm.rank == 0:

        print(y_neighbors_to_x.shape) # should be (N2, num_neighbors, 3)
        print(neighbor_disances.shape) # should be (N2, num_neighbors)

    mesh = dm.initialize_mesh([-1,], ["domain"])
    placements = (Shard(0),)
    a_sharded = scatter_tensor(a, 0, mesh, placements)
    b_sharded = scatter_tensor(b, 0, mesh, placements)



    def knn_ring(func, types, args, kwargs):
        # Wrapper to intercept knn and compute it in a ring.
        # Never fully realizes the distance product.
        
        def extract_args(x, y, n, *args, **kwargs):
            return x, y, n
        x, y, n = extract_args(*args, **kwargs)

        
        # Each tensor has a _spec attribute, which contains information about the tensor's placement
        # and the devices it lives on:
        x_spec = x._spec
        y_spec = y._spec

        # ** In general ** you want to do some checking on the placements, since each 
        # point cloud might be sharded differently.  By construction, I know they're both 
        # sharded along the points axis here (and not, say, replicated).

        if not x_spec.mesh == y_spec.mesh:
            raise NotImplementedError("Tensors must be sharded on the same mesh")
        
        mesh = x_spec.mesh
        local_group = mesh.get_group(0)
        local_size = dist.get_world_size(group=local_group)
        mesh_rank = mesh.get_local_rank()

        # x and y are both sharded - and since we're returning the nearest
        # neighbors to x, let's make sure the output keeps that sharding too.
        
        # One memory-efficient way to do this is with with a ring computation.
        # We'll compute the knn on the local tensors, get the distances and outputs, 
        # then shuffle the y shards along the mesh.
        
        # we'll need to sort the results and make sure we have just the top-k, 
        # which is a little extra computation.
        
        # Physics nemo has a ring passing utility we can use.
        ring_config = RingPassingConfig(
            mesh_dim = 0,
            mesh_size = local_size,
            ring_direction = "forward",
            communication_method = "p2p"
        )
        
        local_x, local_y = x.to_local(), y.to_local()
        current_dists = None
        current_topk_y = None
        
        x_sharding_shapes = x._spec.sharding_shapes()[0]


        for i in range(local_size):
            source_rank = (mesh_rank - i) % local_size
            
            # For point clouds, we need to pass the size of the incoming shard.
            next_source_rank = (source_rank - 1) % local_size
            recv_shape = x_sharding_shapes[next_source_rank]
            if i != local_size - 1:
                # Don't do a ring on the last iteration.
                next_local_x = perform_ring_iteration(
                    local_x,
                    mesh,
                    ring_config,
                    recv_shape=recv_shape,
                )

            # Compute the knn on the local tensors:
            local_x_results, local_distances = func(local_x, local_y, n)


            if current_dists is None:
                current_dists = local_distances
                current_topk_y = local_x_results
            else:
                # Combine with the topk so far:
                current_dists = torch.cat([current_dists, local_distances], dim=1)
                current_topk_y = torch.cat([current_topk_y, local_x_results], dim=1)
                # And take the topk again:
                current_dists, running_indexes = torch.topk(current_dists, k=n, dim=1, largest=False)

                # This creates proper indexing to select specific elements along dim 1
                current_topk_y = torch.gather(current_topk_y, 1, 
                                            running_indexes.unsqueeze(-1).expand(-1, -1, 3))



            if i != local_size - 1:
                # Don't do a ring on the last iteration.
                local_x = next_local_x
                
        # Finally, return the outputs as ShardTensors.
        topk_y = ShardTensor.from_local(
            current_topk_y, 
            device_mesh = mesh,
            placements = y._spec.placements,
            sharding_shapes = y._spec.sharding_shapes(),
        )
        
        distances = ShardTensor.from_local(
            current_dists, 
            device_mesh = mesh,
            placements = y._spec.placements,
            sharding_shapes = y._spec.sharding_shapes(),
        )
        
        return topk_y, distances


    ShardTensor.register_function_handler(knn, knn_ring)

    # Get the sharded result
    y_neighbors_to_x_sharded, neighbor_disances_sharded = knn(a_sharded,b_sharded, num_neighbors)

    # Check for agreement:
    y_neighbors_to_x_sharded = y_neighbors_to_x_sharded.full_tensor()
    neighbor_disances_sharded = neighbor_disances_sharded.full_tensor()

    if dm.rank == 0:
        print(f"Neighbors agreement? {torch.allclose(y_neighbors_to_x, y_neighbors_to_x_sharded)}")
        print(f"Distances agreement? {torch.allclose(neighbor_disances, neighbor_disances_sharded)}")
        

    # run a couple times to warmup:
    for i in range(5):
        _ = knn(a_sharded,b_sharded, num_neighbors)
    # Optional: Benchmark it if you like:

    # Measure execution time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(10):
        _ = knn(a_sharded,b_sharded, num_neighbors)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    if dm.rank == 0:
        print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")

Run this (``torchrun --nproc-per-node 8 example_2_sharded.py``), and you'll see the time per iteration is more like 20.7ms.  That's an 8x speed up over the original, single device implementation - much better!

.. note:: 
        There is an important piece of that previous example, in case you overlooked it.  The ``knn`` function has a few extra lines registering it with pytorch's overrides system (`torch.overrides <https://docs.pytorch.org/docs/stable/torch.overrides.html>`_). This step let's pytorch track the ``knn`` function, and registering it with ``ShardTensor`` sends execution to the ``knn_ring`` function instead.  When that function in turn calls the ``knn`` function on standard ``torch.Tensor`` objects, it is executed normally on the local objects.

What collectives do I need for my operation?
--------------------------------------------

If you're looking to extend ``ShardTensor`` to support a new domain parallelism operation, it can fall into one of several - not exhaustive - categories.  Use this to guide your thinking about performant domain-parallel implementations.

- **Fully Local** operations can be computed locally at every value of a tensor, with a one-to-one mapping between input and output tensors.  Activations are an obvious example of this, but tensor-wise math can be too: ``c = a + b``, where ``a`` and ``b`` are both tensors, can follow this patten (absent reshaping/broadcasting, as we saw above, which complicates things).  In these cases, the "domain parallel" component of an operation is really just a purely local operation + making sure the output tensor ``c`` is represented properly as a distributed object.  No communication is needed.

- **Semi-Local** operations depend on neighboring values, but not on _every_ value.  Depending on the details of operation, and the pattern of distributing a tensor across devices, to correctly perform this operation some information at the edges of each local tensor may need to be exchanged.  One example of this is `convolution` operations, where information must be exchanged across the domain decomposition boundary for most cases.  A more complicated example is a distributed graph, where some graph nodes share an edge that spans a domain boundary.  In many cases, this type of information exchange across a boundary is referred to as a ``halo``.  As long as the halo is small compared to the computation, these operations scale well through domain decomposition.

- **Reduction** based operations that require a large scale reduction of data, such as ``sum`` but also normalization layers, can usually be implemented in two or less passes: first, compute local reductions on the local piece of a tensors, and ``allreduce`` it across the domain.  Then, update the output by applying a correction factor based on the local to global statistics calculated.  

- **Global**  operations require a global view of tensors to compute correctly: each output point depends on information from all possible input locations.  Two examples that appear very different but are both global for domain decomposition are _Attention_ mechanisms, and distance-based queries on point clouds such as the _kNN_ we implemented earlier.  In both cases, one particular value of output can depend on any or all values of input tensors.  There are multiple ways to proceed for these operations, but in this case a ``ring`` collective can be quite efficient: tensors will perform the computation on local chunks, and then part of the input (KV, for attention, or part of the point cloud) will be passed to the next rank in a ring topology.  With overlapping communication and computation, these algorithms can acheive excellent scaling properties.  A challenge may be that the outputs of each iteration of the ring may need to be combined in non-intuitive ways.  RingAttention (TODO - citation), which inspired the implementation in ``PhysicsNeMo``, necessitates log- and sign-based accumulation of outputs.  The ``kNN`` layer has two ``topk`` calls per iteration - one for the real operation, and one to combine the output.

It isn't necessarily true that all operations might fall in to these categories for domain parallelism.  However, thinking about the way input data is decomposed, how output data must be decomposed, and what communication patterns are needed will often be enough to guide you to a correct, efficient implementation of a domain parellel function.

Supporting Your Model
---------------------

``ShardTensor``, like ``DTensor`` upstream in pytorch, is designed to drop in and replace ``torch.Tensor`` objects.  As such, you rarely have to modify your model code directly to have multiple execution paths for distributed vs. single-device tensors.  Instead, ensure support for all the torch functions in your model and let pytorch's dispatch techniques route everything appropriately.


Going Backwards
---------------

One thing we haven't covered in this tutorial is the backwards pass of sharded operations.  There are two components to this:

1. When you call ``backward()`` on a ``ShardTensor`` object ... what happens?  We have designed ``ShardTensor`` to smoothly handle the most common cases: you compute a loss via a reduction (``outputs.mean()``) and then call backward on the output of the reduction.  ``ShardTensor`` will then ensure the loss moves backwards correctly through the reduction and the gradients are also sharded - just like their inputs.
2. When you have implemented a custom operation and registered it with ``ShardTensor.register_function_handler``, what do the gradients do? If you use the ``to_local`` and ``from_local`` operations on ``ShardTensor`` objects, which are differentiable, and the in-between operations are also differentiable, it will work correctly.  Everything between ``to_local`` and ``from_local`` will use standard autograd operations from upstream PyTorch.  If you need something more complex (like our ring computation, above), you can implement a custom autograd layer in pytorch that performs the collectives directly.  See the excellent pytorch documentation on `defining new autograd functions <https://docs.pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html>`_ for many more details.


Summary
-------


Up to here, we've seen a couple examples of distributed computation with ``ShardTensor``.  Let's recap:

- ``ShardTensor`` is built on top of ``DTensor``, enabling it to fall back to ``DTensor`` computations whenever it doesn't have a custom implementation.  In nearly all simple operations, this is functional.
- When necessary, ``ShardTensor`` can have a dedicated execution path for sharded operations via ``ShardTensor.register_function_handler(target, handler)``.  This technique will route calls to ``target`` to ``handler`` when ``target`` is called on ``ShardTensor`` objects, as long as the function is a torch function.
- Not every function you want to use, of course, is part of ``torch``.  In this case, you can use pytorch's overrides system to inform torch about the function and then route calls appropriately.
- Even though many operations are functional out of the box from ``DTensor``, it does not mean they are efficient.  ``DTensor`` is optimized for Large Langauge Model applications.  In ``PhysicsNeMo``, we are providing a number of efficient distributed operations for common scientific AI needs - and if want you need isn't supported, feel free to reach out on github for support!

``ShardTensor`` is still under active development, and we're working to add more model support.  To see how to use it with an end-to-end training example, see the :ref:`fsdp_and_shard_tensor` tutorial.  In particular, ``ShardTensor`` is fully compatible with ``torch.distributed.fsdp.FullyShardedDataParallel`` enabling you to even deploy multiple levels of parallelism: domain parallelism + batch parallelism (+ model parallelism, if needed!).

In general, ``ShardTensor`` is meant to be a seamless, nearly drop in replacement to ``torch.Tensor`` that will parallelize your model - see :ref:`fsdp_and_shard_tensor` for more info.

``ShardTensor`` is especially useful when memory constraints limit the ability to run a model during training or inference on a single GPU.  See :ref:`domain_parallelism.rst` for more discussion of this topic.  With extra computation and bookkeeping needed, we can never expect ``ShardTensor`` to outperform single-device computations when run on very small data and very small models.  However, as the data grows, the extra overhead becomes a very small portion of the computational cost.  And, datasets that don't fit into memory even with batch size 1 can be enabled.



Torch Compile and External Kernels
==================================

**The Challenge: Getting the Best of Both Worlds**

Scientific AI applications often face a performance dilemma: ``torch.compile`` can accelerate PyTorch models, but many scientific workloads require specialized libraries outside the PyTorch ecosystem—like RAPIDS cuML for accelerated k-nearest neighbors, NVIDIA Warp for differentiable physics simulation, or other domain-specific kernels. Using these external libraries typically causes "graph breaks" in ``torch.compile``, limiting the potential performance benefits.

.. note::
    PyTorch deployed ``torch.compile`` in version 2.0 - older versions of pytorch will not be compatible with this tutorial.

This tutorial demonstrates how to solve this challenge by integrating external kernels with ``torch.compile`` using PyTorch's custom operator API, enabling you to leverage both PyTorch's graph optimization and high-performance external libraries simultaneously.

**What You'll Learn**

By the end of this tutorial, you'll understand how to:

- Register external library functions as PyTorch custom operators
- Enable ``torch.compile`` to work seamlessly with libraries like cuML and Warp
- Implement custom backward passes for external kernels
- Share memory pools between PyTorch and RAPIDS for additional performance gains
- Achieve significant speedups (10x+ in our toy examples) without sacrificing ``torch.compile`` benefits

**Who is this for?**

This is a more advanced tutorial, for AI developers who are actively working on new models, applications, data pipelines, etc.  Strong familiarity with ``torch`` is a prerequisite, and basic familiarity with unstructed data operations (like k-Nearest Neighbors) is good to have.  And, you should know the `basics <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial_.html>`_ of how ``torch.compile`` works and how to use it on your code.



**Table of Contents**

.. contents::
   :local:
   :depth: 3

What does ``torch.compile`` do?
-------------------------------

If you're interested in ``torch.compile``, you've probably already found the `tutorial from PyTorch <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial_.html>`_.  At a high level, ``torch.compile`` is a tool that allows pytorch to inspect your model ahead of time, find places where kernels can be optimized or combined, and enable those optimizations.  The performance gain is heavily dependent on the application: kernel fusion (like a convolution + activation) can help reduce runtime by mitigating the memory-bound characteristics of one kernel when fusing it to compute bound kernel.  Further, performance gains are highly dependent on compute precision as well: the thresholds for what is "compute-bound" and what is "memory-bound" are different depending on the precision. Lower precisions can take advantage of smaller memory footprints (so less bandwidth is necessary from memory) as well as dedicated processing units like Tensor cores for faster math operations.

With all of that in mind - that tutorial is focused on pure PyTorch functionality.  In PhysicsNeMo workloads, however, we often need to leverage tools that live outside the pytorch ecosystem.  But with large, complex, and end-to-end models, we still want to take advantage of the performance benefits we can get with `torch.compile`.  So in the rest of this tutorial, we'll look at exactly how to solve that problem.  

This tutorial is broken into two models: first, we'll work on a k-Nearest-Neighbors type problem, which we can accelerate with ``cuml``.  Second, we'll do a closer examination of the backwards-pass functionality in ``torch.compile`` (and you'll learn why it wasn't necessary in the first example, even though we're doing training!)

Introducing the Application
---------------------------

For demonstration purposes, we've invented a small operator that works on point-cloud like data.  This means the input to the operator is a , 2D ``torch.Tensor`` of 3D points, unstructured and unordered.  You can find small scale point-cloud data, for example, in "ModelNet-10 - Princeton 3D Object Dataset" (`link <https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset>`_) - but the exact features of the data aren't the focus here in this tutorial, so don't worry about the details of the input/outputs.  

Regarding the application - for PhysicsNeMo users, you'll recognize similar ideas in architectures such as DoMINO and FigConvNet, and local aggregation of points is a well studied topic in many graph neural networks on point clouds.  We're not specifically using these models, however, this is a fully independent example application.

We start with a simple, 3-layer MLP:

.. code-block:: python

    class MLP(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            A simple 3 layer MLP that takes in a tensor of 
            shape (N, input_dim) and outputs a tensor of 
            shape (N, output_dim)
            """
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            

This MLP is used twice, in a simple model:

.. code-block:: python

    class kNN_Projector(torch.nn.Module):
        def __init__(self, k: int, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.proj = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
            self.k = k
            
            self.proj_out = MLP(input_dim=hidden_dim, hidden_dim = hidden_dim, output_dim = output_dim)
            
        def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
            """
            Accept two point clouds, p1 and p2.  Compute a learnable projection onto p2 to 
            learn features.  Then, use a kNN-weighted aggregation to project those features
            onto p1.
            """
            p2_features = self.proj(p2)
            
            p1_features = knn_weighted_feature_aggregation(p1, p2, p2_features, k=self.k)

            return self.proj_out(p1_features)
    
In basic terms, this model operates on two sets of point clouds.  A reference set of points, ``p2``, has some features learned on it by the first MLP.  Then, using the k nearest neighbors to each point in ``p1``, the features in ``p2`` are projected (``knn_weighted_feature_aggregation``) onto the locations in ``p1``.  Finally, the output features from the aggregation are projected to a final latent space via a second MLP.  The details of the projection look like this:

.. code-block:: python

    def knn_weighted_feature_aggregation(
        p1: torch.Tensor, 
        p2: torch.Tensor,
        p2_features: torch.Tensor, 
        k: int = 3, 
        sigma: float = 0.1,
        eps: float = 1e-8
        ) -> torch.Tensor:
        """
        Perform differentiable kNN-weighted feature aggregation.

        Args:
            p1 (torch.Tensor): Query points, shape (B, M, D)
            p2 (torch.Tensor): Reference points, shape (B, N, D)
            p2_features (torch.Tensor): Features at reference points, shape (B, N, D_feat)
            k (int): Number of neighbors
            sigma (float): RBF temperature parameter
            eps (float): Numerical stability for normalization

        Returns:
            torch.Tensor: Aggregated features at p1, shape (B, M, D_feat)
        """
        # M, D = p1.shape
        # N, D_feat = p2_features.shape

        # Compute pairwise distances: (M, N)
        dists = torch.norm(p1[:,None,:] - p2[None,:,:], dim=-1)


        # Find top-k nearest neighbors
        topk_dists, topk_idx = torch.topk(dists, k=k, dim=1, largest=False)

        # Gather neighbor features: (M, k, D_feat)
        neighbors = p2_features[topk_idx]

        # Compute weights: (M, k)
        weights = torch.softmax(-topk_dists / sigma, dim=1)

        # Weighted sum of neighbor features: (M, D_feat)
        agg = torch.sum(weights.unsqueeze(-1) * neighbors, dim=1)

        return agg

You make recognize this as a brute-force implementation of a kNN, followed by a weight calculation based on how far apart two points are.  

.. seealso::

    Don't read into the high level algorithm too closely!  Remember, we're here in this tutorial to talk about computational performance.  This is just a made-up example that uses a kNN.

Just for completeness, so you can run this example on your own, here are some helper functions needed to initialize the data and, optionally, ensure deterministic inputs:

.. code-block:: python

    def generate_data(N_points_to_search, grid_points, target_features, dtype=torch.bfloat16):
        device = torch.device("cuda")
        
        
        # Make a random point cloud:
        point_cloud = torch.randn(N_points_to_search, 3, device=device, requires_grad=False, dtype=dtype)
        
        # And this is a set of 3D points on a grid, that we'll flatten:
        x = torch.linspace(-1, 1, 30, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, 30, device=device, dtype=dtype)
        z = torch.linspace(-1, 1, 30, device=device, dtype=dtype)
        
        # Create 3D meshgrid
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Flatten and stack to get grid points as (N, 3) tensor
        grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

        grid_features = torch.randn(grid_points.shape[0], target_features, device=device, requires_grad=False, dtype=dtype)

        return point_cloud, grid_points, grid_features

    def set_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def train_step(model, optimizer, grid_points, point_cloud, grid_features):
        # Pretent to train the model!
        optimizer.zero_grad()
        output = model.forward(grid_points, point_cloud)
        loss = torch.mean((output - grid_features)**2)
        loss.backward()
        optimizer.step()
        return loss

To run this, you'll need to use a function like this to measure the performance.  Note the presence of the PhysicsNeMo profiler to quickly and easily enable pytorch profiling.  You'll want ``from physicsnemo.utils.profiling import Profiler`` at the top level of your python script (along with ``import torch``!)


.. code-block:: python

    def measure_performance(model, inputs, warmup_iters, benchmark_iters, profile=False):
        
        grid_points, point_cloud, grid_features = inputs
        
        profiler = Profiler()
        if profile:
            profiler.enable("torch")
        
        # Make a dummy optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Warm up:
        for i in range(warmup_iters):
            # Forward only:
            model.forward(grid_points, point_cloud)
        
            # Training:
            loss = train_step(model, optimizer, grid_points, point_cloud, grid_features)

            
        torch.cuda.synchronize()
        
        with profiler:
            
            
            with torch.no_grad():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                # Benchmark the forward pass
                for i in range(benchmark_iters):
                    output =model.forward(grid_points, point_cloud)
                end_event.record()
                torch.cuda.synchronize()

            print(f"Time taken in forward: {start_event.elapsed_time(end_event) / benchmark_iters:.3f} ms per iteration")

            # Benchmark the training loop:

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            # Benchmark the backward pass
            for i in range(benchmark_iters):
                loss = train_step(model, optimizer, grid_points, point_cloud, grid_features)
                
            end_event.record()
            torch.cuda.synchronize()
        
        
            print(f"Time taken in backward: {start_event.elapsed_time(end_event) / benchmark_iters:.3f} ms per iteration")

Finally, we can execute the script like this:

.. code-block:: python
    
    if __name__ == "__main__":
        
        set_seed(42)
        
        target_features = 1
        n_grid_points = 30
        n_cloud_points = 100000
        dtype = torch.float32
        
        point_cloud, grid_points, grid_features = generate_data(n_cloud_points, n_grid_points, target_features, dtype=dtype)
        print(point_cloud.shape)
        print(grid_points.shape)
        print(grid_features.shape)
        
        model = kNN_Projector(k=7, hidden_dim=25, output_dim=target_features).cuda().to(dtype)
        
        warmup_iters = 5
        benchmark_iters = 15
        
        measure_performance(model, (grid_points, point_cloud, grid_features), warmup_iters, benchmark_iters, profile=False)


On an A100 GPU, we see performance like this:

.. code-block:: text

    torch.Size([100000, 3])
    torch.Size([27000, 3])
    torch.Size([27000, 1])
    Time taken in forward: 144.045 ms per iteration
    Time taken in backward: 144.758 ms per iteration

And, by introducing ``model = torch.compile(model)`` and no other changes, performance jumps by a factor of two:

.. code-block:: text

    Time taken in forward: 74.237 ms per iteration
    Time taken in backward: 74.657 ms per iteration

Sidebar: Profiling the compiled application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Why?  It's interesting to explore exactly what happened, here, to enable a 2x performance boost in this pretend application.  If you run this application with profiling on, and look at the two profiles (with and without compilation) you'll see pretty clearly some top kernels.

.. seealso::

    Want to learn more about how to profile your pytorch code?  Check out our profiling tutorial: :ref:`Profiling Applications in PhysicsNeMo`

**Uncompiled Application Performance (Top Operations):**

.. list-table::
   :header-rows: 1
   :widths: 35 12 10 12 12 12 12 8

   * - Name
     - Self CUDA
     - Self CUDA %
     - CUDA total
     - CUDA time avg
     - CUDA Mem
     - Self CUDA Mem
     - # of Calls
   * - ``aten::topk``
     - 1.864s
     - 41.69%
     - 1.864s
     - 62.133ms
     - 64.91 Mb
     - 64.19 Mb
     - 30
   * - ``aten::linalg_vector_norm``
     - 1.670s
     - 37.36%
     - 1.670s
     - 55.672ms
     - 301.76 Gb
     - 301.76 Gb
     - 30
   * - ``reduce_kernel<512, 1, ...>``
     - 1.670s
     - 37.36%
     - 1.670s
     - 3.480ms
     - 0 b
     - 0 b
     - 480
   * - ``radixFindKthValues<float, ...>``
     - 1.125s
     - 25.17%
     - 1.125s
     - 9.377ms
     - 0 b
     - 0 b
     - 120
   * - ``aten::sub``
     - 904.517ms
     - 20.23%
     - 904.517ms
     - 20.100ms
     - 905.27 Gb
     - 905.27 Gb
     - 45

**Compiled Application Performance (Top Operations):**

.. list-table::
   :header-rows: 1
   :widths: 35 12 10 12 12 12 12 8

   * - Name
     - Self CUDA
     - Self CUDA %
     - CUDA total
     - CUDA time avg
     - CUDA Mem
     - Self CUDA Mem
     - # of Calls
   * - ``aten::topk``
     - 1.863s
     - 83.53%
     - 1.863s
     - 54.802ms
     - 64.91 Mb
     - 64.91 Mb
     - 34
   * - ``radixFindKthValues<float, ...>``
     - 1.125s
     - 50.42%
     - 1.125s
     - 9.373ms
     - 0 b
     - 0 b
     - 120
   * - ``Torch-Compiled Region: 0/0``
     - 0.000us
     - 0.00%
     - 1.112s
     - 74.163ms
     - 436.00 Mb
     - 0 b
     - 15
   * - ``CompiledFunction``
     - 0.000us
     - 0.00%
     - 1.112s
     - 74.163ms
     - 436.00 Mb
     - 403.54 Mb
     - 15
   * - ``gatherTopK<float, unsigned>``
     - 682.819ms
     - 30.61%
     - 682.819ms
     - 22.761ms
     - 0 b
     - 0 b
     - 30
   * - ``triton_poi_fused_linalg_vector_norm_sub_0``
     - 352.928ms
     - 15.82%
     - 352.928ms
     - 11.764ms
     - 0 b
     - 0 b
     - 30

Take note of the top kernels before compilation: ``aten::topk`` was (and still is) dominant.  But right before we call ``topk`` in user code, we compute the norm of all the points together in the point cloud: ``aten::linalg_vector_norm`` takes 55ms and it's significantly less in the compiled version (and, it shows up under a different name!)  This doesn't account for all of the difference, though it's a lot.  To learn more about understanding the profiling results, check out :ref:`Profiling Applications in PhysicsNeMo`.


Improving Performance with RAPIDS
---------------------------------

Now, we've seen that torch.compile can accelerate our code, but if we step back at think about the kNN algorithm, we'll realize it's not ideal.  We are computing this with an N*M algorithm (every point in ``p1`` compared to every point in ``p2``) - and it's expensive particularly in memory usage.  Better algorithms exist - and it's not the subject of this tutorial to get into them - and we already have a good example in Nvidia's RAPIDS ecosystem: `cuML Nearest Neighbors <https://docs.rapids.ai/api/cuml/stable/api/#neighbors>`_.  These days, integrating into pytorch is straightforward.  Update our ``knn_weighted_feature_aggregation`` function:

.. code-block:: python

    def knn_weighted_feature_aggregation(
        p1: torch.Tensor, 
        p2: torch.Tensor,
        p2_features: torch.Tensor, 
        k: int = 3, 
        sigma: float = 0.1,
        eps: float = 1e-8
        ) -> torch.Tensor:
        """
        """
        
        
        # Find top-k nearest neighbors (Make sure to cast to float32 for cuml)
        topk_dists, topk_idx = knn_search_with_cuml(p1.to(torch.float32), p2.to(torch.float32), k)

        # Gather neighbor features: (M, k, D_feat)
        neighbors = p2_features[topk_idx]

        # Compute weights: (M, k)
        weights = torch.softmax(-topk_dists / sigma, dim=1)

        # Weighted sum of neighbor features: (M, D_feat)
        agg = torch.sum(weights.unsqueeze(-1) * neighbors, dim=1)

        # Cast back to original dtype
        return agg.to(p1.dtype)

The difference is: replace the pointwise norm call with a ``knn_search_with_cuml`` call, and then directly get the neighbors based on the index. The rest is the same.  As for the ``knn_search_with_cuml`` function, it does the real heavy lifting with calls to cuml:

.. code-block:: python

    def knn_search_with_cuml(p1: torch.Tensor, p2: torch.Tensor, k: int = 3):
        # Use dlpack to move the data without copying between pytorch and cuml:
        p1 = cp.from_dlpack(p1)
        p2 = cp.from_dlpack(p2)
        
        # Construct the knn:
        knn = cuml.neighbors.NearestNeighbors(n_neighbors=k)
        # First pass partitions everything in p2 to make lookups fast
        knn.fit(p2)
        
        # Second pass uses that partition to quickly find neighbors of points in p1
        distance, indices = knn.kneighbors(p1)
        
        # convert back to pytorch:
        distance = torch.from_dlpack(distance)
        indices = torch.from_dlpack(indices)
        
        # Return torch objects.
        return distance, indices

A couple things to note about this function: it's pytorch in, pytorch out.  We've encapsulated all ``cuml`` contact to one region of code, which will be useful later.  Second, this function returns the distances and the indexes, which are both used, but the gradient in ``knn_weighted_feature_aggregation`` will flow through the output selected features, through the distance-weighted aggregation, and then through the ``neighbors = p2_features[topk_idx]`` line.  The ``topk_idx`` directs which indexes the gradients flow to but they are not themselves differentiable.  Likewise, the ``topk_dists`` tensor provides weights for gradeints in the backwards pass, but is itself not expecting gradients.  So: ``knn_search_with_cuml`` does not need to have a derivative implementation, and the backwards pass of this model just works.  It "just works" quite well, too:

.. code-block:: text

    Time taken in forward: 14.139 ms per iteration
    Time taken in backward: 15.842 ms per iteration

Now, if you try to compile this you will hit a warning:

.. code-block:: text

    /usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:700: UserWarning: Graph break due to unsupported builtin cupy._core.dlpack.from_dlpack. This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind). If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround. If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use torch.compiler.allow_in_graph.

And, performance is a little worse:

.. code-block:: text

    Time taken in forward: 15.697 ms per iteration
    Time taken in backward: 17.670 ms per iteration

The issue of course is our function, ``knn_search_with_cuml``, is calling operations that pytorch has no idea what to do with.  

Registering External Ops With PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

However, if you follow along with the `PyTorch Custom Ops Tutorial <https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial>`_, it's not hard to see how to extend this.  We have to register the function with pytorch:

.. code-block:: python

    @torch.library.custom_op("cuml::knn", mutates_args=())
    def knn_search_with_cuml(p1: torch.Tensor, p2: torch.Tensor, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
        p1 = cp.from_dlpack(p1)
        p2 = cp.from_dlpack(p2)
        
        knn = cuml.neighbors.NearestNeighbors(n_neighbors=k)
        knn.fit(p2)
        
        distance, indices = knn.kneighbors(p1)
        
        # convert back to pytorch:
        distance = torch.from_dlpack(distance)
        indices = torch.from_dlpack(indices)
        
        return distance, indices

And, we have to define a "fake" tensor function for this function: based on the inputs, it tells pytorch what the outputs will look like.  It's easily done with a decorator:

.. code-block:: python

    @knn_search_with_cuml.register_fake
    def _(p1, p2, k):
        assert p1.device == p2.device
        
        dist_output = torch.empty(p1.shape[0], k, device=p1.device, dtype=p1.dtype)
        idx_output = torch.empty(p1.shape[0], k, device=p1.device, dtype=torch.int64)
        
        return dist_output, idx_output

.. note:: 
    We don't even need to name this function.  It's consumed and registered with PyTorch, and PyTorch takes care of the rest.

With these changes, now ``torch.compile`` will work! You won't actually see a significant speedup, though - in fact you'll probably see negligible change in performance (< 1ms difference).  The challenge, here, is that while the ``cuml`` implementation is much much faster, it includes  cuda synchronize calls - which block execution on the GPU.  Since the rest of the model is so tiny, the compilation does almost nothing to improve it: we're bound now by kernel launch latency outside of that call.  You can - and should! - run the profiles and take a look to see that the GPU is now significantly more idle than it was in the first iteration of the code.  However, for real models, with much deeper and larger layers, which will not be a major issue.

Performance Bonus! Shared Memory Pools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do look at the profile, you'll see a lot of memory operations in the ``cuml`` region of the code.  Why?  It has to allocate memory for itself, and while both RAPIDS and PyTorch have dedicated memory management tools to accelerate this, they are not using the same pool of memory.  Fortunately, PyTorch easily allows you to swap in another memory allocator tool, and RAPID's memory mananger is easy to plug in.  Add these to your imports:

.. code-block:: python

    import rmm
    from rmm.allocators.torch import rmm_torch_allocator

And, before you initialize any data or models in pytorch, plug the RAPIDS memory managemer into pytorch:

.. code-block:: python

    rmm.reinitialize(pool_allocator=True)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    
This improves the runtime by a further ~3ms (which is > 20% faster on an already good speedup!):

.. code-block:: python

    # Not Compiled:
    Time taken in forward: 11.185 ms per iteration
    Time taken in backward: 12.818 ms per iteration

    # Compiled:
    Time taken in forward: 11.163 ms per iteration
    Time taken in backward: 12.571 ms per iteration

What about the backwards pass?
------------------------------

Implementing Custom Backward Passes with NVIDIA Warp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes - the backwards pass is supported too.  Our original example didn't actually need the backwards pass of the kNN operator.  Let's write a new kernel that does need one so we can see how this backwards incorporation works.  Instead of a kNN, we'll use **all** points within a specified radius to compute the same distance weighted feature aggregation.  In pytorch, it looks like this:

.. code-block:: python

    def radius_bounded_feature_aggregation(
        p1: torch.Tensor, 
        p2: torch.Tensor,
        p2_features: torch.Tensor, 
        radius: float, 
        sigma: float,
        ) -> torch.Tensor:
        """
        Perform differentiable radius-bounded feature aggregation.

        Args:
            p1 (torch.Tensor): Query points, shape (B, M, D)
            p2 (torch.Tensor): Reference points, shape (B, N, D)
            p2_features (torch.Tensor): Features at reference points, shape (B, N, D_feat)
            radius (float): Radius for neighbor search
            sigma (float): RBF temperature parameter

        Returns:
            torch.Tensor: Aggregated features at p1, shape (B, M, D_feat)
        """
        
        # Compute pairwise distances: (M, N)
        dists = torch.norm(p1[:,None,:] - p2[None,:,:], dim=-1)
        
        # Create mask for neighbors within radius
        mask = dists <= radius
        
        # Compute weights from all distances first
        weights = torch.softmax(-dists / sigma, dim=-1)
        
        # Apply mask to zero out weights for points outside radius
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        
        # Renormalize weights so they sum to 1 for each query point
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted sum of all reference features: (M, D_feat)
        agg = torch.sum(weights.unsqueeze(-1) * p2_features.unsqueeze(0), dim=1)
        
        return agg


Everything else about the pretend application can stay the same: an MLP on the features before the aggregation, an MLP on the aggregated features. You may need to decrease the number of points in the point cloud though - the pure-torch version of this operator is more memory hungry than the kNN query.

With 25k points in the point cloud, and grids of 30x30x30 points (27k), the uncompiled timing looks like this:

.. code-block:: text 

    Time taken in forward: 161.396 ms per iteration
    Time taken in backward: 260.719 ms per iteration

While compiled, we have a bit better performance:

.. code-block:: text

    Time taken in forward: 38.718 ms per iteration
    Time taken in backward: 86.226 ms per iteration

To implement this better, we'll turn to NVIDIA's `Warp <https://nvidia.github.io/warp/>`_ kernel language.  Warp is designed for differentiable physics simulation - and for this kernel, we can write it directly in Warp and it will generate the adjoint (aka, the backward pass) for us automatically.  The trick is that we can write a much more efficient version using Warp's ``HashGrid`` (`docs <https://nvidia.github.io/warp/modules/runtime.html#hash-grids>`_) object.

.. code-block:: python

    @wp.kernel
    def warp_radius_bounded_feature_aggregation(
        query_points: wp.array(dtype=wp.vec3),
        ref_points:  wp.array(dtype=wp.vec3),
        ref_features: wp.array2d(dtype=wp.float32),
        grid: wp.uint64,
        radius: float,
        sigma: float,
        output_features: wp.array2d(dtype=wp.float32),
    ):
        # Get the thread ID:
        tid = wp.tid()

        # Get position from query points
        pos = query_points[tid]

        feature_dim = ref_features.shape[1]
        local_output = output_features[tid]
        
        # Find all the neighbors using the hash grid:
        neighbors = wp.hash_grid_query(id=grid, point=pos, max_dist=radius)

        weight = float(0.0)
        # Loop through neighbors.  Compute a weighted distance and accumulate it, 
        # but also track the weight to normalize at the end.
        for index in neighbors:
            # Get the neighbor position:
            pos2 = ref_points[index]
            
            # Compute the distance:
            
            dist = wp.length(pos - pos2)
            # disp = pos - pos2 
            # dist = wp.sqrt(disp[0]**2. + disp[1]**2. + disp[2]**2.)
            
            if dist > radius:
                continue
            
            # Get the features at this index:
            feature = ref_features[index]
            
            # Compute the weight:
            this_weight = wp.exp( - dist / sigma)
            
            # Accumulate the weight, and weight * feature
            weight += this_weight
            # Work directly with the 2D array indexing instead of the slice
            for j in range(feature_dim):
                local_output[j] += feature[j] * this_weight 
        
        if weight > 0.0:
            # Normalize by working directly with the 2D array
            for j in range(feature_dim):
                local_output[j] /= (weight + 1e-8)
        
        # Write the output
        for j in range(feature_dim):
            output_features[tid,j] = local_output[j]

.. note::

    The function above is a ``warp`` *kernel*, not a python function.  It's written in python, but in reality it's compiled to CUDA and launched like any other kernel - but all from python itself.  Read more about warp kernels `here <https://nvidia.github.io/warp/basics.html#kernels>`_

This kernel replaces nearly the entirety of the ``torch`` code: it finds all the points within the radius and then directly accumulates them into the output.  Launching a ``warp`` kernel from pytorch is straightforward, due to Warp's interoperability with several other languages:

.. code-block:: python

    @torch.library.custom_op("warp::radius_bounded_feature_aggregation", mutates_args=())
    def radius_bounded_feature_aggregation_impl(
        query_points: torch.Tensor, 
        ref_points: torch.Tensor, 
        ref_features: torch.Tensor, 
        radius: float, 
        sigma: float,
    ) -> torch.Tensor:


        # Convert to warp
        # We can only build and query the points in wp.vec3 format:
        wp_query_points = wp.from_torch(
            query_points, 
            dtype=wp.vec3, # vec3 here!
            requires_grad=query_points.requires_grad
        )
        wp_ref_points = wp.from_torch(
            ref_points, 
            dtype=wp.vec3, # and here!
            requires_grad=ref_points.requires_grad
        )
        wp_ref_features = wp.from_torch(
            ref_features, 
            dtype=wp.float32, # but, the features we keep as an array!
            requires_grad=ref_features.requires_grad
        )
        
        
        # In the data generation, we had set the grid to range over -1 to 1 with 30 points.
        # We can use that to dictate the grid size:
        grid_size = int(2. / radius) + 1
        
        # **In general** to do this, you'd have to incur a performance penalty:
        # 1. Find the min max of all the points in your query set.
        # 2. Divide the range by the radius to get the grid size.
        # 3. Move the grid size to the CPU (which is blocking if done wrong)
        # 4. Construct the grid on the CPU.
        
        # But, it's **likely** you'd be constructing the grid once and caching it, anyways.
        

        
        # Build the grid used in the kernel:
        hash_grid = wp.HashGrid(grid_size, grid_size, grid_size)

        # This actually loops over the points and does the hashing
        hash_grid.build(wp_ref_points, radius)
        
                
        # Allocate output space (with pytorch!) and convert to warp:
        output_features = torch.zeros(
            (query_points.shape[0], ref_features.shape[1], ),
            device=ref_features.device, 
            dtype=ref_features.dtype,
            requires_grad=ref_features.requires_grad
        )
        
        wp_output = wp.from_torch(
            output_features,
            dtype=wp.float32,
            requires_grad=ref_features.requires_grad
        )
        
        # Launch the kernel:
        feature_dim = ref_features.shape[1]
        wp.launch(
            warp_radius_bounded_feature_aggregation,
            inputs=[
                wp_query_points, 
                wp_ref_points, 
                wp_ref_features, 
                hash_grid.id, 
                radius, 
                sigma,
            ],
            outputs =[
                wp_output,
            ],
            dim=[query_points.shape[0]],
        )

        
        # return the output features:
        return output_features

For more details on Warp, it's interface, and tools available, head over to their `documentation <https://nvidia.github.io/warp/>`_.

If you followed along with the first half, you'll recognize the declaration to the function: ``@torch.library.custom_op("warp::radius_bounded_feature_aggregation", mutates_args=())``.   The fake registration is still necessary too:

.. code-block:: python

    @radius_bounded_feature_aggregation_impl.register_fake
    def _(
        query_points: torch.Tensor, 
        ref_points: torch.Tensor, 
        ref_features: torch.Tensor, 
        radius: float, 
        sigma: float,
    ) -> torch.Tensor:

        assert query_points.is_cuda
        assert ref_points.is_cuda
        assert ref_features.is_cuda
        
        output = torch.empty(
            (query_points.shape[0], ref_features.shape[1], ),
            device=query_points.device, 
            dtype=query_points.dtype)
        
        return output

Now, we do the fun part: going backwards.  The actual backwards pass to launch the warp kernel is very similar to the forwards pass: 

.. code-block:: python

    @torch.library.custom_op("warp::radius_bounded_feature_aggregation_bwd", mutates_args=())
    def radius_bounded_feature_aggregation_bwd_impl(    
        query_points: torch.Tensor, 
        ref_points: torch.Tensor, 
        ref_features: torch.Tensor, 
        radius: float, 
        sigma: float,
        output_features: torch.Tensor,
        grad_outputs : torch.Tensor,
    ) -> torch.Tensor:
        
        # This function only needs to get the gradients of p2_features,
        # based on the grad_output of the forward outputs. 
        # Everything else is None for a grad.
        
        # Convert to warp:
        wp_query_points = wp.from_torch(
            query_points,
            dtype=wp.vec3,
            requires_grad=False
        )
        # We can only build and query the points in float32:
        wp_ref_points = wp.from_torch(
            ref_points,
            dtype=wp.vec3,
            requires_grad=False
        )
        #########################################################
        # Because we set requires_grad True here, warp will 
        # populate gradients HERE in the .grad attribute.
        #########################################################
        wp_ref_features = wp.from_torch(
            ref_features,
            dtype=wp.float32,
            requires_grad=True,
        )

        # In the data generation below, we set the grid to range over -1 to 1 with 30 points.
        # We can use that to dictate the grid size:
        grid_size = int(2. / radius) + 1
        
        # Build the grid used in the kernel:
        # In a real application, you'd cache and retrieve this in the backwards pass.
        # The trick would be to make sure the hash_grid object persists but is not 
        # actually in the torch interface (it would break things for the compiler!)
        hash_grid = wp.HashGrid(grid_size, grid_size, grid_size)

        # We're rebuilding here just to make the implementation straightforward
        hash_grid.build(wp_ref_points, radius)
        
        wp_output = wp.from_torch(output_features)
        
        wp_grad_outputs = wp.from_torch(grad_outputs)

        
        feature_dim = ref_features.shape[1]
        # Launch the kernel:
        wp.launch(
            warp_radius_bounded_feature_aggregation,
            inputs=[
                wp_query_points, 
                wp_ref_points, 
                wp_ref_features, 
                hash_grid.id, 
                radius, 
                sigma,
            ],
            outputs =[
                wp_output,
            ],
            adj_inputs = [
                None,
                None,
                wp_ref_features.grad,
                None,
                None,
                None,
            ],
            adj_outputs = [
                wp_grad_outputs,
            ],
            adjoint=True,  ############ Pay attention here!  Launch the kernel adjoint!
            dim=[query_points.shape[0]],
        )

        # return the gradient of the features:
        return ref_features.grad

Registering the fake is also straightforward and just like the forward pass:

.. code-block:: python

    @radius_bounded_feature_aggregation_bwd_impl.register_fake
    def _(
        query_points: torch.Tensor, 
        ref_points: torch.Tensor, 
        ref_features: torch.Tensor, 
        radius: float, 
        sigma: float,
        output_features: torch.Tensor,
        grad_outputs : torch.Tensor
    ) -> torch.Tensor:
        grad_outputs = torch.empty_like(ref_features)
        return grad_outputs

Connecting all the pieces
^^^^^^^^^^^^^^^^^^^^^^^^^

So far, we haven't actually told PyTorch this is a backwards pass function.  Instead, we've just declared a similar function, that happens to have ``bwd`` in the name.  For the autograd system in PyTorch, we need two more steps to enable this all to connect.  First, we have to set up the context, which is useful to save the forward pass objects for the backwards calculations.  And second, we need to tell PyTorch exactly which function is the backwards pass and which function is the context setup - a simple one-liner to register them in the autograd system:
 
.. code-block:: python

    # Use this to save any inputs or outputs for the backward pass.
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
        query_points, ref_points, ref_features, radius, sigma = inputs
        out_features = output
        ctx.save_for_backward(query_points, ref_points, ref_features, out_features)

        ctx.radius = radius
        ctx.sigma = sigma
        
    # And this connects the forward pass to the backward pass with pytorch.
    radius_bounded_feature_aggregation_impl.register_autograd(radius_bounded_feature_aggregation_backward_worker, setup_context=setup_context)
 
.. note:: 

    If you've written a ``torch.autograd.Function`` subclass before, you're familiar with the context though it may look different to you in this form.  PyTorch recommends this syntax to enable their functional API as well, but, if you don't need that you are welcome to use your tried and true inheritance from ``torch.autograd.Function``.

    If you're **not** familiar with how PyTorch does autograd work - check out their excellent guide `here <https://docs.pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html>`_


The rest of the application proceeds just like before. With Warp, though, the performance is even better than compiled pytorch:

.. code-block:: text

    Time taken in forward: 13.511 ms per iteration
    Time taken in backward: 29.953 ms per iteration

Success!

Conclusion
----------

In this tutorial, we saw some cool features of integrating highly performant code into pytorch applications, and how to combine them with ``torch.compile``.  Some key takeaways:

- ``torch.compile`` is generally great for performance.  Use it unless there is a reason you can't!
- If the reason you can't use ``torch.compile`` is that you have to leave the pytorch ecosystem to get a better performing kernel - then use `this method <https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html>`_ to register your wrapper functions and enable the compiler to seamlessly incorporate them.
- If you're calling out to other libraries in the NVIDIA ecosystem, like RAPIDS, you can get **even better** performance sharing a memory manager!  Check out the `RAPIDS Memory Manager <https://github.com/rapidsai/rmm>`_, which plugs in to pytorch (and also ``cupy`` and ``numba``!)
- You can just as easily incorporate a backwards pass as a forwards pass.  In this tutorial we leveraged the autograd capabilites of ``warp``, but in other cases you can of course write the backwards pass yourself.

Without these techniques, for many scientific AI workloads users are faced with a choice: use torch.compile on their model, including inefficient functions; or, skip ``torch.compile`` and leverage fast, accelerated calls in the NVIDIA python ecosystem.  The reality is, though, that you don't need to make that choice: you can have both performance enhancements.


.. toctree::
    :local:
    :depth: 1
    :hidden:
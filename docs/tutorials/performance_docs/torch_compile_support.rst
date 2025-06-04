Torch Compile and External Kernels
==================================

In 2023, with the `release of PyTorch 2.0 <https://pytorch.org/blog/pytorch-2-0-release/>`, PyTorch deployed the `torch.compile` API as a user facing, ready-to-use model compiler.  `torch.compile` is a handy tool for improving performance of AI models, in both training and inference, but often there can be edge cases where certain functionality needed in the model is impossible to use with `torch.compile`.  In this tutorial, we'll look at a small, toy example application that benefits from functionality that torch can't offer - and then we'll see how to enable torch.compile to use this functionality without graph breaks.

What does `torch.compile`` do?
------------------------------

If you're interested in `torch.compile`, you've probably already found the `tutorial from PyTorch <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial_.html>`_.  At a high level, `torch.compile` is a tool that allows pytorch to inspect your model ahead of time, find places where kernels can be optimized or combined, and enable those optimizations.  The performance gain is heavily dependent on the application: kernel fusion (like a convolution + activation) can help reduce runtime by mitigating the memory-bound characteristics of one kernel when fusing it to compute bound kernel.  Further, performance gains are highly dependent on compute precision as well: the thresholds for what is "compute-bound" and what is "memory-bound" are different depending on the precision. Lower precisions can take advantage of smaller memory footprints (so less bandwidth is necessary from memory) as well as dedicated processing units like Tensor cores for faster math operations.

With all of that in mind - that tutorial is focused on pure PyTorch functionality.  In PhysicsNeMo workloads, however, we often need to leverage tools that live outside the pytorch ecosystem.  But with large, complex, and end-to-end models, we still want to take advantage of the performance benefits we can get with `torch.compile`.  So in the rest of this tutorial, we'll look at exactly how to solve that problem.  

This tutorial is broken into two models: first, we'll work on a k-Nearest-Neighbors type problem, which we can accelerate with `cuml`.  Second, we'll do a closer examination of the backwards-pass functionality in torch.compile (and you'll learn why it wasn't necessary in the first example, even though we're doing training!)

Introducing the Application
---------------------------

For demonstration purposes, we've invented a small operator that works on point-cloud like data.  For PhysicsNeMo users, you'll recognize similar ideas in architectures such as DoMINO and FigConvNet.  We're not specifically using these models, however, this is a fully independent example application.

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
            A simple 3 layer MLP that takes in a tensor of shape (N, k) and outputs a tensor of shape (N, 1)
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
            Accept two point clouds, p1 and p2.  Compute a learnable projection on p2 to 
            learn features.  Then, use a kNN-weighted aggregation to project those features
            on to p1.
            """
            p2_features = self.proj(p2)
            
            p1_features = knn_weighted_feature_aggregation(p1, p2, p2_features, k=self.k)

            return self.proj_out(p1_features)
    
In basic terms, this model operates on two sets of point clouds.  A reference set of points, `p2`, has some features learned on it by the first MLP.  Then, using the k nearest neighbors to each point in `p1`, the features in `p2` are projected (`knn_weighted_feature_aggregation`) onto the locations in `p1`.  Finally, the output features from the aggregation are projected to a final latent space via a second MLP.  The details of the projection look like this:

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

.. note:: 
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

To run this, you'll need to use a function like this to measure the performance.  Note the presence of the PhysicsNeMo profiler to quickly and easily enable pytorch profiling.  You'll want `from physicsnemo.utils.profiling import Profiler` at the top level of your python script (along with `import torch`!)


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

.. text ::

    torch.Size([100000, 3])
    torch.Size([27000, 3])
    torch.Size([27000, 1])
    Time taken in forward: 144.045 ms per iteration
    Time taken in backward: 144.758 ms per iteration

And, by introducing `model = torch.compile(model)` and no other changes, performance jumps by a factor of two:

.. text ::

    Time taken in forward: 74.237 ms per iteration
    Time taken in backward: 74.657 ms per iteration

Why?  It's interesting to explore exactly what happened, here, to enable a 2x performance boost in this pretend application.  If you run this application with profiling on, and look at the two profiles (with and without compilation) you'll see pretty clearly some top kernels.

For the uncompiled application (first few lines):
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                             aten::topk         0.03%       1.560ms         0.09%       4.169ms     138.968us        1.864s        41.69%        1.864s      62.133ms           0 b           0 b      64.91 Mb      64.19 Mb            30            --
                               aten::linalg_vector_norm         0.04%       1.674ms         3.73%     166.620ms       5.554ms        1.670s        37.36%        1.670s      55.672ms           0 b           0 b     301.76 Gb     301.76 Gb            30            --
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us        1.670s        37.36%        1.670s       3.480ms           0 b           0 b           0 b           0 b           480            --
void at::native::mbtopk::radixFindKthValues<float, u...         0.00%       0.000us         0.00%       0.000us       0.000us        1.125s        25.17%        1.125s       9.377ms           0 b           0 b           0 b           0 b           120            --
                                              aten::sub         0.03%       1.349ms         9.42%     420.451ms       9.343ms     904.517ms        20.23%     904.517ms      20.100ms           0 b           0 b     905.27 Gb     905.27 Gb            45            --
void at::native::elementwise_kernel<128, 2, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us     904.480ms        20.23%     904.480ms       1.884ms           0 b           0 b           0 b           0 b           480            --
void at::native::mbtopk::gatherTopK<float, unsigned ...         0.00%       0.000us         0.00%       0.000us       0.000us     682.982ms        15.28%     682.982ms      22.766ms           0 b           0 b           0 b           0 b            30            --
void at::native::mbtopk::computeBlockwiseWithinKCoun...         0.00%       0.000us         0.00%       0.000us       0.000us      53.654ms         1.20%      53.654ms     447.114us           0 b           0 b           0 b           0 b           120            --
                                           aten::linear         0.02%     857.446us         1.78%      79.658ms     442.546us       0.000us         0.00%      19.827ms     110.148us           0 b           0 b       1.03 Gb           0 b           180            --
                                            aten::addmm         0.11%       4.726ms         1.74%      77.436ms     430.199us      19.827ms         0.44%      19.827ms     110.148us           0 b           0 b       1.03 Gb       1.03 Gb           180     10015.500
                                 ampere_sgemm_32x128_tn         0.00%       0.000us         0.00%       0.000us       0.000us      19.551ms         0.44%      19.551ms     130.342us           0 b           0 b           0 b           0 b           150            --
    autograd::engine::evaluate_function: AddmmBackward0         0.03%       1.291ms         3.66%     163.129ms       1.813ms       0.000us         0.00%       4.450ms      49.444us           0 b           0 b    -186.06 Mb    -588.26 Mb            90            --
                                         AddmmBackward0         0.03%       1.201ms         1.80%      80.509ms     894.542us       0.000us         0.00%       3.326ms      36.956us           0 b           0 b     402.16 Mb           0 b            90            --
                                               aten::mm         0.08%       3.590ms         1.74%      77.697ms     470.890us       3.326ms         0.07%       3.326ms      20.158us           0 b           0 b     402.16 Mb     402.16 Mb           165      9790.500
    autograd::engine::evaluate_function: IndexBackward0         0.00%     141.205us         0.08%       3.377ms     225.140us       0.000us         0.00%       2.355ms     157.028us           0 b           0 b    -135.07 Mb    -292.01 Mb            15            --
                                         IndexBackward0         0.00%      99.489us         0.07%       3.236ms     215.726us       0.000us         0.00%       2.355ms     157.028us           0 b           0 b     156.94 Mb           0 b            15            --
                                 aten::_index_put_impl_         0.01%     609.802us         0.06%       2.732ms     182.106us       2.023ms         0.05%       2.260ms     150.645us           0 b         -24 b           0 b    -108.18 Mb            15            --

And for the compiled version (again, first few lines):

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Total MFLOPs
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                             aten::topk         0.14%       3.687ms         0.26%       6.643ms     195.372us        1.863s        83.53%        1.863s      54.802ms           0 b           0 b      64.91 Mb      64.91 Mb            34            --
void at::native::mbtopk::radixFindKthValues<float, u...         0.00%       0.000us         0.00%       0.000us       0.000us        1.125s        50.42%        1.125s       9.373ms           0 b           0 b           0 b           0 b           120            --
                             Torch-Compiled Region: 0/0         0.03%     682.642us         2.76%      71.251ms       4.750ms       0.000us         0.00%        1.112s      74.163ms           0 b           0 b     436.00 Mb           0 b            15            --
                                       CompiledFunction         0.19%       4.922ms         2.74%      70.568ms       4.705ms       0.000us         0.00%        1.112s      74.163ms           0 b           0 b     436.00 Mb     403.54 Mb            15            --
                             Torch-Compiled Region: 0/1         0.16%       4.240ms         0.43%      11.183ms     745.543us       0.000us         0.00%        1.112s      74.160ms           0 b           0 b       1.55 Mb     -30.91 Mb            15            --
void at::native::mbtopk::gatherTopK<float, unsigned ...         0.00%       0.000us         0.00%       0.000us       0.000us     682.819ms        30.61%     682.819ms      22.761ms           0 b           0 b           0 b           0 b            30            --
              triton_poi_fused_linalg_vector_norm_sub_0         0.02%     440.999us         0.03%     795.212us      26.507us     352.928ms        15.82%     352.928ms      11.764ms           0 b           0 b           0 b           0 b            30            --
              triton_poi_fused_linalg_vector_norm_sub_0         0.00%       0.000us         0.00%       0.000us       0.000us     352.928ms        15.82%     352.928ms      11.764ms           0 b           0 b           0 b           0 b            30            --
void at::native::mbtopk::computeBlockwiseWithinKCoun...         0.00%       0.000us         0.00%       0.000us       0.000us      53.537ms         2.40%      53.537ms     446.146us           0 b           0 b           0 b           0 b           120            --
                                 ampere_sgemm_32x128_tn         0.00%       0.000us         0.00%       0.000us       0.000us       6.996ms         0.31%       6.996ms      46.643us           0 b           0 b           0 b           0 b           150            --
                                            aten::addmm         0.51%      13.150ms         1.62%      41.816ms     497.806us       5.694ms         0.26%       5.694ms      67.787us           0 b           0 b           0 b           0 b            84      5125.900
autograd::engine::evaluate_function: CompiledFunctio...         0.01%     184.032us         3.28%      84.465ms       5.631ms       0.000us         0.00%       5.036ms     335.757us           0 b           0 b    -435.79 Mb      -1.55 Mb            15            --
                               CompiledFunctionBackward         0.33%       8.388ms         3.27%      84.281ms       5.619ms       0.000us         0.00%       5.036ms     335.757us           0 b           0 b    -434.25 Mb    -434.25 Mb            15            --
                                               aten::mm         0.24%       6.292ms         2.97%      76.567ms     263.117us       4.834ms         0.22%       4.834ms      16.613us           0 b           0 b           0 b           0 b           291     16335.700
void at::native::bitonicSortKVInPlace<2, -1, 16, 16,...         0.00%       0.000us         0.00%       0.000us       0.000us       1.044ms         0.05%       1.044ms      34.800us           0 b           0 b           0 b           0 b            30            --
                                 ampere_sgemm_32x128_nn         0.00%       0.000us         0.00%       0.000us       0.000us     985.340us         0.04%     985.340us      13.138us           0 b           0 b           0 b           0 b            75            --
void cublasLt::splitKreduce_kernel<32, 16, int, floa...         0.00%       0.000us         0.00%       0.000us       0.000us     780.026us         0.03%     780.026us       8.667us           0 b           0 b           0 b           0 b            90            --

Take note of the top kernels before compilation: `aten::topk` was (and still is) dominant.  But right before we call `topk` in user code, we compute the norm of all the points together in the point cloud: `aten::linalg_vector_norm` takes 55ms and it's significantly less in the compiled version (and, it shows up under a different name!)  This doesn't account for all of the difference, though it's a lot.  To learn more about understanding the profiling results, check out :ref:`Profiling Applications in PhysicsNeMo`.

How can we make this faster?
----------------------------

Now, we've seen that torch.compile can accelerate our code, but if we step back at think about the kNN algorithm, we'll realize it's not ideal.  We are computing this with an N*M algorithm (every point in `p1` compared to every point in `p2`) - and it's expensive particularly in memory usage.  Better algorithms exist - and it's not the subject of this tutorial to get into them - and we already have a good example in Nvidia's RAPIDS ecosystem: `cuML Nearest Neighbors <https://docs.rapids.ai/api/cuml/stable/api/#neighbors>`_.  These days, integrating into pytorch is straightforward.  Update our `knn_weighted_feature_aggregation` function:

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

The difference is: replace the pointwise norm call with a `knn_search_with_cuml` call, and then directly get the neighbors based on the index. The rest is the same.  As for the `knn_search_with_cuml` function, it does the real heavy lifting with calls to cuml:

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

A couple things to note about this function: it's pytorch in, pytorch out.  We've encapsulated all `cuml` contact to one region of code, which will be useful later.  Second, this function returns the distances and the indexes, which are both used, but the gradient in `knn_weighted_feature_aggregation` will flow through the output selected features, through the distance-weighted aggregation, and then through the `neighbors = p2_features[topk_idx]` line.  The `topk_idx` directs which indexes the gradients flow to but they are not themselves differentiable.  Likewise, the `topk_dists` tensor provides weights for gradeints in the backwards pass, but is itself not expecting gradients.  So: `knn_search_with_cuml` does not need to have a derivative implementation, and the backwards pass of this model just works.  It "just works" quite well, too:

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

The issue of course is our function, `knn_search_with_cuml`, is calling operations that pytorch has no idea what to do with.  However, if you follow along with the `PyTorch Custom Ops Tutorial <https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial>`_, it's not hard to see how to extend this.  We have to register the function with pytorch:

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

With these changes, now `torch.compile` will work! You won't actually see a significant speedup, though - in fact you'll probably see negligible change in performance (< 1ms difference).  The challenge, here, is that while the `cuml` implementation is much much faster, it includes  cuda synchronize calls - which block execution on the GPU.  Since the rest of the model is so tiny, the compilation does almost nothing to improve it: we're bound now by kernel launch latency outside of that call.  You can - and should! - run the profiles and take a look to see that the GPU is now significantly more idle than it was in the first iteration of the code.  However, for real models, with much deeper and larger layers, which will not be a major issue.

Bonus!
------

If you do look at the profile, you'll see a lot of memory operations in the `cuml` region of the code.  Why?  It has to allocate memory for itself, and while both RAPIDS and PyTorch have dedicated memory management tools to accelerate this, they are not using the same pool of memory.  Fortunately, PyTorch easily allows you to swap in another memory allocator tool, and RAPID's memory mananger is easy to plug in.  Add these to your imports:

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
==============================

[Coming Next]
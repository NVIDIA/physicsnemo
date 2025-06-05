Performance 
===========

In the :ref:`Profiling Applications in PhysicsNeMo`, we saw an end to end application speedup by leveraging performance analysis tools applied to an AI application.  In this tutorial, we're going to dive a little deeper into a variety of subjects to get a better understanding of how to apply some of the best performance tricks - specifically focused on the tools you need to make your scientific AI applications faster.

One major challenge in performance optimization is understanding where to start.  Always remember `Amdahl's Law <https://en.wikipedia.org/wiki/Amdahl%27s_law?>`_.  As cited on Wikipedia, reproduced here for conviencence:

.. admonition:: Amdahl's Law - Key Performance Principle
   
   "The overall performance improvement gained by optimizing a single part of a system is limited by the fraction of time that the improved part is actually used".

   --  Reddy, Martin (2011). API Design for C++. Burlington, Massachusetts: Morgan Kaufmann Publishers. p. 210. doi:10.1016/C2010-0-65832-9. ISBN 978-0-12-385003-4. LCCN 2010039601. OCLC 666246330.

As it applies to performance of AI applications, Amdahl's law reminds us to view AI as end-to-end, CPU+GPU applications that have many computational subsystems.  Doubling the performance of your AI model won't really matter if your application spends 90% of it's time blocked on IO and data preprocessing.  While it's not a comprehensive list, some of the critical components of an AI pipeline can be:

- **Model Performance** Once inputs are loaded on to the GPU, the AI model itself has a number of tools for improving computational performance.  We'll get in to some of these below, but ``torch.compile`` (`tutorial <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_), ``CUDA Graphs`` (`blog post <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>`_), ``mixed precision`` (`examples <https://docs.pytorch.org/docs/stable/notes/amp_examples.html>`_), ``multi-device parallelism with NCCL`` (`docs <https://docs.pytorch.org/docs/stable/distributed.html>`_), and specialized kernels from the Nvidia ecosystem such as ``cuML`` (`docs <https://docs.rapids.ai/api/cuml/stable/>`_), ``cuGraphs`` (`docs <https://docs.rapids.ai/api/cugraph/stable/>`_), and ``Warp`` (`docs <https://nvidia.github.io/warp/>`_) are all powerful tools to improve model performance.

- **Data Loading** In small scale AI applications, a dataset might be loaded to CPU RAM and streamed to the GPU in batches, as needed.  For Scientific AI, however, datasets often are measured in units of **TB**.  Data loading can become a serious bottleneck for application performance, in both training and inference.  Several libraries exist (``HDF5`` (`docs <https://docs.h5py.org/en/stable/index.html>`_), ``Zarr`` (`docs <https://zarr.dev/>`_), and `higher level tools <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_) that can load data faster than pure numpy.  But interactions with storage systems, CPU cores, CPU-GPU transfers, and other hardware components can quickly complicate data loading, often with unexpected performance degregations.  Some out of the box solutions can be easily handled with tools like `NVIDIA DALI <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html>`_, while other applications take more effort.

- **Data Preprocessing** Once data has been loaded from file, there are often preprocessing steps required before the data can flow to the AI model.  This can include anything from deterministic data transformations (padding data for your model, normalization, or others) to stochastic, run-time transformations (subsampling of large data, augmentation of data with noise, random cropping, mirroring, etc.). If done on the GPU, this can limit application performance by starving the GPU of work.

- **Scaling Performance**  Applications that run well on a single GPU may struggle when deploying to a multi-GPU or multi-node system.  There is great documentation (see ``PyTorch DDP`` (`tutorial <https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_) for example) on model scale up.  Less tools and tutorials exist, however, to help you manage parallel-IO, checkpoint writing and restoring, or aggregating and tracking metrics efficiently. 

When it comes time to evaluate your model for performance optimization, keep these ideas in mind.  The sub-sections below can help you improve performance of certain areas, but where you spend your time for optimizations should be guided empirically by application performance and bottlenecks.  Meaning, of course: you need to profile your whole application, and often with multiple tools to get a full picture of performance bottlenecks.  Further - after you've made improvements, make sure to *reprofile* before you decide what to optimize next.


**Performance Topics**

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Topic
     - Description
   * - :doc:`performance_docs/torch_compile_support`
     - Learn how to integrate other kernels effectively into your models and use ``torch.compile`` to enable end-to-end model compilation for maximum performance.
   * - IO
     - *[Coming Soon]* When IO becomes a bottleneck, learn how to read and write data asynchronously, prefetch data to overlap IO and computation, and leverage GPU acceleration.


.. note::
   These performance guides are works in progress. Look for much more updated content in our next release!

Is there a performance critical component of PhysicsNeMo or scientific AI workloads that doesn't get enough attention? Let us know on `GitHub <https://github.com/NVIDIA/physicsnemo>`_!


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   performance_docs/torch_compile_support
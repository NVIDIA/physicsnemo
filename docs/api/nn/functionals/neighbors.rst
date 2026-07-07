Neighbor Functionals
====================

KNN
---

.. autofunction:: physicsnemo.nn.functional.knn

Radius Search
-------------

.. autofunction:: physicsnemo.nn.functional.radius_search

.. note::

   **Experimental CUDA backends (env-gated).** When ``max_points`` is set and the
   inputs are on CUDA, alternative neighbor-search kernels can be selected via the
   ``PHYSICSNEMO_RADIUS_SEARCH_MORTON`` environment variable: ``scalar`` /
   ``fma`` / ``gemm`` (hash-grid Morton), ``dense_fma`` / ``dense_gemm``
   (dense-cell Morton), and ``pysdf_cuda``. The ``pysdf_cuda`` value uses a
   vendored, OptiX-free software-QBVH point-range query (NVIDIA pysdf / minigql
   plus owl, all Apache-2.0; see
   ``physicsnemo/nn/functional/neighbors/radius_search/_pysdf_cuda_ext/third_party/NOTICE``)
   that is JIT-compiled on first use and therefore requires a CUDA toolchain
   (``nvcc``). All of these backends are CUDA-only, require ``max_points`` to be
   set, and return the same output contract as the default backend (neighbor
   order within a query is unspecified). Leaving the variable unset uses the
   default Warp hash-grid implementation.

.. rubric:: Benchmarks (ASV)

.. figure:: ../../../img/nn/functional/radius_search/benchmark.png
   :alt: Radius search benchmark comparison
   :width: 100%

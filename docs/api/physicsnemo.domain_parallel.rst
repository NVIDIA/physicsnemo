
PhysicsNeMo ``domain_parallel``
================================

In scientific AI applications, the parallelization techniques to enable state of the art 
models are different from those used in training large language models.  PhysicsNeMo 
introduces a new parallelization primitive called a ``ShardTensor`` that is designed for 
large-input AI applications to enable domain parallelization.

``ShardTensor`` provides a distributed tensor implementation that supports uneven sharding across devices. 
It builds on PyTorch's DTensor while adding flexibility for cases where different ranks may have 
different local tensor sizes.

.. autosummary::
   :toctree: generated

``ShardTensor``
---------------

.. autoclass:: physicsnemo.domain_parallel.ShardTensor
    :members:
    :show-inheritance:

Utility Functions
-----------------

.. autofunction:: physicsnemo.domain_parallel.scatter_tensor

For detailed information on ``ShardTensor`` and domain parallelism, please refer to the :doc:`Domain Parallelism <../../user-guide/domain_parallelism_entry_point>` tutorial.

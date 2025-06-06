Domain Parallelism
==================

In large scale AI applications, spanning multiple GPUs, an AI programmer has multiple tools available for coordination of GPUs to scale an application.  In ``PhysicsNeMo``, we have focused on enabling one particular technique, called "domain parallelism", which is designed to parallelize execution of a model over the input data.  Several models in ``PhysicsNeMo`` enable this directly - such as MeshGraphNets and SFNO - while other models rely on more generic tools to enable domain parallelism.

To learn more about the domain parallel tools in ``PhysicsNemo``, dive in to the following tutorials:

- :ref:`Domain Parallelism and Shard Tensor` provides an overview of what domain parallelism is, when you might need it, and how ``PhysicsNeMo`` is used to support it.
- :ref:`Implementing new layers for ShardTensor` provides a deeper dive into how domain parallelism is extended to operations that may not be supported yet - and especially how you might use the tools in PyTorch and ``PhysicsNeMo`` to extend domain parallelism yourself.
- :ref:`Domain Decomposition, ShardTensor and FSDP Tutorial` provides an end-to-end example with synthetic data to show you how domain parallelism can be combined with other parallelism paradigms, like data parallel training.

If you have questions about domain parallelism and its applications in scientific AI, please find us on `GitHub <https://github.com/NVIDIA/physicsnemo>`_ to discuss!
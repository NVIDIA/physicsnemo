Domain Parallelism
==================

In large scale AI applications, spanning multiple GPUs, an AI programmer has multiple tools available for coordination of GPUs to scale an application.  In ``PhysicsNeMo``, we have focused on enabling one particular technique, called "domain parallelism", which is designed to parallelize execution of a model over the input data.  Several models in ``PhysicsNeMo`` enable this directly - such as MeshGraphNets and SFNO - while other models rely on more generic tools to enable domain parallelism.

To learn more about the domain parallel tools in ``PhysicsNemo``, dive in to the following tutorials:

.. grid:: 1 1 2 3
    :gutter: 3

    .. grid-item-card:: 
        :link: domain_parallelism/domain_parallelism
        :link-type: doc

        **Domain Parallelism and Shard Tensor**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Provides an overview of what domain parallelism is, when you might need it, and how ``PhysicsNeMo`` is used to support it.

    .. grid-item-card::
        :link: domain_parallelism/implementing_new_layers_in_shard_tensor
        :link-type: doc

        **Implementing New Layers**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Provides a deeper dive into how domain parallelism is extended to operations that may not be supported yet - and especially how you might use the tools in PyTorch and ``PhysicsNeMo`` to extend domain parallelism yourself.

    .. grid-item-card::
        :link: domain_parallelism/fsdp_and_shard_tensor
        :link-type: doc

        **ShardTensor and FSDP Tutorial**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Provides an end-to-end example with synthetic data to show you how domain parallelism can be combined with other parallelism paradigms, like data parallel training.

If you have questions about domain parallelism and its applications in scientific AI, please find us on `GitHub <https://github.com/NVIDIA/physicsnemo>`_ to discuss!



.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   domain_parallelism/domain_parallelism
   domain_parallelism/implementing_new_layers_in_shard_tensor
   domain_parallelism/fsdp_and_shard_tensor
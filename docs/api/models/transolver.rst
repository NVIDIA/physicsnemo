Transolver
==========

The Transolver model adapts the transformer architecture with a physics-attention
mechanism for solving partial differential equations on structured and unstructured
meshes. It projects inputs onto physics-informed slices before applying attention,
enabling efficient learning of physical systems.

Activation Checkpointing
------------------------

Transolver can checkpoint its transformer blocks during training to reduce the
activations retained for the backward pass. Set ``activation_checkpointing=True``
to checkpoint every block, or provide a fraction in ``(0, 1)`` to checkpoint that
leading fraction of the block stack. The default is ``False`` and preserves the
existing execution path.

.. code-block:: python

    model = Transolver(
        functional_dim=2,
        embedding_dim=3,
        out_dim=1,
        structured_shape=None,
        use_te=False,
        activation_checkpointing=True,
    )

Checkpointing is active only in training mode when gradients are enabled. It
uses non-reentrant checkpointing and trades additional recomputation in the
backward pass for lower peak activation memory. The native backend uses
PyTorch's checkpoint implementation, while ``use_te=True`` uses Transformer
Engine's checkpoint wrapper so that its activation-recompute and FP8 state are
handled correctly. Evaluation and ``torch.no_grad()`` execution are unchanged.

The option works in eager execution and can also be combined with
``torch.compile``. The explicit block policy remains useful under compilation:
it provides a predictable recomputation boundary rather than relying only on
the compiler's automatic, speed-oriented rematerialization choices.

.. autoclass:: physicsnemo.models.transolver.transolver.Transolver
    :show-inheritance:
    :members:
    :exclude-members: forward

Building blocks
---------------

.. autoclass:: physicsnemo.models.transolver.transolver.TransolverBlock
    :show-inheritance:
    :members:
    :exclude-members: forward

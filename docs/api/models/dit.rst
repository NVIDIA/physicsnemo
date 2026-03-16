.. _dit_model:

Diffusion Transformer (DiT)
============================

The Diffusion Transformer (DiT) is a Vision Transformer backbone for diffusion
models.  It operates on image patches via a patchify embedding, processes
tokens with a sequence of transformer blocks conditioned through adaptive
layer normalization (adaLN-Zero), and reconstructs the output via an unpatchify
step.

DiT was introduced in `Scalable Diffusion Models with Transformers, Peebles &
Xie <https://arxiv.org/abs/2212.09748>`_.

:code:`DiT`
------------

.. autoclass:: physicsnemo.models.dit.DiT
    :show-inheritance:
    :members:
    :exclude-members: forward

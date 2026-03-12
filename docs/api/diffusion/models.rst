.. _diffusion_model_backbones:

Model Backbones
===============

.. currentmodule:: physicsnemo.diffusion

The backbone is the underlying neural network architecture in a diffusion
model.  A backbone does not necessarily satisfy the
:class:`~physicsnemo.diffusion.DiffusionModel` protocol directly---many
popular architectures have their own forward signatures.  When a backbone's
signature differs, a thin adapter or wrapper is needed to bridge the gap (see
:ref:`adapters <backbone_adapters>` below).

PhysicsNeMo ships two families of architectures specialized for diffusion
(UNets and DiT), but the framework is not restricted to them.  Any model from
the :doc:`PhysicsNeMo model zoo <../../api_models>`, such as
:class:`~physicsnemo.models.mlp.FullyConnected`,
:class:`~physicsnemo.models.fno.FNO`, or graph neural networks, can be used
as a backbone, either directly or as a building block inside a larger model.
Models from external libraries can also be integrated (see
:ref:`below <backbone_external_models>`).


UNet-Based Architectures
------------------------

UNet-based architectures process features at multiple resolutions through an
encoder-decoder structure with skip connections.  Their spatial inductive
biases, locality, multi-scale structure, and translation equivariance, make
them particularly effective when the data has strong spatial structure and
training data is (relatively) limited.  However, they are less flexible than
transformers and do not scale as favorably to very large datasets and domains.

For complete API documentation, parameter descriptions, and usage examples,
see the :ref:`Diffusion UNets API reference <diffusion_unet_api>`.


Diffusion Transformer (DiT)
---------------------------

The Diffusion Transformer (DiT) is a Vision Transformer backbone for diffusion
models.  It operates on image patches via a patchify embedding, processes
tokens with a sequence of transformer blocks conditioned through adaptive
layer normalization, and reconstructs the output using an unpatchify step.

Compared to UNets, DiT has fewer inductive biases, that is, it does not assume locality
or multi-scale structure.  This makes it more flexible. For instance,
conditioning can be injected through customizable tokenization operators rather
than being limited to channel concatenation or class embeddings.  DiT
architectures also tend to scale more favorably with increasing amounts of
training data.

For complete API documentation, refer to the :ref:`DiT API reference <dit_model>`.

.. _backbone_adapters:

Adapting Backbones to the DiffusionModel Interface
---------------------------------------------------

Many backbone architectures have forward signatures that differ from the
:class:`~physicsnemo.diffusion.DiffusionModel` protocol.  For example,
:class:`~physicsnemo.models.diffusion_unets.SongUNet` expects
``(x, noise_labels, class_labels, augment_labels)``, while the framework
expects ``(x, t, condition)``.  A short adapter class solves this:

.. code-block:: python

    import torch
    from physicsnemo.core import Module
    from physicsnemo.models.diffusion_unets import SongUNet

    class SongUNetAdapter(Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.net = SongUNet(**kwargs)

        def forward(self, x, t, condition=None):
            return self.net(x, noise_labels=t, class_labels=condition)



.. _backbone_external_models:

Using External or Custom Models
-------------------------------

The diffusion framework is not limited to PhysicsNeMo-native backbones.  Any
model that satisfies the :class:`~physicsnemo.diffusion.DiffusionModel`
interface can be used with the framework's
:ref:`preconditioners <diffusion_preconditioners>`,
:ref:`losses <diffusion_metrics>`, and :ref:`samplers <diffusion_samplers>`.

You can bring in models from external libraries, such as:

* `HuggingFace Diffusers <https://huggingface.co/docs/diffusers>`_
* `timm <https://huggingface.co/docs/timm>`_ by writing a thin wrapper
* by using :meth:`~physicsnemo.core.module.Module.from_torch` to convert a
``torch.nn.Module`` subclass into a :class:`~physicsnemo.core.module.Module`
with PhysicsNeMo serialization support


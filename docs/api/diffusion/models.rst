.. _diffusion_model_backbones:

Model Backbones
===============

.. currentmodule:: physicsnemo.diffusion

The backbone is the underlying neural network architecture in a diffusion
or flow-matching model. A backbone does not necessarily meet the
:class:`~physicsnemo.diffusion.DiffusionModel` protocol directly---many
popular architectures have their own forward signatures.  When a backbone's
signature differs, a thin adapter or wrapper is needed to bridge the gap (see
:ref:`adapters <backbone_adapters>` below).

PhysicsNeMo ships two families of architectures specialized for diffusion
and flow matching (UNets and DiT), but the framework is not restricted to
them. Any model from the :doc:`PhysicsNeMo model zoo <../../api_models>` can
serve as a backbone. Examples include
:class:`~physicsnemo.models.mlp.FullyConnected`,
:class:`~physicsnemo.models.fno.FNO`, and graph neural networks, used
directly or as a building block inside a larger model.
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

The Diffusion Transformer (DiT) is a Vision Transformer backbone for
diffusion and flow-matching models. It operates on image patches through a
patchify embedding, processes tokens with a sequence of transformer blocks
conditioned through adaptive layer normalization, and reconstructs the
output using an unpatchify step.

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

Most diffusion and flow-matching models take one time tensor and return one
prediction tensor.
A compatible :class:`~physicsnemo.diffusion.DiffusionModel` can instead
accept ``(t, extra_times)`` with
:class:`~physicsnemo.diffusion.DiffusionExtraTimes` and return
``(prediction, extra_outputs)`` with
:class:`~physicsnemo.diffusion.DiffusionExtraOutputs`. For example, MeanFlow
uses a destination time ``r``, while sCM and EDM2 can use ``logvar`` for loss
weighting. The primary time and prediction remain required tensors.

The adapter below combines the destination time with the primary time for a
DiT backbone. A small
:class:`~physicsnemo.models.mlp.FullyConnected` head predicts a log-scale
from the primary time. This example does not define a canonical MeanFlow
architecture. A compatible loss decides how to use ``logvar`` during
training. Inference code that needs only a prediction can ignore it.

.. code-block:: python

    import torch
    from physicsnemo.core import Module
    from physicsnemo.diffusion import DiffusionExtraOutputs, DiffusionExtraTimes
    from physicsnemo.models.dit import DiT
    from physicsnemo.models.mlp import FullyConnected

    class ExtendedDiTAdapter(Module):
        def __init__(self):
            super().__init__()
            self.backbone = DiT(
                input_size=8, in_channels=2, out_channels=2,
                patch_size=4, hidden_size=32, depth=1, num_heads=4,
            )
            self.r_embed = FullyConnected(
                in_features=1, layer_size=8, out_features=1, num_layers=1,
            )
            self.logvar_head = FullyConnected(
                in_features=1, layer_size=8, out_features=1, num_layers=1,
            )

        def forward(self, x, t, condition=None):
            t, extra_times = t
            # DiT takes one time tensor, so combine the two times in the adapter.
            r_embed = self.r_embed(extra_times.r[:, None]).squeeze(-1)
            prediction = self.backbone(x, t + r_embed, condition)
            logvar = self.logvar_head(t[:, None]).squeeze(-1)
            return prediction, DiffusionExtraOutputs(
                logvar=logvar, batch_size=[x.shape[0]],
            )

    model = ExtendedDiTAdapter()
    x = torch.randn(2, 2, 8, 8)
    t = torch.ones(2)
    times = (t, DiffusionExtraTimes(r=torch.zeros(2), batch_size=[2]))
    prediction, extra_outputs = model(x, times)
    assert prediction.shape == x.shape
    assert extra_outputs.logvar.shape == t.shape

This adapter illustrates one way to embed the reserved ``r`` time and compute
the reserved ``logvar`` output. Other implementations can use different
architectures. Its ``forward`` method implements
:class:`~physicsnemo.diffusion.DiffusionModel` and works with utilities that
support both extra times and extra outputs.

.. _backbone_external_models:

Using External or Custom Models
-------------------------------

The framework is not limited to PhysicsNeMo-native backbones. Any
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

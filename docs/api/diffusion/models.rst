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


Extra diffusion times and outputs
--------------------------------

Most diffusion and flow-matching models take one diffusion time tensor and
return one prediction tensor. These calls remain unchanged. For models that
need extra values, a tuple pairs the primary tensor with a named container.

Use :class:`DiffusionExtraTimes` for the destination time ``r`` in MeanFlow.
Use :class:`DiffusionExtraOutputs` for the learned loss weight ``logvar``
required by sCM or an EDM2-style loss. The primary time and prediction are
always tensors, never ``None``.

.. code-block:: python

    import torch
    from physicsnemo.diffusion import DiffusionExtraOutputs, DiffusionExtraTimes

    t = torch.full((2,), 0.8)
    times = (t, DiffusionExtraTimes(r=torch.full((2,), 0.2), batch_size=[2]))
    prediction = torch.zeros(2, 3, 5, 7)
    output = (prediction, DiffusionExtraOutputs(logvar=torch.zeros(2), batch_size=[2]))

    current_time, extra_times = times
    prediction, extra_outputs = output

The primary time and ``r`` both have shape :math:`(B,)` and use the same
scheduler coordinates. The output prediction has the noisy state's shape.
``logvar`` has shape :math:`(B,)`. It controls adaptive loss weighting,
not diffusion transition variance. Both containers have batch size ``[B]``.

Affine preconditioners encode both ``t`` and ``r`` and apply their affine
formula only to the prediction. The learned loss weight keeps its value and
gradients. Conditioning wrappers forward the time argument and model output
unchanged when the wrapped backbone supports these formats.

Existing DSM and flow-matching losses, denoisers, and samplers still use
single time and prediction tensors. If a model returns auxiliary outputs,
select its prediction before passing it to a tensor-only API:

.. code-block:: python

    def predictor(x, t):
        prediction, _ = model(x, t)
        return prediction

Both containers inherit from ``tensordict.TensorClass` and can be subclassed.
PhysicsNeMo currently interprets only the reserved ``r`` and ``logvar``
fields. Models and wrappers must define how to handle any extra fields.
General support for custom fields remains future work.

.. autoclass:: DiffusionExtraTimes

.. autoclass:: DiffusionExtraOutputs


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

.. _diffusion_unets:

Diffusion UNets
===============

This page documents the UNet family of backbone architectures for diffusion
models.  These are built-in architectures specialized for diffusion on
structured 2D domains (images, spatial fields).  For other domains or
architectures, see the :ref:`DiT backbone <dit_model>`, or use any model from
the :doc:`PhysicsNeMo model zoo <../../api_models>` or external libraries as
described in :ref:`Model Backbones <diffusion_model_backbones>`.

All models on this page are based on the
:class:`~physicsnemo.core.module.Module` class.

.. important::

    These UNet backbones do **not** implement the
    :class:`~physicsnemo.diffusion.DiffusionModel` protocol directly.  Their
    forward signatures differ (e.g.,
    ``forward(x, noise_labels, class_labels, augment_labels)``).  To use them
    with :ref:`preconditioners <diffusion_preconditioners>`,
    :ref:`losses <diffusion_metrics>`, and
    :ref:`samplers <diffusion_samplers>`, wrap them with a thin adapter.
    See the :ref:`adapter examples <diffusion_unet_adapter_example>` below.


.. _diffusion_unet_api:

SongUNet --- The Primary Backbone
---------------------------------

The :class:`~physicsnemo.models.diffusion_unets.SongUNet` is the primary UNet
backbone.  It is a highly configurable multi-resolution architecture that
supports both conditional and unconditional modeling.

Its latent state :math:`\mathbf{x}` is a tensor of shape
:math:`(B, C, H, W)`, where :math:`B` is the batch size, :math:`C` is the
number of channels, and :math:`H` and :math:`W` are the height and width.
The model is always conditional on the noise level, and can additionally be
conditioned on vector-valued class labels and/or images.

The model is organized into *levels*, whose number is determined by
``len(channel_mult)``, and each level operates at half the resolution of the
previous level.  Each level is composed of a sequence of UNet blocks, that
optionally contain self-attention layers, as controlled by the
``attn_resolutions`` parameter.

Here we create a ``SongUNet`` with three levels, that applies self-attention
at levels one and two.  The model is unconditional (*that is,* it is not
conditioned on any class labels or images, but is still conditional on the
noise level, as is standard practice for diffusion models).

.. code:: python

    import torch
    from physicsnemo.models.diffusion_unets import SongUNet

    B, C_x, res = 3, 6, 40   # Batch size, channels, and resolution of the latent state

    model = SongUNet(
        img_resolution=res,
        in_channels=C_x,
        out_channels=C_x,  # No conditioning on image: number of output channels is the same as the input channels
        label_dim=0,  # No conditioning on vector-valued class labels
        augment_dim=0,
        model_channels=64,
        channel_mult=[1, 2, 3],  # 3-levels UNet with 64, 128, and 192 channels at each level, respectively
        num_blocks=4,  # 4 UNet blocks at each level
        attn_resolutions=[20, 10],  # Attention is applied at level 1 (resolution 20x20) and level 2 (resolution 10x10)
    )

    x = torch.randn(B, C_x, res, res)  # Latent state
    noise_labels = torch.randn(B)  # Noise level for each sample

    # The feature map resolution is 40 at level 0, 20 at level 1, and 10 at level 2
    out = model(x, noise_labels, None)
    print(out.shape)  # Shape: (B, C_x, res, res), same as the latent state

    # The same model can be used on images of different resolution
    # Note: the attention is still applied at levels 1 and 2
    x_32 = torch.randn(B, C_x, 32, 32)  # Lower resolution latent state
    out_32 = model(x_32, noise_labels, None)  # None means no conditioning on class labels
    print(out_32.shape)  # Shape: (B, C_x, 32, 32), same as the latent state

.. _example_song_unet_conditional:

The unconditional ``SongUNet`` can be extended to be conditional on class labels and/or
images. Conditioning on images is performed by channel-wise concatenation of the image
to the latent state :math:`\mathbf{x}` before passing it to the model. The model does not perform
conditioning on images internally, and this operation is left to the user. For
conditioning on class labels (or any vector-valued quantity whose dimension is ``label_dim``),
the model internally generates embeddings for the class labels
and adds them to intermediate activations within the UNet blocks. Here we
extend the previous example to be conditional on a 16-dimensional vector-valued
class label and a 3-channel image.

.. code:: python

    import torch
    from physicsnemo.models.diffusion_unets import SongUNet

    B, C_x, res = 3, 10, 40
    C_cond = 3

    model = SongUNet(
        img_resolution=res,
        in_channels=C_x + C_cond,  # Conditioning on an image with C_cond channels
        out_channels=C_x,  # Output channels: only those of the latent state
        label_dim=16,  # Conditioning on 16-dimensional vector-valued class labels
        augment_dim=0,
        model_channels=64,
        channel_mult=[1, 2, 2],
        num_blocks=4,
        attn_resolutions=[20, 10],
    )

    x = torch.randn(B, C_x, res, res)  # Latent state
    cond = torch.randn(B, C_cond, res, res)  # Conditioning image
    x_cond = torch.cat([x, cond], dim=1)  # Channel-wise concatenation of the conditioning image before passing to the model
    noise_labels = torch.randn(B)
    class_labels = torch.randn(B, 16)  # Conditioning on vector-valued class labels

    out = model(x_cond, noise_labels, class_labels)
    print(out.shape)  # Shape: (B, C_x, res, res), same as the latent state


.. _diffusion_unet_adapter_example:

Using UNet Backbones with the Diffusion Framework
--------------------------------------------------

Because these UNet backbones have their own forward signature, they need to be
adapted to the :class:`~physicsnemo.diffusion.DiffusionModel` protocol before
they can be used with :ref:`preconditioners <diffusion_preconditioners>`,
:ref:`losses <diffusion_metrics>`, and :ref:`samplers <diffusion_samplers>`.

The simplest approach is a thin adapter class:

.. code:: python

    import torch
    from physicsnemo.core import Module
    from physicsnemo.models.diffusion_unets import SongUNet
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner

    class SongUNetAdapter(Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.net = SongUNet(**kwargs)

        def forward(self, x, t, condition=None):
            return self.net(x, noise_labels=t, class_labels=condition)

    backbone = SongUNetAdapter(
        img_resolution=64, in_channels=3, out_channels=3,
        model_channels=64, channel_mult=[1, 2, 2], num_blocks=2,
    )

    # The adapter satisfies DiffusionModel, so it can be used with preconditioners
    precond = EDMPreconditioner(backbone, sigma_data=0.5)


DhariwalUNet
------------

The :class:`~physicsnemo.models.diffusion_unets.DhariwalUNet` is an
alternative UNet backbone that can be used interchangeably with ``SongUNet``.
It follows the same adapter pattern described above.


.. _diffusion_specialized_architectures:

Lead-Time Aware Models
----------------------

In many diffusion applications, the latent state is time-dependent, and the
diffusion process should account for the time-dependence of the latent state.
For instance, a *forecast* model could provide latent states :math:`\mathbf{x}(T)` (current time),
:math:`\mathbf{x}(T + \Delta t)` (one time step forward), ..., up to :math:`\mathbf{x}(T + K \Delta t)`
(K time steps forward). Such prediction horizons are called *lead-times* (a term
adopted from the weather and climate forecasting community) and we want to apply
diffusion to each of these latent states while accounting for their associated
lead-time information.

PhysicsNeMo provides a specialized architecture
:class:`~physicsnemo.models.diffusion_unets.SongUNetPosLtEmbd` that implements
lead-time aware models.  It extends ``SongUNet`` with learnable positional
embeddings and lead-time embeddings. In its forward pass, the model uses the
``lead_time_label`` parameter to retrieve the associated lead-time embeddings
and conditions the diffusion process on those with a channel-wise concatenation
to the latent state before the first UNet block.

This is the recommended architecture for lead-time aware diffusion problems
such as weather forecasting.

Here we show an example with lead-time information.
We assume that we have a batch of three latent states at times :math:`T + 2 \Delta t`
(two time intervals forward), :math:`T + 0 \Delta t` (current time),
and :math:`T + \Delta t` (one time interval forward). The associated lead-time labels are
``[2, 0, 1]``. In addition, the ``SongUNetPosLtEmbd`` model has the ability to
predict probabilities for some channels of the latent state, specified by the
``prob_channels`` parameter. Here we assume that channels one and three are
probability (that is, classification) outputs, while other channels are regression
outputs.

.. code:: python

    import torch
    from physicsnemo.models.diffusion_unets import SongUNetPosLtEmbd

    B, C_x, res = 3, 10, 40
    C_cond = 3
    C_PE = 8
    lead_time_steps = 3  # Maximum supported lead-time is 2 * dt
    C_LT = 6  # 6 channels for each lead-time embeddings

    # Create a SongUNet with a lead-time embedding grid of shape
    # (lead_time_steps, C_lt_emb, res, res)
    model = SongUNetPosLtEmbd(
        img_resolution=res,
        in_channels=C_x + C_cond + C_PE + C_LT,  # in_channels must include the number of channels in lead-time grid
        out_channels=C_x,
        label_dim=16,
        augment_dim=0,
        model_channels=64,
        channel_mult=[1, 2, 2],
        num_blocks=4,
        attn_resolutions=[10, 5],
        gridtype="learnable",
        N_grid_channels=C_PE,
        lead_time_channels=C_LT,
        lead_time_steps=lead_time_steps,  # Maximum supported lead-time horizon
        prob_channels=[1, 3],  # Channels 1 and 3 fromn the latent state are probability outputs
    )

    x = torch.randn(B, C_x, res, res)  # Latent state at times T+2*dt, T+0*dt, and T + 1*dt
    cond = torch.randn(B, C_cond, res, res)
    x_cond = torch.cat([x, cond], dim=1)
    noise_labels = torch.randn(B)
    class_labels = torch.randn(B, 16)
    lead_time_label = torch.tensor([2, 0, 1])  # Lead-time labels for each sample

    # The model internally extracts the lead-time embeddings corresponding to the
    # lead-time labels 2, 0, 1 and concatenates them to the input x_cond before the first
    # UNet block. In training mode, the model outputs logits for channels 1 and 3.
    out = model(x_cond, noise_labels, class_labels, lead_time_label=lead_time_label)
    print(out.shape)  # Shape: (B, C_x, res, res), same as the latent state

    # If eval mode the model outputs probabilities for channels 1 and 3
    model.eval()
    out = model(x_cond, noise_labels, class_labels, lead_time_label=lead_time_label)

.. note::

   - The ``SongUNetPosLtEmbd`` *is not* an autoregressive model that performs a rollout
     to produce future predictions. From the point of view of the ``SongUNetPosLtEmbd``,
     the lead-time information is *frozen*. The lead-time dependent latent state :math:`\mathbf{x}`
     might however be produced by such an autoregressive or rollout model.
   - The ``SongUNetPosLtEmbd`` model cannot be scaled to very long lead-time
     horizons (controlled by the ``lead_time_steps`` parameter). This is because
     the lead-time embeddings are represented by a grid of learnable parameters of
     shape ``(lead_time_steps, C_LT, res, res)``. For very long lead-time, the
     size of this grid of embeddings becomes prohibitively large.
   - In a given input batch ``x``, the associated lead-times might be not necessarily
     consecutive or in order. They do not even need to originate from the same forecast
     trajectory. For example, the lead-time labels might be ``[0, 1, 2]`` instead of
     ``[2, 0, 1]``, or even ``[2, 2, 1]``.


Positional Embeddings (SongUNetPosEmbd)
---------------------------------------

The :class:`~physicsnemo.models.diffusion_unets.SongUNetPosEmbd` extends
``SongUNet`` with learnable positional embeddings.  It was originally designed
for multi-diffusion (patch-based diffusion), where each patch needs to be
informed of its position in the global domain.

.. note::

    ``SongUNetPosEmbd`` bakes multi-diffusion logic directly into the
    architecture.  For new projects, the recommended approach is to use the
    :ref:`multi-diffusion APIs <diffusion_multi_diffusion>`, which decouple
    patching from the backbone and can be combined with *any* architecture
    (UNet, DiT, or custom).  ``SongUNetPosEmbd`` remains available for
    backward compatibility and for use cases where integrated positional
    embeddings are specifically desired.


API Reference
-------------

:code:`SongUNet`
~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.models.diffusion_unets.SongUNet
    :show-inheritance:
    :members:
    :exclude-members: forward

:code:`DhariwalUNet`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.models.diffusion_unets.DhariwalUNet
    :show-inheritance:
    :members:
    :exclude-members: forward

:code:`SongUNetPosEmbd`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.models.diffusion_unets.SongUNetPosEmbd
    :show-inheritance:
    :members:
    :exclude-members: forward

:code:`SongUNetPosLtEmbd`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.models.diffusion_unets.SongUNetPosLtEmbd
    :show-inheritance:
    :members:
    :exclude-members: forward

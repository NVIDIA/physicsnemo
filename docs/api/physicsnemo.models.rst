
PhysicsNeMo Models
===================

.. automodule:: physicsnemo.models
.. currentmodule:: physicsnemo.models

Basics
------

PhysicsNeMo contains its own Model class for constructing neural networks. This model class
is built on top of PyTorch's ``nn.Module`` and can be used interchangeably within the
PyTorch ecosystem. Using PhysicsNeMo models allows you to leverage various features of
PhysicsNeMo aimed at improving performance and ease of use. These features include, but are
not limited to, model zoo, automatic mixed-precision, CUDA Graphs, and easy checkpointing.
We discuss each of these features in the following sections.

Model Zoo
---------

PhysicsNeMo contains several optimized, customizable and easy-to-use models.
These include some very general models like Fourier Neural Operators (FNOs),
ResNet, and Graph Neural Networks (GNNs) as well as domain-specific models like
Deep Learning Weather Prediction (DLWP) and Spherical Fourier Neural Operators (SFNO).

For a list of currently available models, please refer the `models on GitHub <https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/models>`_.

Below are some simple examples of how to use these models.

.. code:: python

    >>> import torch
    >>> from physicsnemo.models.mlp.fully_connected import FullyConnected
    >>> model = FullyConnected(in_features=32, out_features=64)
    >>> input = torch.randn(128, 32)
    >>> output = model(input)
    >>> output.shape
    torch.Size([128, 64])

.. code:: python

    >>> import torch
    >>> from physicsnemo.models.fno.fno import FNO
    >>> model = FNO(
            in_channels=4,
            out_channels=3,
            decoder_layers=2,
            decoder_layer_size=32,
            dimension=2,
            latent_channels=32,
            num_fno_layers=2,
            padding=0,
        )
    >>> input = torch.randn(32, 4, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 3, 32, 32])

How to write your own PhysicsNeMo model
--------------------------------------

There are a few different ways to construct a PhysicsNeMo model. If you are a seasoned
PyTorch user, the easiest way would be to write your model using the optimized layers and
utilities from PhysicsNeMo or Pytorch. Let's take a look at a simple example of a UNet model
first showing a simple PyTorch implementation and then a PhysicsNeMo implementation that
supports CUDA Graphs and Automatic Mixed-Precision.

.. code:: python

    import torch.nn as nn

    class UNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1):
            super(UNet, self).__init__()

            self.enc1 = self.conv_block(in_channels, 64)
            self.enc2 = self.conv_block(64, 128)

            self.dec1 = self.upconv_block(128, 64)
            self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        def conv_block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        def upconv_block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x1 = self.enc1(x)
            x2 = self.enc2(x1)
            x = self.dec1(x2)
            return self.final(x)

Now we show this model rewritten in PhysicsNeMo. First, let us subclass the model from
``physicsnemo.Module`` instead of ``torch.nn.Module``. The
``physicsnemo.Module`` class acts like a direct replacement for the
``torch.nn.Module`` and provides additional functionality for saving and loading
checkpoints, etc. Refer to the API docs of ``physicsnemo.Module`` for further
details. Additionally, we will add metadata to the model to capture the optimizations
that this model supports. In this case we will enable CUDA Graphs and Automatic Mixed-Precision.

.. code:: python

    from dataclasses import dataclass
    import physicsnemo
    import torch.nn as nn

    @dataclass
    class UNetMetaData(physicsnemo.ModelMetaData):
        name: str = "UNet"
        # Optimization
        jit: bool = True
        cuda_graphs: bool = True
        amp_cpu: bool = True
        amp_gpu: bool = True

    class UNet(physicsnemo.Module):
        def __init__(self, in_channels=1, out_channels=1):
            super(UNet, self).__init__(meta=UNetMetaData())

            self.enc1 = self.conv_block(in_channels, 64)
            self.enc2 = self.conv_block(64, 128)

            self.dec1 = self.upconv_block(128, 64)
            self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        def conv_block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        def upconv_block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x1 = self.enc1(x)
            x2 = self.enc2(x1)
            x = self.dec1(x2)
            return self.final(x)

Now that we have our PhysicsNeMo model, we can make use of these optimizations using the
``physicsnemo.utils.StaticCaptureTraining`` decorator. This decorator will capture the
training step function and optimize it for the specified optimizations.

.. code:: python

    import torch
    from physicsnemo.utils import StaticCaptureTraining

    model = UNet().to("cuda")
    input = torch.randn(8, 1, 128, 128).to("cuda")
    output = torch.zeros(8, 1, 64, 64).to("cuda")

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create training step function with optimization wrapper
    # StaticCaptureTraining calls `backward` on the loss and
    # `optimizer.step()` so you don't have to do that
    # explicitly.
    @StaticCaptureTraining(
        model=model,
        optim=optim,
        cuda_graph_warmup=11,
    )
    def training_step(invar, outvar):
        predvar = model(invar)
        loss = torch.sum(torch.pow(predvar - outvar, 2))
        return loss

    # Sample training loop
    for i in range(20):
        # In place copy of input and output to support cuda graphs
        input.copy_(torch.randn(8, 1, 128, 128).to("cuda"))
        output.copy_(torch.zeros(8, 1, 64, 64).to("cuda"))

        # Run training step
        loss = training_step(input, output)

For the simple model above, you can observe ~1.1x speed-up due to CUDA Graphs and AMP.
The speed-up observed changes from model to model and is typically greater for more
complex models.

.. note::
    The ``ModelMetaData`` and ``physicsnemo.Module`` do not make the model
    support CUDA Graphs, AMP, etc. optimizations automatically. The user is responsible
    to write the model code that enables each of these optimizations.
    Models in the PhysicsNeMo Model Zoo are written to support many of these optimizations
    and checked against PhysicsNeMo's CI to ensure that they work correctly.

.. note::
    The ``StaticCaptureTraining`` decorator is still under development and may be
    refactored in the future.


.. _physicsnemo-models-from-torch:

Converting PyTorch Models to PhysicsNeMo Models
----------------------------------------------

In the above example we show constructing a PhysicsNeMo model from scratch. However, you
can also convert existing PyTorch models to PhysicsNeMo models in order to leverage
PhysicsNeMo features. To do this, you can use the ``Module.from_torch`` method as shown
below.

.. code:: python

    from dataclasses import dataclass
    import physicsnemo
    import torch.nn as nn

    class TorchModel(nn.Module):
        def __init__(self):
            super(TorchModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = self.conv1(x)
            return self.conv2(x)

    @dataclass
    class ConvMetaData(ModelMetaData):
        name: str = "UNet"
        # Optimization
        jit: bool = True
        cuda_graphs: bool = True
        amp_cpu: bool = True
        amp_gpu: bool = True

    PhysicsNeMoModel = physicsnemo.Module.from_torch(TorchModel, meta=ConvMetaData())




.. _saving-and-loading-physicsnemo-models:

Saving and Loading PhysicsNeMo Models
-------------------------------------

As mentioned above, PhysicsNeMo models are interoperable with PyTorch models. This means that
you can save and load PhysicsNeMo models using the standard PyTorch APIs however, we provide
a few additional utilities to make this process easier. A key challenge in saving and
loading models is keeping track of the model metadata such as layer sizes, etc. PhysicsNeMo
models can be saved with this metadata to a custom ``.mdlus`` file. These files allow
for easy loading and instantiation of the model. We show two examples of this below.
The first example shows saving and loading a model from an already instantiated model.

.. code:: python

    >>> from physicsnemo.models.mlp.fully_connected import FullyConnected
    >>> model = FullyConnected(in_features=32, out_features=64)
    >>> model.save("model.mdlus") # Save model to .mdlus file
    >>> model.load("model.mdlus") # Load model weights from .mdlus file from already instantiated model
    >>> model
    FullyConnected(
     (layers): ModuleList(
       (0): FCLayer(
         (activation_fn): SiLU()
         (linear): Linear(in_features=32, out_features=512, bias=True)
       )
       (1-5): 5 x FCLayer(
         (activation_fn): SiLU()
         (linear): Linear(in_features=512, out_features=512, bias=True)
       )
     )
     (final_layer): FCLayer(
       (activation_fn): Identity()
       (linear): Linear(in_features=512, out_features=64, bias=True)
     )
   )

The second example shows loading a model from a ``.mdlus`` file without having to
instantiate the model first. We note that in this case we don't know the class or
parameters to pass to the constructor of the model. However, we can still load the
model from the ``.mdlus`` file.

.. code:: python

    >>> from physicsnemo import Module
    >>> fc_model = Module.from_checkpoint("model.mdlus") # Instantiate model from .mdlus file.
    >>> fc_model
    FullyConnected(
     (layers): ModuleList(
       (0): FCLayer(
         (activation_fn): SiLU()
         (linear): Linear(in_features=32, out_features=512, bias=True)
       )
       (1-5): 5 x FCLayer(
         (activation_fn): SiLU()
         (linear): Linear(in_features=512, out_features=512, bias=True)
       )
     )
     (final_layer): FCLayer(
       (activation_fn): Identity()
       (linear): Linear(in_features=512, out_features=64, bias=True)
     )
   )



.. note::
   In order to make use of this functionality, the model must have ``.json`` serializable
   inputs to the ``__init__`` function. It is highly recommended that all PhysicsNeMo
   models be developed with this requirement in mind.

.. note::
   Using ``Module.from_checkpoint`` will not work if the model has any buffers or
   parameters that are registered outside of the model's ``__init__`` function due to
   the above requirement. In that case, one should use ``Module.load``, or ensure 
   that all model parameters and buffers are registered inside ``__init__``.


PhysicsNeMo Model Registry and Entry Points
------------------------------------------

PhysicsNeMo contains a model registry that allows for easy access and ingestion of
models. Below is a simple example of how to use the model registry to obtain a model
class.

.. code:: python

    >>> from physicsnemo.registry import ModelRegistry
    >>> model_registry = ModelRegistry()
    >>> model_registry.list_models()
    ['AFNO', 'DLWP', 'FNO', 'FullyConnected', 'GraphCastNet', 'MeshGraphNet', 'One2ManyRNN', 'Pix2Pix', 'SFNO', 'SRResNet']
    >>> FullyConnected = model_registry.factory("FullyConnected")
    >>> model = FullyConnected(in_features=32, out_features=64)

The model registry also allows exposing models via entry points. This allows for
integration of models into the PhysicsNeMo ecosystem. For example, suppose you have a
package ``MyPackage`` that contains a model ``MyModel``. You can expose this model
to the PhysicsNeMo registry by adding an entry point to your ``toml`` file. For
example, suppose your package structure is as follows:

.. code:: python

    # setup.py

    from setuptools import setup, find_packages

    setup()

.. code:: python

    # pyproject.toml

    [build-system]
    requires = ["setuptools", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "MyPackage"
    description = "My Neural Network Zoo."
    version = "0.1.0"

    [project.entry-points."physicsnemo.models"]
    MyPhysicsNeMoModel = "mypackage.models.MyPhysicsNeMoModel:MyPhysicsNeMoModel"

.. code:: python

   # mypackage/models.py

   import torch.nn as nn
   from physicsnemo.models import Module

   class MyModel(nn.Module):
       def __init__(self):
           super(MyModel, self).__init__()
           self.conv1 = nn.Conv2d(1, 20, 5)
           self.conv2 = nn.Conv2d(20, 20, 5)

       def forward(self, x):
           x = self.conv1(x)
           return self.conv2(x)

   MyPhysicsNeMoModel = Module.from_pytorch(MyModel)


Once this package is installed, you can access the model via the PhysicsNeMo model
registry.


.. code:: python

   >>> from physicsnemo.registry import ModelRegistry
   >>> model_registry = ModelRegistry()
   >>> model_registry.list_models()
   ['MyPhysicsNeMoModel', 'AFNO', 'DLWP', 'FNO', 'FullyConnected', 'GraphCastNet', 'MeshGraphNet', 'One2ManyRNN', 'Pix2Pix', 'SFNO', 'SRResNet']
   >>> MyPhysicsNeMoModel = model_registry.factory("MyPhysicsNeMoModel")


For more information on entry points and potential use cases, see
`this <https://amir.rachum.com/blog/2017/07/28/python-entry-points/>`_ blog post.

.. autosummary::
   :toctree: generated

Fully Connected Network
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: physicsnemo.models.mlp.fully_connected
    :members:
    :show-inheritance:

Fourier Neural Operators
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: physicsnemo.models.fno.fno
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.afno.afno
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.afno.modafno
    :members:
    :show-inheritance:

Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: physicsnemo.models.meshgraphnet.meshgraphnet
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.mesh_reduced.mesh_reduced
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.meshgraphnet.bsms_mgn
    :members:
    :show-inheritance:


Convolutional Networks
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: physicsnemo.models.pix2pix.pix2pix
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.srrn.super_res_net
    :members:
    :show-inheritance:

Recurrent Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: physicsnemo.models.rnn.rnn_one2many
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.rnn.rnn_seq2seq
    :members:
    :show-inheritance:


Diffusion Models
~~~~~~~~~~~~~~~~

PhysicsNeMo diffusion library provides three categories of models, that serve
different purposes. All models are based on the
:class:`~physicsnemo.models.module.Module` class.

    - :ref:`Model backbones <diffusion_architecture_backbones>`:
        Those are highly configurable architectures that can be used as a
        building block for more complex models.

    - :ref:`Specialized architectures <diffusion_specialized_architectures>`:
        Those are models that usually inherit from the model backbones, with
        some specific additional functionalities.

    - :ref:`Application-specific interfaces <diffusion_application_specific_interfaces>`:
        These Modules are not truly architectures, but rather wrappers around
        the model backbones or specialized architectures. Their intent is to
        provide a more user-friendly interface for specific applications.

In addition of these model architectures, PhysicsNeMo provides
:ref:`diffusion preconditioners <diffusion_preconditioners>`, which are
essentially wrappers around model architectures, that rescale the inputs and
outputs of diffusion models to improve their performance.

.. _diffusion_architecture_backbones:

Architecture Backbones
^^^^^^^^^^^^^^^^^^^^^^

Diffusion model backbones are highly configurable architectures that can be used
as a building block for more complex models. Backbones support
both conditional and unconditional modeling. Currently, there are two provided
backbones: the SongUNet, as implemented in the
:class:`~physicsnemo.models.diffusion.song_unet.SongUNet` class and the DhariwalUNet,
as implemented in the :class:`~physicsnemo.models.diffusion.dhariwal_unet.DhariwalUNet`
class. These models were introduced in the papers `Score-based generative modeling through stochastic
differential equations, Song et al. <https://arxiv.org/abs/2011.13456>`_ and
`Diffusion models beat gans on image synthesis, Dhariwal et al.
<https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html>`_.
The PhysicsNeMo implementation of these models follows closely that used in the paper
`Elucidating the Design Space of Diffusion-Based Generative Models, Karras et al.
<https://arxiv.org/abs/2206.00364>`_. The original implementation of these
models can be found in the `EDM repository <https://github.com/NVlabs/edm>`_.

Model backbones can be used as is, such as in in
`the StormCast example <../examples/weather/stormcast/README.rst>`_, but they can also be used as a base class for
more complex models.

One of the most common diffusion backbones for image generation is the
:class:`~physicsnemo.models.diffusion.song_unet.SongUNet`
class. Its latent state :math:`\mathbf{x}` is a tensor of shape :math:`(B, C, H, W)`,
where :math:`B` is the batch size, :math:`C` is the number of channels,
and :math:`H` and :math:`W` are the height and width of the feature map. The
model is conditional on the noise level, and can additionally be conditioned on
vector-valued class labels and/or images. The model is organized into *levels*,
whose number is determined by ``len(channel_mult)``, and each level operates at half the resolution of the
previous level (odd resolutions are rounded down). Each level is composed of a sequence of UNet blocks, that optionally contain
self-attention layers, as controlled by the ``attn_resolutions`` parameter. The feature map resolution
is halved at the first block of each level and then remains constant within the level.

Here we start by creating a ``SongUNet`` model with 3 levels, that applies self-attention
at levels 1 and 2. The model is unconditional, *i.e.* it is not conditioned on any
class labels or images (but is still conditional on the noise level, as it is
standard practice for diffusion models).

.. code:: python

    import torch
    from physicsnemo.models.diffusion import SongUNet

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
    from physicsnemo.models.diffusion import SongUNet

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

.. _diffusion_specialized_architectures:

Specialized Architectures
^^^^^^^^^^^^^^^^^^^^^^^^^

Note that even though backbones can be used as is, some of the examples in
PhysicsNeMo examples use specialized architectures. These specialized architectures
typically inherit from the backbones and implement additional functionalities for specific
applications. For example the `CorrDiff example <../examples/weather/corrdiff/README.rst>`_
uses the specialized architectures :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`
and :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd` to implement
the diffusion model.

Positional embeddings
"""""""""""""""""""""

Multi-diffusion (also called *patch-based* diffusion) is a technique to scale
diffusion models to large domains. The idea is to split the full domain into
patches, and run a diffusion model on each patch in parallel. The generated
patches are then fused back to form the final image. This technique is
particularly useful for domains that are too large to fit into the memory of
a single GPU. The `CorrDiff example <../examples/weather/corrdiff/README.rst>`_
uses patch-based diffusion for weather downscaling on large domains. A key
ingredient in the implementation of patch-based diffusion is the use of a
global spatial grid, that is used to inform each patch with their respective
position in the full domain. The :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`
class implements this functionality by providing multiple methods to encode
global spatial coordinates of the pixels into a *global positional embedding grid*.
In addition of multi-diffusion, spatial positional embeddings have also been
observed to improve the quality of the generated images, even for diffusion models
that operate on the full domain.

The following example shows how to use the specialized architecture
:class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd` to implement a
multi-diffusion model. First, we create a ``SongUNetPosEmbd`` model similar to
the one in :ref:`the conditional SongUnet example <example_song_unet_conditional>`
with a global positional embedding grid of shape ``(C_pos_emb, res, res)``. We
show that the model can be used with the entire latent state (full domain).

.. code:: python

    import torch
    from physicsnemo.models.diffusion import SongUNetPosEmbd

    B, C_x, res = 3, 10, 40
    C_cond = 3
    C_PE = 8  # Number of channels in the positional embedding grid

    # Create a SongUNet with a global positional embedding grid of shape (C_PE, res, res)
    model = SongUNetPosEmbd(
        img_resolution=res,  # Define the resolution of the global positional embedding grid
        in_channels=C_x + C_cond + C_PE,  # in_channels must include the number of channels in the positional embedding grid
        out_channels=C_x,
        label_dim=16,
        augment_dim=0,
        model_channels=64,
        channel_mult=[1, 2, 2],
        num_blocks=4,
        attn_resolutions=[20, 10],
        gridtype="learnable",  # Use a learnable grid of positional embeddings
        N_grid_channels=C_PE  # Number of channels in the positional embedding grid
    )

    # Can pass the entire latent state to the model
    x_global = torch.randn(B, C_x, res, res)  # Entire latent state
    cond = torch.randn(B, C_cond, res, res)  # Conditioning image
    x_cond = torch.cat([x_global, cond], dim=1)  # Latent state with conditioning image
    noise_labels = torch.randn(B)
    class_labels = torch.randn(B, 16)

    # The model internally concatenates the global positional embedding grid to the
    # input x_cond before the first UNet block.
    # Note: global_index=None means use the entire positional embedding grid
    out = model(x_cond, noise_labels, class_labels, global_index=None)
    print(out.shape)  # Shape: (B, C_x, res, res), same as the latent state

Now we show that the model can be used on local patches of the latent state
(multi-diffusion approach). We manually extract 3 patches from the latent
state. Patches are treated as individual samples, so they are concatenated along
the batch dimension. We also create a global grid of indices ``grid`` that
contains the indices of the pixels in the full domain, and we exctract *the same
3 patches* from the global grid and pass them to the ``global_index``
parameter. The model internally uses ``global_index`` to extract the corresponding
patches from the positional embedding grid and concatenate them to the input
``x_cond_patches`` before the first UNet block. Note that conditional
multi-diffusion still requires each patch to *be conditioned on the entire
conditioning image* ``cond``, which is why we interpolate the conditioning image
to the patch resolution and concatenate it to each individual patch.
In practice it is not necessary to manually extract the patches from the latent
state and the global grid, as PhysicsNeMo provides utilities to help with the
patching operations, in :mod:`~physicsnemo.utils.patching`. For an example of how
to use these utilities, see the `CorrDiff example <../examples/weather/corrdiff/README.rst>`_.

.. code:: python

    # Can pass local patches to the model
    # Create batch of 3 patches from `x_global` with resolution 16x16
    pres = 16  # Patch resolution
    p1 = x_global[0:1, :, :pres, :pres]  # Patch 1
    p2 = x_global[3:4, :, pres:2*pres, pres:2*pres]  # Patch 2
    p3 = x_global[1:2, :, -pres:, pres:2*pres]  # Patch 3
    patches = torch.cat([p1, p2, p3], dim=0)  # Batch of 3 patches

    # Note: the conditioning image needs interpolation (or other operations) to
    # match the patch resolution
    cond1 = torch.nn.functional.interpolate(cond[0:1], size=(pres, pres), mode="bilinear")
    cond2 = torch.nn.functional.interpolate(cond[3:4], size=(pres, pres), mode="bilinear")
    cond3 = torch.nn.functional.interpolate(cond[1:2], size=(pres, pres), mode="bilinear")
    cond_patches = torch.cat([cond1, cond2, cond3], dim=0)

    # Concatenate the patches and the conditioning image
    x_cond_patches = torch.cat([patches, cond_patches], dim=1)

    # Create corresponding global indices for the patches
    Ny, Nx = torch.arange(res).int(), torch.arange(res).int()
    grid = torch.stack(torch.meshgrid(Ny, Nx, indexing="ij"), dim=0)
    idx_patch1 = grid[:, :pres, :pres]  # Global indices for patch 1
    idx_patch2 = grid[:, pres:2*pres, pres:2*pres]  # Global indices for patch 2
    idx_patch3 = grid[:, -pres:, pres:2*pres]  # Global indices for patch 3
    global_index = torch.stack([idx_patch1, idx_patch2, idx_patch3], dim=0)

    # The model internally extracts the corresponding patches from the global
    # positional embedding grid and concatenates them to the input x_cond_patches
    # before the first UNet block.
    out = model(x_cond_patches, noise_labels, class_labels, global_index=global_index)
    print(out.shape)  # Shape: (3, C_x, pres, pres), same as the patches extracted from the latent state

Lead-time aware models
""""""""""""""""""""""

In many diffusion applications, the latent state is time-dependent, and the
diffusion process should account for the time-dependence of the latent state.
For instance, a *forecast* model could provide latent states :math:`\mathbf{x}(T)` (current time),
:math:`\mathbf{x}(T + \Delta t)` (one time step forward), ..., up to :math:`\mathbf{x}(T + K \Delta t)`
(K time steps forward). Such prediction horizons are called *lead-times* (a term
adopted from the weather and climate forecasting community) and we want to apply
diffusion to each of these latent states while accounting for their associated
lead-time information.

PhysicsNeMo provides a specialized architecture
:class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd` that implements
lead-time aware models. This is an extension of the
:class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd` class, and
additionally supports lead-time information. In its forward pass, the model
uses the ``lead_time_label`` parameter to internally retrieve the associated
lead-time embeddings; it then conditions the diffusion process on those with a
channel-wise concatenation to the latent-state before the first UNet block.

Here we show an example extending the previous ones with lead-time information.
We assume that we have a batch of 3 latent states at times :math:`T + 2 \Delta t`
(2 time intervals forward), :math:`T + 0 \Delta t` (current time),
and :math:`T + \Delta t` (1 time interval forward). The associated lead-time labels are
``[2, 0, 1]``. In addition, the ``SongUNetPosLtEmbd`` model has the ability to
predict probabilities for some channels of the latent state, specified by the
``prob_channels`` parameter. Here we assume that channels 1 and 3 are
probability (i.e. classification) outputs, while other channels are regression
outputs.

.. code:: python

    import torch
    from physicsnemo.models.diffusion import SongUNetPosLtEmbd

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
    The ``SongUNetPosLtEmbd`` *is not* an autoregressive model that performs a rollout
    to produce future predictions. From the point of view of the ``SongUNetPosLtEmbd``,
    the lead-time information is *frozen*. The lead-time dependent latent state :math:`\mathbf{x}`
    might however be produced by such an autoregressive/rollout model.

.. note::
    The ``SongUNetPosLtEmbd`` model cannot be scaled to very long lead-time
    horizons (controlled by the ``lead_time_steps`` parameter). This is because
    the lead-time embeddings are represented by a grid of learnable parameters of
    shape ``(lead_time_steps, C_LT, res, res)``. For very long lead-time, the
    size of this grid of embeddings becomes prohibitively large.

.. note::
    In a given input batch ``x``, the associated lead-times might be not necessarily
    consecutive or in order. The do not even need to originate from the same forecast
    trajectory. For example, the lead-time labels might be ``[0, 1, 2]`` instead of ``[2, 0, 1]``,
    or even ``[2, 2, 1]``.

.. _diffusion_application_specific_interfaces:

Application-specific Interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Application-specific interfaces are not true architectures, but rather wrappers
around the model backbones or specialized architectures that provide a more
user-friendly interface for specific applications. Note that not all these
classes are true diffusion models, but can also be used in conjunction with
diffusion models. For instance, the CorrDiff example in
`CorrDiff example <../examples/weather/corrdiff/README.rst>`_ uses the :class:`~physicsnemo.models.diffusion.unet.UNet`
class to implement a regression model.


.. autoclass:: physicsnemo.models.diffusion.song_unet.SongUNet
    :show-inheritance:

.. autoclass:: physicsnemo.models.diffusion.dhariwal_unet.DhariwalUNet
    :show-inheritance:


.. autoclass:: physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd
    :show-inheritance:
    :members: positional_embedding_indexing, positional_embedding_selector

.. autoclass:: physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd
    :show-inheritance:
    :members: positional_embedding_indexing, positional_embedding_selector


.. autoclass:: physicsnemo.models.diffusion.unet.UNet
    :show-inheritance:
    :members: amp_mode

.. _diffusion_preconditioners:

Diffusion Preconditioners
^^^^^^^^^^^^^^^^^^^^^^^^^

Preconditioning is an essential technique to improve the performance of
diffusion models. It consists in scaling the latent state and the noise
level that are passed to a network. Some preconditioning also requires to
re-scale the output of the network. PhysicsNeMo provides a set of preconditioning
classes that are wrappers around backbones or specialized architectures.

.. automodule:: physicsnemo.models.diffusion.preconditioning
    :members:
    :show-inheritance:


Weather / Climate Models
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: physicsnemo.models.dlwp.dlwp
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.dlwp_healpix.HEALPixRecUNet
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.graphcast.graph_cast_net
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.fengwu.fengwu
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.pangu.pangu
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.swinvrnn.swinvrnn
    :members:
    :show-inheritance:

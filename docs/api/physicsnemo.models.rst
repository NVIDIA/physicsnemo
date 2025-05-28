
PhysicsNeMo Models
===================

.. automodule:: physicsnemo.models
.. currentmodule:: physicsnemo.models

Basics
^^^^^^

PhysicsNeMo contains its own Model class for constructing neural networks. This model class
is built on top of PyTorch's ``nn.Module`` and can be used interchangeably within the
PyTorch ecosystem. Using PhysicsNeMo models allows you to leverage various features of
PhysicsNeMo aimed at improving performance and ease of use. These features include, but are
not limited to, model zoo, automatic mixed-precision, CUDA Graphs, and easy checkpointing.
We discuss each of these features in the following sections.

Model Zoo
^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a few different ways to construct a PhysicsNeMo model. If you are a seasoned
PyTorch user, the easiest way would be to write your model using the optimized layers and
utilities from PhysicsNeMo or Pytorch. Lets take a look at a simple example of a UNet model
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

Now we show this model rewritten in PhysicsNeMo. First, let's subclass the model from
``physicsnemo.Module`` instead of ``torch.nn.Module``. The
``physicsnemo.Module`` class acts like a direct replacement for the
``torch.nn.Module`` and provides additional functionality for saving and loading
checkpoints, etc. Refer to the API docs of ``physicsnemo.Module`` for further
details. Additionally we will add metadata to the model to capture the optimizations
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the above example we show constructing a PhysicsNeMo model from scratch. However you
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   In order to make use of this functionality, the model must have json serializable
   inputs to the ``__init__`` function. It is highly recommended that all PhysicsNeMo
   models be developed with this requirement in mind.

.. note::
   Using ``Module.from_checkpoint`` will not work if the model has any buffers or
   parameters that are registered outside of the model's ``__init__`` function due to
   the above requirement. In that case, one should use ``Module.load``, or ensure 
   that all model parameters and buffers are registered inside ``__init__``.


PhysicsNeMo Model Registry and Entry Points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   from physicsnemo.models import Model

   class MyModel(nn.Module):
       def __init__(self):
           super(MyModel, self).__init__()
           self.conv1 = nn.Conv2d(1, 20, 5)
           self.conv2 = nn.Conv2d(20, 20, 5)

       def forward(self, x):
           x = self.conv1(x)
           return self.conv2(x)

   MyPhysicsNeMoModel = Model.from_pytorch(MyModel)


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
-----------------------

.. automodule:: physicsnemo.models.mlp.fully_connected
    :members:
    :show-inheritance:

Fourier Neural Operators
------------------------

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
---------------------

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
-----------------------

.. automodule:: physicsnemo.models.pix2pix.pix2pix
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.srrn.super_res_net
    :members:
    :show-inheritance:

Recurrent Neural Networks
-------------------------

.. automodule:: physicsnemo.models.rnn.rnn_one2many
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.rnn.rnn_seq2seq
    :members:
    :show-inheritance:


Weather / Climate Models
-------------------------

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


Diffusion Model
---------------

.. automodule:: physicsnemo.models.diffusion.dhariwal_unet
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.diffusion.song_unet
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.models.diffusion.unet
    :members:
    :show-inheritance:

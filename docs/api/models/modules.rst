.. _modules:

PhysicsNeMo Modules
===================

.. automodule:: physicsnemo.models
.. currentmodule:: physicsnemo.models

Basics
------

PhysicsNeMo contains its own Module class for constructing neural networks. This class
is built on top of PyTorch's ``nn.Module`` and can be used interchangeably within the
PyTorch ecosystem. Using PhysicsNeMo modules allows you to leverage several features
aimed at improving ease of use, including:

- Portable :ref:`checkpointing <saving-and-loading-physicsnemo-models>` via
  ``.mdlus`` files.
- A global :ref:`model registry <physicsnemo-model-registry>` for discovering
  and retrieving model classes by name.
- :ref:`Backward compatibility <backward-compatibility>` tooling for evolving
  model classes without breaking existing checkpoints.

In addition, PhysicsNeMo ships a :ref:`model zoo <model-zoo>` of optimized
architectures that can be used off-the-shelf or composed into larger models.
We discuss each of these features in the following sections. For the full
programmatic interface, see the :ref:`modules-api-reference` section at the
bottom of this page.


.. _model-zoo:

Model Zoo
---------

PhysicsNeMo ships several optimized, customizable and easy-to-use model architectures.
These include general-purpose models like Fourier Neural Operators (FNOs),
ResNet, and Graph Neural Networks (GNNs) as well as domain-specific models like
Deep Learning Weather Prediction (DLWP) and Spherical Fourier Neural Operators (SFNO).
Many of these architectures include built-in performance optimizations.

For a list of currently available models, please refer to the `models on GitHub <https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/models>`_.

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

How to Write Your Own PhysicsNeMo Model
---------------------------------------

There are a few different ways to construct a PhysicsNeMo model. If you are a seasoned
PyTorch user, the easiest way would be to write your model using the optimized layers and
utilities from PhysicsNeMo or PyTorch. Let's take a look at a simple example of a UNet model
first showing a simple PyTorch implementation and then the same model rewritten as a
PhysicsNeMo module.

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


To turn this into a PhysicsNeMo model, the only required change is to inherit
from :class:`~physicsnemo.core.module.Module` instead of ``torch.nn.Module``:

.. code:: python

    import physicsnemo

    class UNet(physicsnemo.Module):  # the only change
        def __init__(self, in_channels=1, out_channels=1):
            super().__init__()
            # ... same code as above ...


.. _physicsnemo-models-from-torch:

Converting PyTorch Models to PhysicsNeMo Models
-----------------------------------------------

In the above example we show constructing a PhysicsNeMo model from scratch. However, you
can also convert existing PyTorch models to PhysicsNeMo models in order to leverage
PhysicsNeMo features. To do this, use the
:meth:`~physicsnemo.core.module.Module.from_torch` class method as shown below.

.. code:: python

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

    # from_torch returns a *class* (not an instance).
    # By default, the new class name matches the PyTorch class name ('TorchModel').
    PhysicsNeMoModel = physicsnemo.Module.from_torch(TorchModel)
    PhysicsNeMoModel.__name__  # 'TorchModel'

    # You can override the class name with the ``name`` parameter.
    PhysicsNeMoModel = physicsnemo.Module.from_torch(TorchModel, name="MyConvNet")
    PhysicsNeMoModel.__name__  # 'MyConvNet'

    # Once instantiated, the result is a PhysicsNeMo Module whose class name
    # is the one specified above.
    model = PhysicsNeMoModel()

Optionally, you can register the converted class in the
:ref:`model registry <physicsnemo-model-registry>` by passing
``register=True``. This is useful if you later want to load the model from a
checkpoint by name, but it is not strictly required —
:meth:`~physicsnemo.core.module.Module.from_checkpoint` can also resolve the
class by its module path. You can achieve the same result by calling
:meth:`~physicsnemo.core.registry.ModelRegistry.register` directly.

.. code:: python

    PhysicsNeMoModel = physicsnemo.Module.from_torch(
        TorchModel, name="MyConvNet", register=True
    )

**Importing Models from Third-Party Libraries**

The same approach works for models defined in third-party libraries such as
`timm <https://huggingface.co/docs/timm>`_ or any other library that provides
``torch.nn.Module`` subclasses:

.. code:: python

    import physicsnemo
    import timm

    # Load a pre-trained model from timm
    TimmResNet = timm.create_model("resnet18", pretrained=False).__class__

    # Convert to a PhysicsNeMo Module class
    PNMResNet = physicsnemo.Module.from_torch(
        TimmResNet, name="TimmResNet18", register=True
    )

    # Instantiate as a PhysicsNeMo model
    model = PNMResNet()


.. _saving-and-loading-physicsnemo-models:

Saving and Loading PhysicsNeMo Models
-------------------------------------

PhysicsNeMo models are interoperable with PyTorch models. You can save and load
them using the standard PyTorch APIs, but PhysicsNeMo also provides utilities
that save the full constructor arguments alongside the weights into a single
``.mdlus`` checkpoint file. This means a ``.mdlus`` file is fully
self-describing: it contains everything needed to re-create the model without
knowing its class or constructor arguments ahead of time.

There are two ways to load from a ``.mdlus`` checkpoint:

- **Into an already instantiated model**: use
   :meth:`~physicsnemo.core.module.Module.save` and
   :meth:`~physicsnemo.core.module.Module.load`, just like a regular PyTorch
   ``nn.Module``. This only transfers the weights (``state_dict``).
- **Instantiate and load in one step**: use
   :meth:`~physicsnemo.core.module.Module.from_checkpoint`. This resolves the
   class, creates the instance from the saved constructor arguments, and loads
   the weights. When the class is known ahead of time, calling
   ``KnownClass.from_checkpoint(...)`` is preferred over
   ``Module.from_checkpoint(...)`` because it avoids a class-resolution step.

**Example 1: save and load into an existing instance:**

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

**Example 2: instantiate directly from a checkpoint:**

In this case we don't need to know the class or the constructor arguments. The
``.mdlus`` file contains all the information needed to instantiate the model.

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
   - In order to use :meth:`~physicsnemo.core.module.Module.from_checkpoint`,
     the model must have ``.json``-serializable inputs to the ``__init__``
     function. The only exception is when the argument is itself a
     :class:`~physicsnemo.core.module.Module` instance. In this case, it is
     possible to construct, save and load nested Modules with multiple levels
     of nesting and/or multiple :class:`~physicsnemo.core.module.Module`
     instances at each level. See :ref:`constructing-nested-modules` for
     details. It is highly recommended that all PhysicsNeMo models be
     developed with this requirement in mind.
   - Using ``Module.from_checkpoint`` will not work if the model has any
     buffers or parameters that are registered outside of the model's
     ``__init__`` function. In that case, use
     :meth:`~physicsnemo.core.module.Module.load` instead, or ensure that all
     model parameters and buffers are registered inside ``__init__``.

For training-loop checkpointing that also saves optimizer, scheduler, and
scaler state, see :func:`~physicsnemo.utils.checkpoint.save_checkpoint` and
:func:`~physicsnemo.utils.checkpoint.load_checkpoint`.


.. _constructing-nested-modules:

Constructing Nested Modules
----------------------------

PhysicsNeMo supports constructing nested modules where one
:class:`~physicsnemo.core.module.Module` can accept another
:class:`~physicsnemo.core.module.Module` as an argument to its ``__init__``
function. This allows you to build complex, modular architectures while still
benefiting from PhysicsNeMo's checkpointing and model management features.

**Simple Nesting with PhysicsNeMo Modules**

The simplest case is nesting :class:`~physicsnemo.core.module.Module` instances
directly:

.. code:: python

    import torch
    import physicsnemo

    class EncoderModule(physicsnemo.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.encoder = torch.nn.Linear(input_size, hidden_size)
            self.input_size = input_size
            self.hidden_size = hidden_size

        def forward(self, x):
            return self.encoder(x)

    class DecoderModule(physicsnemo.Module):
        def __init__(self, hidden_size, output_size):
            super().__init__()
            self.decoder = torch.nn.Linear(hidden_size, output_size)
            self.hidden_size = hidden_size
            self.output_size = output_size

        def forward(self, x):
            return self.decoder(x)

    class AutoEncoder(physicsnemo.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)

    # Create nested model
    encoder = EncoderModule(input_size=64, hidden_size=32)
    decoder = DecoderModule(hidden_size=32, output_size=64)
    model = AutoEncoder(encoder=encoder, decoder=decoder)

    # Save and load with full structure preserved
    model.save("autoencoder.mdlus")
    loaded_model = physicsnemo.Module.from_checkpoint("autoencoder.mdlus")

**Nesting Converted PyTorch Modules**

You can also nest PyTorch ``nn.Module`` instances, but they must first be
converted to :class:`~physicsnemo.core.module.Module` using
:meth:`~physicsnemo.core.module.Module.from_torch`. All nested
PyTorch modules must be converted:

.. code:: python

    import torch.nn as nn
    import physicsnemo

    # Define PyTorch modules
    class TorchEncoder(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.encoder = nn.Linear(input_size, hidden_size)
            self.input_size = input_size
            self.hidden_size = hidden_size

        def forward(self, x):
            return self.encoder(x)

    class TorchDecoder(nn.Module):
        def __init__(self, hidden_size, output_size):
            super().__init__()
            self.decoder = nn.Linear(hidden_size, output_size)
            self.hidden_size = hidden_size
            self.output_size = output_size

        def forward(self, x):
            return self.decoder(x)

    # Convert to PhysicsNeMo modules
    PNMEncoder = physicsnemo.Module.from_torch(TorchEncoder)
    PNMDecoder = physicsnemo.Module.from_torch(TorchDecoder)

    # Define top-level model
    class AutoEncoder(physicsnemo.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)

    # Create nested model with converted modules
    encoder = PNMEncoder(input_size=64, hidden_size=32)
    decoder = PNMDecoder(hidden_size=32, output_size=64)
    model = AutoEncoder(encoder=encoder, decoder=decoder)

    # Save and load
    model.save("autoencoder.mdlus")
    loaded_model = physicsnemo.Module.from_checkpoint("autoencoder.mdlus")

**What Does NOT Work**

You cannot directly pass a ``torch.nn.Module`` instance to a
:class:`~physicsnemo.core.module.Module`'s ``__init__`` without converting it first:

.. code:: python

    # This will NOT work and raise an error during save/load:
    class AutoEncoder(physicsnemo.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder  # encoder is a torch.nn.Module

    torch_encoder = TorchEncoder(input_size=64, hidden_size=32)
    model = AutoEncoder(encoder=torch_encoder)  # This creates the model

    # But this will fail:
    model.save("autoencoder.mdlus")
    # Error: Cannot serialize torch.nn.Module arguments.
    # You must use Module.from_torch() to convert it first.


.. _backward-compatibility:

Backward Compatibility
----------------------

When evolving a model class over time, for example, renaming the class,
adding or removing constructor arguments, you may still need to load
checkpoints that were saved with an older version of the class. PhysicsNeMo
provides a versioning and argument-mapping system for this purpose.

The key ingredients are:

- **``__model_checkpoint_version__``**: A version string (e.g. ``"0.2.0"``)
  on each :class:`~physicsnemo.core.module.Module` subclass. This is saved into
  every ``.mdlus`` file and compared at load time.
- **``__supported_model_checkpoint_version__``**: A dict mapping older
  version strings to warning messages. If the checkpoint version is in this
  dict, loading proceeds (with a warning) instead of raising an error.
- **``_backward_compat_arg_mapper``**: A classmethod that transforms
  constructor arguments from an older version into the format expected by the
  current version.
- **``_overridable_args``** and **``override_args``**: Allow callers of
  :meth:`~physicsnemo.core.module.Module.from_checkpoint` to override specific
  constructor arguments at load time. Only arguments listed in
  ``_overridable_args`` can be overridden.

**Example workflow:**

Suppose you have an initial model that has been deployed and trained
checkpoints exist:

.. code:: python

    import physicsnemo

    class MyModel(physicsnemo.Module):
        __model_checkpoint_version__ = "0.1.0"

        def __init__(self, img_channels, hidden_dim=64):
            super().__init__()
            self.img_channels = img_channels
            self.hidden_dim = hidden_dim
            # ... model layers ...

        def forward(self, x):
            return x

    # A trained checkpoint is saved
    model = MyModel(img_channels=3)
    model.save("my_model.mdlus")

Later, you refactor the class: you rename ``img_channels`` to
``in_channels`` and remove the ``hidden_dim`` argument. To still be able to
load old checkpoints:

.. code:: python

    class MyModel(physicsnemo.Module):
        __model_checkpoint_version__ = "0.2.0"
        __supported_model_checkpoint_version__ = {
            "0.1.0": (
                "Loading MyModel checkpoint from version 0.1.0. "
                "Consider re-saving to upgrade to 0.2.0."
            ),
        }

        @classmethod
        def _backward_compat_arg_mapper(cls, version, args):
            args = super()._backward_compat_arg_mapper(version, args)
            if version == "0.1.0":
                # Rename img_channels -> in_channels
                if "img_channels" in args:
                    args["in_channels"] = args.pop("img_channels")
                # Remove deprecated argument
                args.pop("hidden_dim", None)
            return args

        def __init__(self, in_channels):
            super().__init__()
            self.in_channels = in_channels
            # ... updated model layers ...

        def forward(self, x):
            return x

    # Old checkpoint loads successfully (with a warning)
    loaded = MyModel.from_checkpoint("my_model.mdlus")

For the full details of
:meth:`~physicsnemo.core.module.Module.from_checkpoint`, refer to the :ref:`API
Reference <modules-api-reference>`.


.. _physicsnemo-model-registry:

PhysicsNeMo Model Registry and Entry Points
-------------------------------------------

PhysicsNeMo contains a :class:`~physicsnemo.core.registry.ModelRegistry` that
provides a single, global lookup for all model classes in the PhysicsNeMo
ecosystem. This is useful because:

- **Stable access**: Once a class is registered, it can be retrieved by name
  regardless of where its source module lives. If the class is later moved to a
  different package or module path, the registry-based import remains the same.
- **Checkpoint loading**: :meth:`~physicsnemo.core.module.Module.from_checkpoint`
  uses the registry to resolve class names stored in ``.mdlus`` files.
- **Third-party integration**: External packages can expose models via
  Python entry points so that they appear in the registry when installed.

A few important rules:

- **Names must be unique.** Attempting to register a name that is already in
  use raises a ``ValueError``.
- **The class name is used by default.** When registering via
  :meth:`~physicsnemo.core.registry.ModelRegistry.register` or
  ``Module.from_torch(..., register=True)``, the ``__name__`` attribute of the
  class is used unless an explicit name is provided.

Below is a simple example of how to use the model registry to obtain a model
class.

.. code:: python

    >>> from physicsnemo.registry import ModelRegistry
    >>> model_registry = ModelRegistry()
    >>> model_registry.list_models()
    ['AFNO', 'DLWP', 'FNO', 'FullyConnected', 'GraphCastNet', 'MeshGraphNet', 'One2ManyRNN', 'Pix2Pix', 'SFNO', 'SRResNet']
    >>> FullyConnected = model_registry.factory("FullyConnected")
    >>> model = FullyConnected(in_features=32, out_features=64)

You can also register classes automatically when defining them by passing
``register=True`` as a class keyword argument:

.. code:: python

    class MyModel(physicsnemo.Module, register=True):
        def __init__(self, hidden_dim=64):
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, x):
            return x

    # The class is now accessible from the registry
    registry = ModelRegistry()
    ModelClass = registry.factory("MyModel")

**Exposing Models via Entry Points**

The model registry also allows exposing models via
`Python entry points <https://amir.rachum.com/blog/2017/07/28/python-entry-points/>`_.
This allows for integration of models into the PhysicsNeMo ecosystem from
external packages. For example, suppose you have a package ``MyPackage`` that
contains a model ``MyModel``. You can expose this model to the PhysicsNeMo
registry by adding an entry point to your ``toml`` file:

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
    MyPhysicsNeMoModel = "mypackage.models:MyPhysicsNeMoModel"

.. code:: python

   # mypackage/models.py

   import torch.nn as nn
   from physicsnemo.core import Module

   class MyModel(nn.Module):
       def __init__(self):
           super(MyModel, self).__init__()
           self.conv1 = nn.Conv2d(1, 20, 5)
           self.conv2 = nn.Conv2d(20, 20, 5)

       def forward(self, x):
           x = self.conv1(x)
           return self.conv2(x)

   MyPhysicsNeMoModel = Module.from_torch(MyModel)


Once this package is installed, you can access the model via the PhysicsNeMo model
registry.

.. code:: python

   >>> from physicsnemo.registry import ModelRegistry
   >>> model_registry = ModelRegistry()
   >>> model_registry.list_models()
   ['MyPhysicsNeMoModel', 'AFNO', 'DLWP', 'FNO', 'FullyConnected', 'GraphCastNet', 'MeshGraphNet', 'One2ManyRNN', 'Pix2Pix', 'SFNO', 'SRResNet']
   >>> MyPhysicsNeMoModel = model_registry.factory("MyPhysicsNeMoModel")


.. _modules-api-reference:

API Reference
-------------

:code:`Module`
~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.core.module.Module
    :members: save, load, from_checkpoint, from_torch, instantiate, device, num_parameters
    :exclude-members: __init__

:code:`ModelRegistry`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.core.registry.ModelRegistry
    :members: register, factory, list_models
    :exclude-members: __init__

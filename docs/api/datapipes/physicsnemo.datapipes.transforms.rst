Built-in Transforms
===================

.. currentmodule:: physicsnemo.datapipes.transforms

PhysicsNeMo ``Transforms`` are the core data manipulation tool for datapipes.  All
transforms inherit from the transform base class.

.. autoclass:: physicsnemo.datapipes.transforms.base.Transform
    :members:
    :show-inheritance:

To implement a new transform, users are required to override the ``__call__``
method as well as any initialization or configuration details.

The input to a ``transform`` is **mutable** by default, and so the order of transformations matters. 

In general, transforms are transactional: take input in, manipulate it, return output, and almost never update state.  Transforms should be device-agnostic,
and use a compute-follows-data principle, operating on data on the device where
it resides whenever possible.

By default, transforms accept and return ``tensordict`` objects: this is not, 
strictly, a requirement that must be enforced.
If you implement custom transforms that return different data types, downstream
transforms should expect that data type.  One example of this, found in the
minimal datapipe examples, is turning the ``tensordict`` objects into a PyTorch
Geometric graph object.  This type of manipulation is perfectly valid, but
requires custom collation functions and prevents usage of ``tensordict``-based
transforms downstream.

One unique transformation is the ``Compose`` transformation, which takes
a list of transformations and logically applies them in order, as one
transformation.  Use the ``Compose`` transformation similar to the ``torch.nn.Sequential`` container for stacking PyTorch Modules together.

.. autoclass:: physicsnemo.datapipes.transforms.compose.Compose
    :members:
    :show-inheritance:

Below are a collection of commonly used transforms in PhysicsNeMo's datapipes.
Since the input and output are ``tensordict`` objects, a common configuration pattern
for a transformation is to specify a list of input and output keys to operate on,
though this is not always the case.

Most transforms do not have an internal state;  the ones that do, however, will
automatically move tensors to and from the GPU with the ``to()`` syntax as expected.

Normalization
-------------

.. autoclass:: physicsnemo.datapipes.transforms.normalize.Normalize
    :members:
    :show-inheritance:

Subsampling
-----------

.. autoclass:: physicsnemo.datapipes.transforms.subsample.SubsamplePoints
    :members:
    :show-inheritance:

Geometric
---------

.. autoclass:: physicsnemo.datapipes.transforms.geometric.ComputeSDF
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.geometric.ComputeNormals
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.geometric.Translate
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.geometric.Scale
    :members:
    :show-inheritance:

Spatial
-------

.. autoclass:: physicsnemo.datapipes.transforms.spatial.BoundingBoxFilter
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.spatial.CreateGrid
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.spatial.KNearestNeighbors
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.spatial.CenterOfMass
    :members:
    :show-inheritance:

Field Processing
----------------

.. autoclass:: physicsnemo.datapipes.transforms.field_slice.FieldSlice
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.field_processing.BroadcastGlobalFeatures
    :members:
    :show-inheritance:

Feature Building
----------------

.. autoclass:: physicsnemo.datapipes.transforms.concat_fields.ConcatFields
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.concat_fields.NormalizeVectors
    :members:
    :show-inheritance:

Utility
-------

.. autoclass:: physicsnemo.datapipes.transforms.utility.Rename
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.utility.Purge
    :members:
    :show-inheritance:

.. autoclass:: physicsnemo.datapipes.transforms.utility.ConstantField
    :members:
    :show-inheritance:

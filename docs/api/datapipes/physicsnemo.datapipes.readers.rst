Built-in Readers
================

.. currentmodule:: physicsnemo.datapipes.readers

PhysicsNeMo's datapipe ``readers`` are an abstracted interface for enabling data
loading from various sources into the datapipe framework.  By providing a common
interface and API to implement, users can easily implement new dataset readers
and plug them into existing datapipes.


Base Reader
-----------

Each ``reader`` class should inherit from the base ``Reader``, below.  Users should
implement at minimum two functions: ``_load_sample``, which takes an integer index
defining an index into the dataset, and returns a dictionary of CPU tensors from the dataset.  Note that you do not need to move the tensors to the GPU - it will
be handled automatically.

Additionally, users must implement the ``__len__`` method to return the length
of the dataset.

At configuration, each ``reader`` or subclass should configure ``pin_memory`` to
true or false to set CPU memory pinning.  This enables faster, async data transfer
from host to device, sometimes at the cost of higher CPU resource usage.

The ``Reader`` abstraction has configurable support for dataset metadata that will not be passed through the preprocessing pipeline, but can be optionally consumed 
in a training or inference loop.  To control the precise way to fetch and return
metadata, override the ``_get_sample_metadata`` class.

For some datasets, such as very high resolution volumetric datasets that will
get downsampled at training time, the ``Reader`` classes provide a fast-path,
only-read-what-you-need optimization called "coordinated_subsampling".  In
essence, if your input and output fields are both 1 billion points, but you
only will consume 100,000 per training step, there is no reason to read the
other 999 Million points.  However, the IO selection must be properly _coordinated_
to take the same sub-samples per batch, and consume new subsamples each training
iteration.

.. autoclass:: physicsnemo.datapipes.readers.base.Reader
    :members:
    :show-inheritance:


Usage of readers
----------------

Readers are designed to be consumed by physicsnemo Dataset objects. Of course,
use them however is desired.  They support iteration syntax, and random access
indexing through ``__getitem__`` - note that the user should not implement ``__getitem__`` directly.

Each reader will return a ``tensordict`` object of data when accessed.
The conversion from ``dict`` (returned by user-implemented ``_load_sample``)
to ``tensordict`` is automatic.

Readers handle IO exclusively - it is highly encouraged, if you are building a
a custom datapipe, to implement transforms as separate operations.  This will
enable GPU computations and composable, extensible pipelines.

Below are the current built-in readers for physicsnemo.

HDF5Reader
----------

.. autoclass:: physicsnemo.datapipes.readers.hdf5.HDF5Reader
    :members:
    :show-inheritance:

NumpyReader
-----------

.. autoclass:: physicsnemo.datapipes.readers.numpy.NumpyReader
    :members:
    :show-inheritance:

ZarrReader
----------

.. autoclass:: physicsnemo.datapipes.readers.zarr.ZarrReader
    :members:
    :show-inheritance:

TensorStoreZarrReader
---------------------

.. autoclass:: physicsnemo.datapipes.readers.tensorstore_zarr.TensorStoreZarrReader
    :members:
    :show-inheritance:

VTKReader
---------

.. autoclass:: physicsnemo.datapipes.readers.vtk.VTKReader
    :members:
    :show-inheritance:

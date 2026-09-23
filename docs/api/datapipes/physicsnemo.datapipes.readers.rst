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

Every ``reader`` also accepts an optional ``cache`` argument; refer to
:ref:`dataset-cache` below.

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

.. _dataset-cache:

Caching repeated metadata reads
-------------------------------

Loading data from storage on a laptop or node-local storage is qualitatively
different than from remote storage systems such as Lustre, object-store systems
like Amazon S3, Google Cloud Storage etc.  Those systems, while performant at scale,
often suffer from degraded performance on the smallest of reads which are latency-bound,
and for good reason: remote systems are optimized to serve data in large buckets,
which amortizes the cost of delivery.

Unfortunately, these storage systems often have small, sometimes unexpected, costs associated
with every read from *metadata*: every query of a file to read will
potentially traverse a file tree, parse a directory listing, etc.

To ameliorate this, we have developed a metadata caching mechanism ``DatasetCache``.
For repeated accesses, on immutable datasets, once a query has been made and the metadata
cached, it is looked up from local information rather than pulled from remote systems.
``DatasetCache`` is an optional object, shared by any
number of readers, that remembers those results in RAM and on node-local disk.
Bulk array data is not cached, and with no cache configured readers behave as
before.

While we don't yet support full dataset caching with tensor storage, it is a
feature we're considering and exploring in the future.

.. code-block:: python

    from physicsnemo.datapipes import DatasetCache, DomainMeshReader

    cache = DatasetCache(ram_bytes_limit=2 * 2**30, disk_dir="/temp/pn-cache")
    reader = DomainMeshReader("/data/meshes", cache=cache)

.. autoclass:: physicsnemo.datapipes.caching.DatasetCache
    :members:
    :show-inheritance:

.. note::

    Metadata caching will not work correctly on mutable datasets, since the caching
    system will not automatically invalidate metadata against the remote storage.  Checking
    to invalidate the metadata would be, itself, a metadata operation and defeat the purpose.


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

MeshReader
----------

``MeshReader`` loads ``physicsnemo.mesh.Mesh`` objects saved in the ``.pmsh``
format, one sample per file, with optional cell or point subsampling at load
time.  Memory-mapped rows are read with ``preadv`` where available, which avoids
page faults on subsampled and pinned loads.

.. autoclass:: physicsnemo.datapipes.readers.mesh.MeshReader
    :members:
    :show-inheritance:

DomainMeshReader
----------------

``DomainMeshReader`` loads ``physicsnemo.mesh.DomainMesh`` objects saved in the
``.pdmsh`` format: an interior mesh, named boundary meshes, and global data.
Sibling meshes such as a geometry surface can be attached at full resolution
with ``extra_boundaries``.

.. autoclass:: physicsnemo.datapipes.readers.mesh.DomainMeshReader
    :members:
    :show-inheritance:

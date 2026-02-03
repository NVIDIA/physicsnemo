PhysicsNeMo Utils
==================

.. automodule:: physicsnemo.utils
.. currentmodule:: physicsnemo.utils

The PhysicsNeMo Utils module provides a comprehensive set of utilities that support various aspects of scientific computing,
machine learning, and physics simulations. These utilities range from optimization helpers and distributed computing tools
to specialized functions for weather and climate modeling, and geometry processing. The module is designed to simplify common
tasks while maintaining high performance and scalability.

.. autosummary::
   :toctree: generated

Optimization Utils
------------------

The optimization utilities provide tools for capturing and managing training states, gradients, and optimization processes.
These are particularly useful when implementing custom training loops or specialized optimization strategies.

.. automodule:: physicsnemo.utils.capture
    :members:
    :show-inheritance:

Neighbor Functionals
--------------------

Functional wrappers for neighbor searches live under ``physicsnemo.nn.functional``.

.. automodule:: physicsnemo.nn.functional.knn
    :members:
    :show-inheritance:


GraphCast Utils
---------------

A collection of utilities specifically designed for working with the GraphCast model, including data processing,
graph construction, and specialized loss functions. These utilities are essential for implementing and
training GraphCast-based weather prediction models.

.. automodule:: physicsnemo.utils.graphcast.data_utils
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.graphcast.graph
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.graphcast.graph_utils
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.graphcast.loss
    :members:
    :show-inheritance:

Filesystem Utils
----------------

Utilities for handling file operations, caching, and data management across different storage systems.
These utilities abstract away the complexity of dealing with different filesystem types and provide
consistent interfaces for data access.

.. automodule:: physicsnemo.utils.filesystem
    :members:
    :show-inheritance:

.. _diffusion_utils:

Diffusion Utils
---------------

Tools for working with diffusion models and other generative approaches,
including deterministic and stochastic sampling utilities.

.. automodule:: physicsnemo.diffusion.samplers.deterministic_sampler
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.diffusion.samplers.stochastic_sampler
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.diffusion.utils
    :members:
    :show-inheritance:

Geometry Functionals
--------------------

Utilities for geometric operations, including neighbor search and signed distance field calculations.
These are essential for physics simulations and geometric deep learning applications.

.. automodule:: physicsnemo.nn.functional.radius_search
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.nn.functional.sdf
    :show-inheritance:

Weather / Climate Utils
-----------------------

Specialized utilities for weather and climate modeling, including calculations for solar radiation
and atmospheric parameters. These utilities are used extensively in weather prediction models.

.. automodule:: physicsnemo.utils.insolation
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.zenith_angle
    :show-inheritance:

.. _patching_utils:

Patching Utils
--------------

Patching utilities are particularly useful for *patch-based* diffusion, also called
*multi-diffusion*. This approach is used to scale diffusion to very large images.
The following patching utilities extract patches from 2D images, and typically gather
them in the batch dimension. A batch of patches is therefore composed of multiple
smaller patches that are extracted from each sample in the original batch of larger
images. Diffusion models can then process these patches independently. These
utilities also support fusing operations to reconstruct the entire predicted
image from the individual predicted patches.

.. automodule:: physicsnemo.diffusion.multi_diffusion
    :members:
    :show-inheritance:

Domino Utils
------------

Utilities for working with the Domino model, including data processing and grid construction.
These utilities are essential for implementing and training Domino-based models.

.. automodule:: physicsnemo.utils.domino.utils
    :members:
    :show-inheritance:

CorrDiff Utils
--------------

Utilities for working with the CorrDiff model, particularly for the diffusion and regression steps.

.. automodule:: physicsnemo.diffusion.samplers
    :members:
    :show-inheritance:

Profiling Utils
---------------

Utilities for profiling the performance of a model.

.. automodule:: physicsnemo.utils.profiling
    :members:
    :show-inheritance:

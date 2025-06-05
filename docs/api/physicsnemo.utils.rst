PhysicsNeMo Utils
==================

.. automodule:: physicsnemo.utils
.. currentmodule:: physicsnemo.utils

The PhysicsNeMo Utils module provides a comprehensive set of utilities that support various aspects of scientific computing,
machine learning, and physics simulations. These utilities range from optimization helpers and distributed computing tools
to specialized functions for weather/climate modeling and geometry processing. The module is designed to simplify common
tasks while maintaining high performance and scalability.

.. autosummary::
   :toctree: generated

Optimization utils
------------------

The optimization utilities provide tools for capturing and managing training states, gradients, and optimization processes.
These are particularly useful when implementing custom training loops or specialized optimization strategies.

.. automodule:: physicsnemo.utils.capture
    :members:
    :show-inheritance:

GraphCast utils
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

Filesystem utils
----------------

Utilities for handling file operations, caching, and data management across different storage systems.
These utilities abstract away the complexity of dealing with different filesystem types and provide
consistent interfaces for data access.

.. automodule:: physicsnemo.utils.filesystem
    :members:
    :show-inheritance:

Generative utils
----------------

Tools for working with generative models, including deterministic and stochastic sampling utilities.
These are particularly useful when implementing diffusion models or other generative approaches.

.. automodule:: physicsnemo.utils.generative.deterministic_sampler
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.generative.stochastic_sampler
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.generative.utils
    :members:
    :show-inheritance:

Geometry utils
--------------

Utilities for geometric operations, including neighbor search and signed distance field calculations.
These are essential for physics simulations and geometric deep learning applications.

.. automodule:: physicsnemo.utils.neighbor_list
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.sdf
    :show-inheritance:

Weather / Climate utils
------------------------

Specialized utilities for weather and climate modeling, including calculations for solar radiation
and atmospheric parameters. These utilities are used extensively in weather prediction models.

.. automodule:: physicsnemo.utils.insolation
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.zenith_angle
    :show-inheritance:

Patching utils
--------------

Utilities for handling data patching operations, particularly useful in image-based deep learning
models where processing needs to be done on patches of the input data.

.. automodule:: physicsnemo.utils.patching
    :members:
    :show-inheritance:

Domino utils
------------

Utilities for working with the Domino model, including data processing and grid construction.
These utilities are essential for implementing and training Domino-based models.

.. automodule:: physicsnemo.utils.domino.utils
    :members:
    :show-inheritance:

CorrDiff utils
--------------

Utilities for working with the CorrDiff model, particularly for the diffusion and regression steps.

.. automodule:: physicsnemo.utils.corrdiff.utils
    :members:
    :show-inheritance:

Profiling utils
---------------

Utilities for profiling the performance of a model.

.. automodule:: physicsnemo.utils.profiling
    :members:
    :show-inheritance:
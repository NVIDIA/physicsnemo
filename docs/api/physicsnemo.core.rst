PhysicsNeMo Core
================

.. automodule:: physicsnemo.core
.. currentmodule:: physicsnemo.core

The PhysicsNeMo core module provides the base functionality for the entire
PhysicsNeMo framework.  This encompasses filesystem and python utilities, the implementation
of `physicsnemo.Module` and metadata, and the model registry.

PhysicsNemo ``Module``
----------------------

The :class:`Module` provides a base class for all user facing models in PhysicsNeMo.
It provides a unified interface for the registry, checkpointing, and optimization.
Full API information can be found in the :doc:`PhysicsNeMo Modules <models/modules>`.

Filesystem Utils
----------------

Utilities for handling file operations, caching, and data management across different storage systems.
These utilities abstract away the complexity of dealing with different filesystem types and provide
consistent interfaces for data access.

.. automodule:: physicsnemo.core.filesystem
    :members:
    :show-inheritance:

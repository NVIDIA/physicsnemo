PhysicsNeMo Active Learning
===========================

.. currentmodule:: physicsnemo.active_learning

Developing Active Learning Workflows
-------------------------------------

For a high level overview and understanding of how to construct active
learning workflows using PhysicsNeMo, users should consult the `User 
Guide <https://docs.nvidia.com/physicsnemo/latest/user-guide/active_learning.html>`_
. The guide will motivate the need for active learning, the abstraction
provided by PhysicsNeMo, and some additional tips for developing custom
components like querying and labeling strategies.

API Reference
-------------

Protocols
^^^^^^^^^

.. automodule:: physicsnemo.active_learning.protocols
    :members:
    :undoc-members:
    :show-inheritance:

Configuration Classes
^^^^^^^^^^^^^^^^^^^^^

These data structures are used to modify the behavior of different components
of the active learning workflow. The general pattern is to ensure that they
are JSON-serializable so that they can be checkpointed and restarted.

.. automodule:: physicsnemo.active_learning.config
    :members:
    :undoc-members:
    :show-inheritance:


Default Training Loop
^^^^^^^^^^^^^^^^^^^^^

This module and corresponding
:class:`~physicsnemo.active_learning.loop.DefaultTrainingLoop` class
implements the :class:`~physicsnemo.active_learning.protocols.TrainingLoop` interface,
and should provide most of the necessary boilerplate for model training
and fine-tuning; users will need to provide the training, validation,
and testing step protocols when configuring the loop.

.. automodule:: physicsnemo.active_learning.loop
    :members:
    :undoc-members:
    :show-inheritance:

Active Learning Driver
^^^^^^^^^^^^^^^^^^^^^^

This module and class implements the 
:class:`~physicsnemo.active_learning.protocols.DriverProtocol` interface,
and is usable out-of-the-box for most active learning workflows. The
:class:`~physicsnemo.active_learning.driver.Driver` class is configured
by :class:`~physicsnemo.active_learning.config.DriverConfig`, and serves
as the focal point for orchestrating the active learning.

.. automodule:: physicsnemo.active_learning.driver
    :members:
    :undoc-members:
    :show-inheritance:

Active Learning Registry
^^^^^^^^^^^^^^^^^^^^^^^

The registry provides a centralized location for registering and constructing
custom active learning strategies. It enables string-based lookups for
checkpointing and provides argument validation when constructing protocol
instances.

.. note::

    Users should not use the class directly, but rather the instance of the
    class through the :data:`~physicsnemo.active_learning.registry` object,
    documented below.

.. autodata:: physicsnemo.active_learning.registry
    :annotation: = ActiveLearningRegistry()

    Global registry instance for active learning protocols.

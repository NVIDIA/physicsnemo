PhysicsNeMo Optim
=================

.. automodule:: physicsnemo.optim
.. currentmodule:: physicsnemo.optim

The PhysicsNeMo Optim module provides optimization utilities for training physics-informed
machine learning models. These utilities are designed to work seamlessly with PyTorch's
optimizer ecosystem while providing additional functionality for complex training scenarios.

CombinedOptimizer
-----------------

The :class:`CombinedOptimizer` allows combining multiple PyTorch optimizers into a unified
interface. This is particularly useful when different parts of a model require different
optimization strategies - for example, using Adam for encoder layers and SGD with momentum
for decoder layers.

.. autoclass:: physicsnemo.optim.CombinedOptimizer
    :members:
    :show-inheritance:
    :special-members: __init__


PhysicsNeMo Sym
===============

Symbolic PDE residual computation for physics-informed training.

PDE Base Class
--------------

.. autoclass:: physicsnemo.sym.eq.pde.PDE
   :members:
   :show-inheritance:

PhysicsInformer
---------------

.. autoclass:: physicsnemo.sym.eq.phy_informer.PhysicsInformer
   :members:
   :show-inheritance:

Gradient Calculators
--------------------

``GradientCalculator`` is the user-facing dispatcher that ``PhysicsInformer``
uses internally based on the ``grad_method`` argument. The individual
per-method ``Gradients*`` modules are exposed for advanced users that need to
compute spatial derivatives outside of the ``PhysicsInformer`` pipeline.

.. autoclass:: physicsnemo.sym.eq.gradients.GradientCalculator
   :members:

.. autoclass:: physicsnemo.sym.eq.gradients.GradientsAutoDiff
   :members:
   :show-inheritance:

.. autoclass:: physicsnemo.sym.eq.gradients.GradientsFiniteDifference
   :members:
   :show-inheritance:

.. autoclass:: physicsnemo.sym.eq.gradients.GradientsMeshlessFiniteDifference
   :members:
   :show-inheritance:

.. autoclass:: physicsnemo.sym.eq.gradients.GradientsSpectral
   :members:
   :show-inheritance:

.. autoclass:: physicsnemo.sym.eq.gradients.GradientsLeastSquares
   :members:
   :show-inheritance:

.. autofunction:: physicsnemo.sym.eq.gradients.compute_connectivity_tensor

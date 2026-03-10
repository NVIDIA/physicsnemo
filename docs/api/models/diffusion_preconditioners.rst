.. _diffusion_preconditioners:

Diffusion Preconditioners
=========================

Preconditioning is an essential technique to improve the performance of
diffusion models. It consists in scaling the latent state and the noise
level that are passed to a network. Some preconditioning also requires to
re-scale the output of the network. PhysicsNeMo provides a set of preconditioning
classes that are wrappers around backbones or specialized architectures.

.. autoclass:: physicsnemo.diffusion.preconditioners.VPPrecond
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.diffusion.preconditioners.VEPrecond
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.diffusion.preconditioners.iDDPMPrecond
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.diffusion.preconditioners.EDMPrecond
    :show-inheritance:
    :members:
    :exclude-members: forward

.. autoclass:: physicsnemo.diffusion.preconditioners.EDMPrecondSuperResolution
    :show-inheritance:
    :members:
    :exclude-members: forward
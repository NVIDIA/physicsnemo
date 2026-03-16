.. _diffusion_preconditioners:

Preconditioners
===============

.. currentmodule:: physicsnemo.diffusion.preconditioners

Preconditioning is an optional but very common technique that improves
the stability and convergence of diffusion model training.  The core idea is
that the raw inputs and outputs of a neural network span very different
scales depending on the noise level :math:`\sigma(t)`.  A preconditioner wraps
the backbone with an affine rescaling so that the effective input and output
have unit variance across all noise levels, making the learning problem
uniformly well-conditioned.


Three Approaches
----------------

Depending on how much customization you need, there are three ways to use
preconditioning in the framework.

**1. Ready-to-use preconditioners.**
The framework ships preconditioners that pair with each built-in
:ref:`noise scheduler <diffusion_noise_schedulers>`.  These work out of the
box with no additional implementation:

- :class:`EDMPreconditioner` with
  :class:`~physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler`
- :class:`VEPreconditioner` with
  :class:`~physicsnemo.diffusion.noise_schedulers.VENoiseScheduler`
- :class:`VPPreconditioner` with
  :class:`~physicsnemo.diffusion.noise_schedulers.VPNoiseScheduler`
- :class:`IDDPMPreconditioner` with
  :class:`~physicsnemo.diffusion.noise_schedulers.IDDPMNoiseScheduler`

.. code-block:: python

    from physicsnemo.diffusion.preconditioners import EDMPreconditioner

    precond = EDMPreconditioner(backbone_model, sigma_data=0.5)

**2. Subclass the abstract base class.**
For a custom affine preconditioning scheme, subclass
:class:`BaseAffinePreconditioner` and implement
:meth:`~BaseAffinePreconditioner.compute_coefficients`.  Optionally override
:meth:`~BaseAffinePreconditioner.sigma` if :math:`\sigma(t) \neq t`.  The
``forward`` method should not be overridden.

.. code-block:: python

    from physicsnemo.diffusion.preconditioners import BaseAffinePreconditioner

    class MyPreconditioner(BaseAffinePreconditioner):
        def compute_coefficients(self, sigma):
            c_skip = 1 / (sigma**2 + 1)
            c_out = sigma / (sigma**2 + 1).sqrt()
            c_in = 1 / (sigma**2 + 1).sqrt()
            c_noise = sigma.log() / 4
            return c_in, c_noise, c_out, c_skip

**3. Implement preconditioning directly in a Module.**
If the affine formula does not fit your use case, you can implement
preconditioning directly in a :class:`~physicsnemo.core.module.Module`
that satisfies the :class:`~physicsnemo.diffusion.DiffusionModel` protocol.
This gives complete freedom over the preconditioning logic.

.. code-block:: python

    from physicsnemo.core import Module

    class MyPreconditionedModel(Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x, t, condition=None):
            # Custom preconditioning logic
            x_scaled = x / (1 + t.view(-1, 1, 1, 1)**2).sqrt()
            out = self.backbone(x_scaled, t, condition)
            return x + t.view(-1, 1, 1, 1) * out


How Preconditioners Fit in the Pipeline
---------------------------------------

A preconditioner *itself* satisfies the
:class:`~physicsnemo.diffusion.DiffusionModel` interface, so it can be used
anywhere a plain model is expected---in training losses, in the denoiser
factory, and in sampling:

.. code-block:: python

    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.metrics.losses import MSEDSMLoss

    scheduler = EDMNoiseScheduler()
    precond = EDMPreconditioner(backbone_model, sigma_data=0.5)
    loss_fn = MSEDSMLoss(precond, scheduler)

    # Training: the loss sees `precond` as the model
    loss = loss_fn(x0, condition=condition)

    # Sampling: the preconditioner is used as the predictor
    from functools import partial
    x0_predictor = partial(precond, condition=condition)
    denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)

.. important::

    All built-in preconditioners are designed so that the preconditioned output
    is an :math:`\mathbf{x}_0`-prediction (clean data estimate).  They are
    intended for use with
    :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss` with
    ``prediction_type="x0"`` (the default).

See the :class:`BaseAffinePreconditioner` docstring for additional examples,
including how to write thin wrappers to adapt backbones with non-standard
signatures (for example, ``SongUNet``, ``DiT``) to the
:class:`~physicsnemo.diffusion.DiffusionModel` interface.


API Reference
-------------

:code:`BaseAffinePreconditioner`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.preconditioners.BaseAffinePreconditioner
    :show-inheritance:
    :members:
    :exclude-members: forward

:code:`EDMPreconditioner`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.preconditioners.EDMPreconditioner
    :show-inheritance:
    :members:
    :exclude-members: forward

:code:`VEPreconditioner`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.preconditioners.VEPreconditioner
    :show-inheritance:
    :members:
    :exclude-members: forward

:code:`VPPreconditioner`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.preconditioners.VPPreconditioner
    :show-inheritance:
    :members:
    :exclude-members: forward

:code:`IDDPMPreconditioner`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.preconditioners.IDDPMPreconditioner
    :show-inheritance:
    :members:
    :exclude-members: forward

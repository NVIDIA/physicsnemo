.. _diffusion_noise_schedulers:

Noise Schedulers
================

.. currentmodule:: physicsnemo.diffusion.noise_schedulers

Noise schedulers are the central abstraction in the PhysicsNeMo diffusion
framework.  A noise scheduler defines the forward diffusion process (how noise
is added to data) and provides the ingredients needed for both training and
inference.  It is the *only* component that participates in every stage of
the pipeline: training, loss computation, denoiser construction, and sampling.

If you are familiar with the
`Scheduler API in HuggingFace Diffusers <https://huggingface.co/docs/diffusers/api/schedulers/overview>`_,
PhysicsNeMo noise schedulers serve a broadly similar role.  A key design
difference is that the PhysicsNeMo
:class:`NoiseScheduler` also owns the
:meth:`~NoiseScheduler.get_denoiser` factory method, which converts a trained
:class:`~physicsnemo.diffusion.Predictor` into a
:class:`~physicsnemo.diffusion.Denoiser` suitable for the solver.  This keeps
the coupling between noise schedule and reverse process in one place and avoids
the need for schedule-specific sampler code.


Role in Training
----------------

During training, the noise scheduler is responsible for three operations:

1. **Sampling diffusion times** via :meth:`~NoiseScheduler.sample_time`.
2. **Adding noise** via :meth:`~NoiseScheduler.add_noise`.
3. **Computing loss weights** via :meth:`~NoiseScheduler.loss_weight`.

These three methods are consumed by the
:class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss` training loss.

.. code-block:: python

    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.metrics.losses import MSEDSMLoss

    scheduler = EDMNoiseScheduler()
    loss_fn = MSEDSMLoss(model, scheduler)

    # Inside the training loop:
    loss = loss_fn(x0)   # Internally samples t, adds noise, computes weighted MSE
    loss.backward()


Role in Inference (Sampling)
----------------------------

During sampling, the noise scheduler provides:

- **Time-step schedule** via :meth:`~NoiseScheduler.timesteps`.
- **Initial latent state** via :meth:`~NoiseScheduler.init_latents`.
- **Denoiser factory** via :meth:`~NoiseScheduler.get_denoiser`.

.. code-block:: python

    from functools import partial
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.samplers import sample

    scheduler = EDMNoiseScheduler()
    num_steps = 50

    t_steps = scheduler.timesteps(num_steps)
    tN = t_steps[0].expand(4)                      # Batch of 4 samples
    xN = scheduler.init_latents((3, 64, 64), tN)    # 3-channel, 64x64 images

    x0_predictor = partial(trained_model, condition=condition)
    denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
    samples = sample(denoiser, xN, scheduler, num_steps=num_steps)


Three Levels of Customization
------------------------------

Following the framework's
:ref:`design philosophy <diffusion_introduction>`, noise schedulers are
available at three levels:

- **Protocol** (:class:`NoiseScheduler`): The minimal interface.  Any object
  that implements the six required methods can be used as a noise scheduler.
  This is the right choice for fully custom forward processes (for example,
  non-Gaussian, non-linear, discrete).

- **Abstract base class** (:class:`LinearGaussianNoiseScheduler`): For the
  common family of linear-Gaussian forward processes of the form
  :math:`\mathbf{x}(t) = \alpha(t)\,\mathbf{x}_0 + \sigma(t)\,\boldsymbol{\epsilon}`.
  This base class implements noise injection, score conversion, and denoiser
  construction.  Subclasses only need to define the schedule-specific quantities:

  - the signal coefficient :math:`\alpha(t)`
  - the noise level :math:`\sigma(t)`
  - their time derivatives :math:`\dot{\alpha}(t)` and :math:`\dot{\sigma}(t)`
  - the inverse mapping :math:`\sigma^{-1}(\sigma) = t` from noise level back to time
  - the discretization of the diffusion time grid

- **Ready-to-use schedules**: Five concrete implementations that work out of
  the box:

  - :class:`EDMNoiseScheduler` --- :math:`\alpha(t)=1`,
    :math:`\sigma(t)=t`.  The recommended default for most applications.
  - :class:`VENoiseScheduler` --- Variance Exploding schedule.
  - :class:`VPNoiseScheduler` --- Variance Preserving schedule.
  - :class:`IDDPMNoiseScheduler` --- Improved DDPM schedule.
  - :class:`StudentTEDMNoiseScheduler` --- EDM variant with Student-t noise
    for heavy-tailed data.


API Reference
-------------

:code:`NoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.NoiseScheduler
    :members:
    :exclude-members: __init__

:code:`LinearGaussianNoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`EDMNoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`VENoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.VENoiseScheduler
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`VPNoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.VPNoiseScheduler
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`IDDPMNoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.IDDPMNoiseScheduler
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`StudentTEDMNoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.StudentTEDMNoiseScheduler
    :show-inheritance:
    :members:
    :exclude-members: __init__

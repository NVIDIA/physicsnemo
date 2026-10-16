.. _diffusion_noise_schedulers:

Noise Schedulers
================

.. currentmodule:: physicsnemo.diffusion.noise_schedulers

Noise schedulers are the central abstraction in the PhysicsNeMo diffusion
framework. A noise scheduler defines the forward process (how noise
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


Amplitude ratio and inverse
---------------------------

The amplitude ratio :math:`\alpha(t)/\sigma(t)` measures signal strength
relative to noise strength. The method ``scheduler.snr(t)`` returns this
ratio. The usual power signal-to-noise ratio is its square.

Use ``scheduler.snr_inv(q)`` to recover a diffusion time from an amplitude
ratio or to match noise levels between schedules. EDM, VE, VP, rectified-flow,
continuous iDDPM, and TrigFlow provide differentiable inverses. Use finite
positive ratios on the schedule's invertible domain. Inverses do not clip
to training bounds. Add ``snr_inv`` for custom schedulers that need this
conversion, and keep it differentiable for training losses such as sCM.

.. code-block:: python

    import torch
    from physicsnemo.diffusion.noise_schedulers import TrigFlowNoiseScheduler

    scheduler = TrigFlowNoiseScheduler(sigma_data=0.5)
    t = torch.tensor([0.2, 0.7])
    recovered = scheduler.snr_inv(scheduler.snr(t))
    torch.testing.assert_close(recovered, t)

TrigFlow interpolates between data and Gaussian noise with trigonometric
coefficients. Its diffusion time is an angle in radians, whereas EDM uses
the noise standard deviation.

Pair ``TrigFlowNoiseScheduler`` with ``TrigFlowPreconditioner`` and the same
``sigma_data``. Training samples diffusion times uniformly. Inference uses
an evenly spaced grid ending at zero. The denoising weight
:math:`1/(\sigma_d\sin(t))^2` gives unit weight to the normalized-velocity
residual with this preconditioner. Keep training diffusion times above zero
for denoising losses and below :math:`\pi/2` for reverse-process sampling.

.. autoclass:: TrigFlowNoiseScheduler
    :show-inheritance:
    :members:


Role in Training
----------------

During training, the noise scheduler is responsible for three operations:

1. **Sampling diffusion times** via :meth:`~NoiseScheduler.sample_time`.
2. **Adding noise** via :meth:`~NoiseScheduler.add_noise`.
3. **Computing loss weights** via :meth:`~NoiseScheduler.loss_weight`.

The training losses use these three methods, including
:class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss` for diffusion and
:class:`~physicsnemo.diffusion.metrics.losses.FlowMatchingLoss` for flow
matching.

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

- **Ready-to-use schedules**: Multiple concrete implementations that work out of
  the box:

  - :class:`EDMNoiseScheduler` --- :math:`\alpha(t)=1`,
    :math:`\sigma(t)=t`.  The recommended default for most applications.
  - :class:`EDMLogUniformNoiseScheduler` --- EDM variant that samples
    training times uniformly in log-space instead of from a log-normal.
  - :class:`VENoiseScheduler` --- Variance Exploding schedule.
  - :class:`VPNoiseScheduler` --- Variance Preserving schedule.
  - :class:`IDDPMNoiseScheduler` --- Improved DDPM schedule.
  - :class:`StudentTEDMNoiseScheduler` --- EDM variant with Student-t noise
    for heavy-tailed data.
  - :class:`RectifiedFlowNoiseScheduler` --- Linear / optimal transport
    schedule, for flow matching training.


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

:code:`EDMLogUniformNoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.EDMLogUniformNoiseScheduler
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

:code:`RectifiedFlowNoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.RectifiedFlowNoiseScheduler
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`DomainParallelNoiseScheduler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.noise_schedulers.DomainParallelNoiseScheduler
    :members:
    :exclude-members: __init__

.. _diffusion_samplers:

Samplers and Solvers
====================

.. currentmodule:: physicsnemo.diffusion.samplers

The sampler is the main interface for generating new data from a trained
diffusion model.  Starting from pure noise :math:`\mathbf{x}_N`, the solver
iteratively denoises the latent state through a sequence of time-steps until it
reaches a clean sample :math:`\mathbf{x}_0`.

The central entry point is the :func:`sample` function.  It takes a
:class:`~physicsnemo.diffusion.Denoiser`, an initial noisy latent
:math:`\mathbf{x}_N`, and a
:class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`, and iterates
the reverse process to produce samples.


.. _diffusion_sampling_equation:

Generic Sampling Process
------------------------

The :func:`sample` function supports any reverse process that can be written
in the form:

.. math::
    \mathbf{x}_{n-1} = \text{Step}\bigl(
    D\!\bigl(\mathbf{x}_n, t_n;\, P(\mathbf{x}_n, t_n)\bigr);\;
    \mathbf{x}_n, t_n, t_{n-1}\bigr)

This equation is the foundation of the sampling process in the framework.
Every component described in this page maps to one of the three terms:

- :math:`P` is the **predictor** --- the :class:`~physicsnemo.diffusion.Predictor`
  that maps the noisy state and diffusion time to a prediction.  This is where
  all model logic lives, including conditioning and
  :ref:`guidance <diffusion_guidance>`.
- :math:`D` is the **denoiser** --- the :class:`~physicsnemo.diffusion.Denoiser`
  derived from :math:`P` via the noise scheduler's
  :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
  factory.
- :math:`\text{Step}` is the :ref:`solver <diffusion_available_solvers>`'s
  numerical update rule.

This generic formulation encompasses standard ODE/SDE-based sampling, but
also more advanced methods such as physics-informed posterior guidance
(`DPS <https://arxiv.org/abs/2209.14687>`_), score-based data assimilation
(`SDA <https://arxiv.org/abs/2306.10574>`_), and others.  Any
method that can express its update step through this denoiser/solver
decomposition can be used with the :func:`sample` function.


.. _diffusion_sampling_workflow:

Sampling Workflow
-----------------

A complete sampling workflow involves these steps:

1. **Load or reference a trained model** satisfying the
   :class:`~physicsnemo.diffusion.DiffusionModel` interface (typically a
   :ref:`backbone <diffusion_model_backbones>` wrapped in a
   :ref:`preconditioner <diffusion_preconditioners>`).

2. **Build a Predictor** (:math:`P` in the
   :ref:`sampling equation <diffusion_sampling_equation>`) by binding the
   conditioning via ``functools.partial``, converting the three-argument
   :class:`~physicsnemo.diffusion.DiffusionModel` into a two-argument
   :class:`~physicsnemo.diffusion.Predictor`.

3. **Convert to a Denoiser** (:math:`D` in the equation).  There are two paths:

   - **Without guidance** --- pass the predictor directly to the noise
     scheduler's
     :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
     factory (as an ``x0_predictor`` or ``score_predictor``).
   - **With guidance** --- first instantiate one or more
     :class:`~physicsnemo.diffusion.guidance.DPSGuidance` objects, then
     combine them with the predictor using
     :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor` to obtain a
     guided score-predictor.  Finally, pass this guided score-predictor to
     :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`.

4. **Initialize the noisy latent** :math:`\mathbf{x}_N` and the time-step
   schedule using the scheduler.

5. **Optionally configure a custom solver** (:math:`\text{Step}` in the
   equation) by instantiating a
   :class:`~physicsnemo.diffusion.samplers.solvers.Solver` (see
   :ref:`Available Solvers <diffusion_available_solvers>`), or simply pass a
   built-in string key (for example, ``"heun"``) to :func:`sample`.

6. **Call** :func:`sample` to run the reverse diffusion loop.


Example: Unconditional Image Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows the full workflow for an unconditional image model trained
with the EDM formulation.  It uses
:class:`~physicsnemo.models.diffusion_unets.SongUNet` as the backbone,
wrapped with a thin adapter to match the
:class:`~physicsnemo.diffusion.DiffusionModel` interface, and
:class:`~physicsnemo.diffusion.preconditioners.EDMPreconditioner` for
preconditioning.

The noise scheduler must generally be consistent between training and
sampling---in particular, the same schedule *family* (for example, EDM, VP)
should be used.  Schedule parameters (for example, ``sigma_min``, ``rho``) can
be adjusted at sampling time for experimentation, but the model was
optimized for the training schedule, so large deviations may degrade
sample quality.

.. code-block:: python

    import torch
    from functools import partial
    from physicsnemo.core import Module
    from physicsnemo.models.diffusion_unets import SongUNet
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.samplers import sample

    # --- Backbone: wrap SongUNet to match the DiffusionModel interface ---
    class UNetBackbone(Module):
        def __init__(self, img_resolution, channels, **kwargs):
            super().__init__()
            self.net = SongUNet(
                img_resolution=img_resolution,
                in_channels=channels,
                out_channels=channels,
                **kwargs,
            )
        def forward(self, x, t, condition=None):
            return self.net(x, noise_labels=t, class_labels=condition)

    backbone = UNetBackbone(img_resolution=64, channels=3, model_channels=64,
                            channel_mult=[1, 2, 2], num_blocks=2)

    # --- Preconditioner + training (sketch) ---
    scheduler = EDMNoiseScheduler(sigma_min=0.002, sigma_max=80.0, rho=7)
    precond = EDMPreconditioner(backbone, sigma_data=0.5)
    # ... train with MSEDSMLoss(precond, scheduler) ...

    # --- Sampling ---
    precond.eval()

    # Build predictor "P": bind condition=None for unconditional model
    x0_predictor = partial(precond, condition=None)

    # Convert to denoiser "D"
    denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)

    # Initialize time-steps and noisy latent
    num_steps = 50
    tN = scheduler.timesteps(num_steps)[0].expand(8)
    xN = scheduler.init_latents((3, 64, 64), tN)

    # Run sampling loop with Heun solver ("Step")
    samples = sample(denoiser, xN, scheduler, num_steps=num_steps, solver="heun")
    # samples.shape: (8, 3, 64, 64)

It is also possible to adjust the schedule parameters at sampling time.  For
instance, one might increase ``sigma_max`` or change ``rho`` to explore the
effect on sample quality:

.. code-block:: python

    sampling_scheduler = EDMNoiseScheduler(sigma_min=0.002, sigma_max=120.0, rho=5)
    denoiser = sampling_scheduler.get_denoiser(x0_predictor=x0_predictor)
    tN = sampling_scheduler.timesteps(num_steps)[0].expand(8)
    xN = sampling_scheduler.init_latents((3, 64, 64), tN)
    samples = sample(denoiser, xN, sampling_scheduler, num_steps=num_steps)


Example: Vector-Space Diffusion (Non-Image Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The diffusion framework is not limited to image data.  Any tensor-valued data
can be used, including 1D vectors.  Here the backbone uses the
:class:`~physicsnemo.models.mlp.FullyConnected` model from PhysicsNeMo,
wrapped with a thin adapter to match the
:class:`~physicsnemo.diffusion.DiffusionModel` interface.

.. code-block:: python

    import torch
    from functools import partial
    from physicsnemo.core import Module
    from physicsnemo.models.mlp import FullyConnected
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.samplers import sample

    # Backbone: wrap FullyConnected to match the DiffusionModel interface
    class FCBackbone(Module):
        def __init__(self, dim, hidden=256, num_layers=4):
            super().__init__()
            self.net = FullyConnected(
                in_features=dim, layer_size=hidden,
                out_features=dim, num_layers=num_layers,
            )
        def forward(self, x, t, condition=None):
            return self.net(x)

    data_dim = 32
    backbone = FCBackbone(dim=data_dim)
    scheduler = EDMNoiseScheduler()
    precond = EDMPreconditioner(backbone, sigma_data=1.0)
    # ... train with MSEDSMLoss(precond, scheduler) ...

    # Sampling
    precond.eval()
    x0_predictor = partial(precond, condition=None)
    denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
    num_steps = 50
    tN = scheduler.timesteps(num_steps)[0].expand(16)
    xN = scheduler.init_latents((data_dim,), tN)  # 1D latent: shape (16, 32)
    samples = sample(denoiser, xN, scheduler, num_steps=num_steps)
    # samples.shape: (16, 32)


Example: Conditional Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For conditional generation (for example, super-resolution), the model backbone
processes both the noisy latent state and the conditioning input.  A common
pattern is to concatenate the conditioning image along the channel dimension
inside a thin adapter.  At sampling time, the conditioning is bound into the
predictor (:math:`P`) via ``functools.partial``.

.. code-block:: python

    import torch
    from functools import partial
    from physicsnemo.core import Module
    from physicsnemo.models.diffusion_unets import SongUNet
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.samplers import sample

    C_x, C_cond, res = 3, 3, 64   # Image channels, conditioning channels, resolution

    # Backbone: SongUNet wrapped with an adapter that concatenates the
    # conditioning image along the channel dimension
    class ConditionalUNet(Module):
        def __init__(self):
            super().__init__()
            self.net = SongUNet(
                img_resolution=res,
                in_channels=C_x + C_cond,
                out_channels=C_x,
                model_channels=64,
                channel_mult=[1, 2, 2],
                num_blocks=2,
            )
        def forward(self, x, t, condition=None):
            x_cat = torch.cat([x, condition], dim=1)
            return self.net(x_cat, noise_labels=t, class_labels=None)

    backbone = ConditionalUNet()
    scheduler = EDMNoiseScheduler()
    precond = EDMPreconditioner(backbone, sigma_data=0.5)
    # ... train with MSEDSMLoss(precond, scheduler, condition=...) ...

    # --- Sampling ---
    precond.eval()

    # Conditioning image (for example, low-resolution input for super-resolution)
    low_res = torch.randn(4, C_cond, res, res)

    # Bind condition into the predictor "P"
    x0_predictor = partial(precond, condition=low_res)

    # Convert to denoiser "D" and sample
    denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)
    tN = scheduler.timesteps(50)[0].expand(4)
    xN = scheduler.init_latents((C_x, res, res), tN)
    samples = sample(denoiser, xN, scheduler, num_steps=50)
    # samples.shape: (4, 3, 64, 64)


Example: Conditional Sampling with DPS Guidance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DPS (Diffusion Posterior Sampling) guidance steers the sampling toward
satisfying observation constraints.  Guidance modifies the **predictor**
:math:`P` in the :ref:`sampling equation <diffusion_sampling_equation>`:
the guidance objects are combined with the x0-predictor into a guided
score-predictor via :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor`,
and then converted to a denoiser :math:`D` via the noise scheduler.

.. code-block:: python

    import torch
    from functools import partial
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.guidance import DPSScorePredictor, DataConsistencyDPSGuidance
    from physicsnemo.diffusion.samplers import sample

    scheduler = EDMNoiseScheduler()

    # Build predictor "P" from trained conditional model
    x0_predictor = partial(trained_model, condition=condition)

    # Build guidance objects
    mask = torch.zeros(4, 3, 64, 64, dtype=torch.bool)
    mask[:, :, ::8, ::8] = True  # Observe every 8th pixel
    y_obs = torch.randn(4, 3, 64, 64)
    guidance = DataConsistencyDPSGuidance(mask=mask, y=y_obs, std_y=0.1)

    # Combine predictor + guidance into a guided score-predictor
    guided_score_predictor = DPSScorePredictor(
        x0_predictor=x0_predictor,
        x0_to_score_fn=scheduler.x0_to_score,
        guidances=guidance,
    )

    # Convert to denoiser "D" via the noise scheduler
    denoiser = scheduler.get_denoiser(score_predictor=guided_score_predictor)

    tN = scheduler.timesteps(50)[0].expand(4)
    xN = scheduler.init_latents((3, 64, 64), tN)
    samples = sample(denoiser, xN, scheduler, num_steps=50)


Example: Custom Solver and Custom Time-Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to define a solver from scratch by implementing the
:class:`~physicsnemo.diffusion.samplers.solvers.Solver` protocol.  Any object
with a ``step(x, t_cur, t_next)`` method can serve as :math:`\text{Step}` in
the :ref:`sampling equation <diffusion_sampling_equation>`.  Here we implement
a simple implicit trapezoidal rule (second-order Runge-Kutta), and pair it with
custom time-steps and trajectory snapshots.

.. code-block:: python

    import torch
    from functools import partial
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.samplers import sample

    # Custom solver: implicit trapezoidal rule (second-order)
    class TrapezoidalSolver:
        def __init__(self, denoiser, num_inner_iters=3):
            self.denoiser = denoiser
            self.num_inner_iters = num_inner_iters

        def step(self, x, t_cur, t_next):
            t_cur_bc = t_cur.reshape(-1, *([1] * (x.ndim - 1)))
            t_next_bc = t_next.reshape(-1, *([1] * (x.ndim - 1)))
            h = t_next_bc - t_cur_bc
            d_cur = self.denoiser(x, t_cur)
            # Predictor: Euler step to get initial guess
            x_next = x + h * d_cur
            # Corrector: fixed-point iterations for the implicit trapezoidal rule
            for _ in range(self.num_inner_iters):
                d_next = self.denoiser(x_next, t_next)
                x_next = x + 0.5 * h * (d_cur + d_next)
            return x_next

    scheduler = EDMNoiseScheduler()
    x0_predictor = partial(trained_model, condition=None)
    denoiser = scheduler.get_denoiser(x0_predictor=x0_predictor)

    # Custom time-steps
    custom_t = torch.tensor([80.0, 40.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.0])
    tN = custom_t[0].expand(4)
    xN = scheduler.init_latents((3, 64, 64), tN)

    solver = TrapezoidalSolver(denoiser, num_inner_iters=3)
    trajectory = sample(
        denoiser, xN, scheduler,
        num_steps=0,            # Ignored when time_steps is provided
        time_steps=custom_t,
        solver=solver,
        time_eval=[0, 4, 7],    # Collect snapshots at steps 0, 4, 7
    )
    # trajectory is a list of 3 tensors, each of shape (4, 3, 64, 64)


.. _diffusion_available_solvers:

Available Solvers
-----------------

Solvers implement the :math:`\text{Step}` operator in the
:ref:`sampling equation <diffusion_sampling_equation>`.  At each iteration,
the solver receives the denoiser output :math:`D(\mathbf{x}_n, t_n)` and
advances the latent state from time :math:`t_n` to :math:`t_{n-1}`.

There are two ways to use solvers:

**Built-in solvers** can be selected by passing a string key to :func:`sample`:

- ``"euler"`` --- :class:`~physicsnemo.diffusion.samplers.solvers.EulerSolver`.
  First-order.  Fast (one denoiser evaluation per step) but lower quality.
- ``"heun"`` --- :class:`~physicsnemo.diffusion.samplers.solvers.HeunSolver`.
  Second-order.  Higher quality but twice as expensive per step.
- ``"edm_stochastic_euler"`` ---
  :class:`~physicsnemo.diffusion.samplers.solvers.EDMStochasticEulerSolver`.
  First-order with configurable stochastic noise injection.
- ``"edm_stochastic_heun"`` ---
  :class:`~physicsnemo.diffusion.samplers.solvers.EDMStochasticHeunSolver`.
  Second-order with configurable stochastic noise injection.

**Custom solvers** can be defined by implementing the
:class:`~physicsnemo.diffusion.samplers.solvers.Solver` protocol: any object
with a ``step(x, t_cur, t_next)`` method.  Pass the instance directly to
:func:`sample` for full control over the integration method.


.. _diffusion_guidance:

Guidance
--------

Guidance techniques modify the **predictor** :math:`P` in the
:ref:`sampling equation <diffusion_sampling_equation>` to steer the generated
samples toward desired properties, such as consistency with observed data,
satisfaction of physical constraints, or other task-specific objectives.

In the framework, guidance operates at the
:class:`~physicsnemo.diffusion.Predictor` level: you compose or modify
predictors *before* converting them into a
:class:`~physicsnemo.diffusion.Denoiser` :math:`D`.  Concretely, guidance
objects implement the :class:`~physicsnemo.diffusion.guidance.DPSGuidance`
protocol and are combined with an x0-predictor into a guided score-predictor
via :class:`~physicsnemo.diffusion.guidance.DPSScorePredictor`.  The resulting
score-predictor is then passed to the noise scheduler's
:meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
factory to obtain a denoiser :math:`D` for sampling.  The sampler and solver
(:math:`\text{Step}`) are unchanged---they only see the final denoiser.

The framework provides two ready-to-use guidance implementations:

- :class:`~physicsnemo.diffusion.guidance.DataConsistencyDPSGuidance` ---
  For masked observations (inpainting, sparse probes, data assimilation).
- :class:`~physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance` ---
  For generic (potentially nonlinear) observation operators.

Custom guidances can be defined by implementing the
:class:`~physicsnemo.diffusion.guidance.DPSGuidance` protocol---any callable
with the signature ``(x, t, x_0) -> guidance_term``.  Multiple guidances can
be combined by passing a list to
:class:`~physicsnemo.diffusion.guidance.DPSScorePredictor`.


API Reference
-------------

Sample Entry Point
~~~~~~~~~~~~~~~~~~

:code:`sample`
^^^^^^^^^^^^^^

.. autofunction:: physicsnemo.diffusion.samplers.sample

Solvers
~~~~~~~

:code:`Solver`
^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.diffusion.samplers.solvers.Solver
    :members:
    :exclude-members: __init__

:code:`EulerSolver`
^^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.diffusion.samplers.solvers.EulerSolver
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`HeunSolver`
^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.diffusion.samplers.solvers.HeunSolver
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`EDMStochasticEulerSolver`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.diffusion.samplers.solvers.EDMStochasticEulerSolver
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`EDMStochasticHeunSolver`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.diffusion.samplers.solvers.EDMStochasticHeunSolver
    :show-inheritance:
    :members:
    :exclude-members: __init__

Guidance
~~~~~~~~

:code:`DPSGuidance`
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.diffusion.guidance.DPSGuidance
    :members:
    :exclude-members: __init__

:code:`DPSScorePredictor`
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.diffusion.guidance.DPSScorePredictor
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`ModelConsistencyDPSGuidance`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.diffusion.guidance.ModelConsistencyDPSGuidance
    :show-inheritance:
    :members:
    :exclude-members: __init__

:code:`DataConsistencyDPSGuidance`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.diffusion.guidance.DataConsistencyDPSGuidance
    :show-inheritance:
    :members:
    :exclude-members: __init__

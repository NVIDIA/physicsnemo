.. _diffusion_introduction:

Introduction
============

.. currentmodule:: physicsnemo.diffusion

The PhysicsNeMo diffusion framework provides a modular, composable toolkit for
building, training, and sampling from diffusion models.  It is designed for
scientists and engineers who want to apply diffusion-based generative modeling
to real-world problems, from weather forecasting and climate downscaling to
geophysical inversion and materials design, while remaining flexible enough for
research-level experimentation.

Diffusion models learn to generate data by reversing a gradual noising process.
During training, the model sees data that has been corrupted by varying amounts
of noise, and it learns to predict the clean data (or, equivalently, the noise
or the score) from the corrupted version.  At inference time, the model starts
from pure noise and iteratively removes it, step by step, to produce a new
sample.  

This basic recipe turns out to be remarkably powerful. Diffusion models achieve state-of-the-art results in image generation, and, therefore, are
increasingly used for scientific applications where the goal is to sample from
complex, high-dimensional distributions conditioned on physical observations
or constraints.

The framework is organized around a small number of clearly defined
abstractions.  Each abstraction maps to a specific role in the diffusion
pipeline and the abstractions compose naturally. The abstractions include: 

* a :ref:`noise scheduler <diffusion_noise_schedulers>` to control the forward and reverse processes 
* a :ref:`model backbone <diffusion_model_backbones>` to implement the neural network 
* a :ref:`preconditioner <diffusion_preconditioners>` to rescale model inputs and outputs for stable training 
* a :ref:`loss function <diffusion_metrics>` to define the training objective 
* a :ref:`sampler <diffusion_samplers>` to generate new data at inference time 
* a :ref:`guidance <diffusion_guidance>` to steer the sampling toward desired properties


Framework Components at a Glance
---------------------------------

The table below summarizes the main framework components and their role in
training and inference.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Component
     - Training
     - Inference
   * - :ref:`Noise schedulers <diffusion_noise_schedulers>`
     - Sample times, add noise, compute loss weights
     - Generate time-steps, initialize latent state, build denoisers
   * - :ref:`Model backbones <diffusion_model_backbones>`
     - Combined with preconditioner and losses to learn to denoise
     - Serves as the predictor
   * - :ref:`Preconditioners <diffusion_preconditioners>`
     - Optionally rescales inputs/outputs for stable training
     - Acts as the predictor passed to the denoiser factory
   * - :ref:`Losses <diffusion_metrics>`
     - Denoising score matching objective
     -
   * - :ref:`Samplers and solvers <diffusion_samplers>`
     -
     - Numerically integrate the reverse process
   * - :ref:`Guidance <diffusion_guidance>`
     -
     - Steer sampling toward observations or constraints
   * - :ref:`Multi-diffusion <diffusion_multi_diffusion>`
     - Patch-based model wrapper and losses
     - Patch-based denoiser and guidance

The :ref:`noise scheduler <diffusion_noise_schedulers>` is a particularly central
component that is used in *both* stages.  Its role loosely parallels that of
the `Scheduler <https://huggingface.co/docs/diffusers/api/schedulers/overview>`_
in HuggingFace Diffusers, which similarly encapsulates the noise schedule and
provides methods for both training and inference.  A key difference is that the
PhysicsNeMo noise scheduler also owns the
:meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
factory, which converts a predictor into a denoiser suitable for the chosen
solver.


Design Philosophy: Layered Customization
-----------------------------------------

Diffusion models are used across a wide spectrum of applications in scientific
machine learning and physics-AI, by users with very different needs, for example: 

* diffusion experts who require full control over the forward process, the solver, or the guidance mechanism
* domain experts in science and engineering who use diffusion as a tool and need reliable, easy-to-use components  

The framework is designed to serve both audiences.

A central design principle of the framework is to offer multiple levels of
customization, so that users can choose the trade-off between convenience and
flexibility that best fits their needs.  Not every abstraction exposes all
levels, the exact set depends on the component, but the underlying philosophy
is consistent throughout:

**Protocols (maximum flexibility).**
Where appropriate, an abstraction defines a minimal *protocol* (a set of
method signatures with no implementation).  Any object that satisfies the
protocol can be used within the framework.  This is the right level when you
want to drop in a fully custom component, for instance, a noise scheduler for
a non-Gaussian forward process, or a custom solver.

.. code-block:: python

    # Any class with the right methods satisfies the NoiseScheduler protocol
    from physicsnemo.diffusion.noise_schedulers import NoiseScheduler

    class MyCustomScheduler:
        def sample_time(self, N, *, device=None, dtype=None): ...
        def add_noise(self, x0, time): ...
        def timesteps(self, num_steps, *, device=None, dtype=None): ...
        def init_latents(self, spatial_shape, tN, *, device=None, dtype=None): ...
        def get_denoiser(self, **kwargs): ...
        def loss_weight(self, t): ...

    assert isinstance(MyCustomScheduler(), NoiseScheduler)  # True

**Abstract base classes (structured extensibility).**
Some components provide abstract base classes that implement shared logic.
Subclasses only override a few methods to define their variant.  For example,
:class:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler`
handles noise injection, score conversion, and denoiser construction for any
linear-Gaussian schedule. You just define :math:`\alpha(t)`, :math:`\sigma(t)`,
and the discretization.

.. code-block:: python

    from physicsnemo.diffusion.noise_schedulers import LinearGaussianNoiseScheduler

    class MyScheduler(LinearGaussianNoiseScheduler):
        def sigma(self, t): return t
        def sigma_inv(self, sigma): return sigma
        def sigma_dot(self, t): return torch.ones_like(t)
        def alpha(self, t): return torch.ones_like(t)
        def alpha_dot(self, t): return torch.zeros_like(t)
        def timesteps(self, num_steps, *, device=None, dtype=None): ...
        def sample_time(self, N, *, device=None, dtype=None): ...
        def loss_weight(self, t): ...

**Ready-to-use components (zero boilerplate).**
The framework ships fully configured components that work out of the box.
Most can still be subclassed for light customization.

.. code-block:: python

    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler

    scheduler = EDMNoiseScheduler(sigma_min=0.002, sigma_max=80.0, rho=7)

Each component's documentation describes which of these levels it supports.
For example, :ref:`noise schedulers <diffusion_noise_schedulers>` offer all
three levels, while :ref:`solvers <diffusion_samplers>` provide a protocol and
concrete implementations, and :ref:`preconditioners <diffusion_preconditioners>`
provide an abstract base class and concrete implementations.


Core Concepts: DiffusionModel, Predictor, and Denoiser
------------------------------------------------------

The framework defines three protocol classes that capture the key signatures
involved in a diffusion pipeline.  Understanding the distinction between them
is essential.

**DiffusionModel** (:class:`DiffusionModel`)
    The interface for models during **training**.  A :class:`DiffusionModel`
    takes the noisy state :math:`\mathbf{x}_t`, the diffusion time :math:`t`,
    and optional conditioning information, and returns a prediction.  The
    prediction target can be anything: clean data
    :math:`\hat{\mathbf{x}}_0`, score
    :math:`\nabla_{\mathbf{x}} \log p(\mathbf{x})`, noise
    :math:`\boldsymbol{\epsilon}`, or velocity :math:`\mathbf{v}`.
    Which target the model predicts depends on the training objective and
    the choice of preconditioner.

**Predictor** (:class:`Predictor`)
    The interface for trained models during **inference**.  A
    :class:`Predictor` is a callable ``(x, t) -> prediction`` that does not
    require conditioning as a separate argument.  A predictor is typically
    obtained from a :class:`DiffusionModel` by binding the conditioning using
    ``functools.partial``, but it can be anything: an x0-predictor, a
    score-predictor, a guidance-augmented predictor, or any combination.
    The type of prediction a predictor returns depends on how the underlying
    model was trained.  Importantly, not all predictors originate from a
    trained model. For example, DPS-style :ref:`guidance <diffusion_guidance>` predictors
    are computed on the fly during sampling and always produce a score.
    Although all predictors share the same ``(x, t) -> prediction`` signature,
    it is your responsibility to know what kind of prediction is
    returned and to use it accordingly (for example, passing an x0-predictor versus a
    score-predictor to the noise scheduler's ``get_denoiser`` factory).

**Denoiser** (:class:`Denoiser`)
    The update function consumed by
    :ref:`solvers <diffusion_samplers>` during sampling.  A denoiser is
    obtained from a predictor using the noise scheduler's
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
    factory.  For more details on how the denoiser fits in the sampling
    loop, refer to the :ref:`samplers documentation <diffusion_samplers>`.

These three types form a pipeline:

.. code-block:: text

    Training:   data  ->  NoiseScheduler.add_noise  ->  DiffusionModel  ->  Loss

    Inference:  DiffusionModel  ->  partial(...) / closure  ->  Predictor
                Predictor  ->  (optional: + Guidance)       ->  Predictor
                Predictor  ->  (optional: x0 <-> score)     ->  Predictor
                Predictor  ->  NoiseScheduler.get_denoiser  ->  Denoiser
                Denoiser  ->  Solver.step  (sample loop)    ->  samples

In the inference pipeline, ``partial(...)`` binds the conditioning into the
predictor (a closure or wrapper function achieves the same result).
:ref:`Guidance <diffusion_guidance>` is optionally composed at the predictor
level---for example, DPS guidance combines an x0-predictor with observation
constraints to produce a guided score-predictor.  When the predictor type does
not match what the denoiser factory expects (for example, the model was trained as an
x0-predictor but guidance produces a score), the noise scheduler can convert
between the two via ``x0_to_score`` / ``score_to_x0``.


Prediction Types
----------------

Diffusion models can be trained to predict different targets.  The PhysicsNeMo
framework currently supports two prediction types:

- **x0-predictor**: The model estimates the clean data
  :math:`\hat{\mathbf{x}}_0` from the noisy state :math:`\mathbf{x}_t`.
  This is the most common choice when using a
  :ref:`preconditioner <diffusion_preconditioners>`.

- **Score-predictor**: The model estimates the score function
  :math:`\nabla_{\mathbf{x}} \log p(\mathbf{x}_t)`.

For linear-Gaussian noise schedules, these two representations are analytically
interchangeable, and the framework handles the conversion internally when
building a denoiser for sampling.  For other schedule families, the conversion
depends on the specific formulation and may need to be handled by the noise
scheduler implementation.

Other prediction types (noise-predictor, velocity-predictor) can be supported
by implementing a custom :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
that handles the appropriate conversions in its
:meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser` method.


API Reference
-------------

:code:`DiffusionModel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.DiffusionModel
    :members:
    :exclude-members: __init__

:code:`Predictor`
~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.Predictor
    :members:
    :exclude-members: __init__

:code:`Denoiser`
~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.Denoiser
    :members:
    :exclude-members: __init__

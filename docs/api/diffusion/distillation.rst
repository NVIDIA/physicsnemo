Distillation losses
===================

.. warning::

   This API is alpha. Imports emit ``ExperimentalFeatureWarning`` and future
   releases may introduce changes that break backward compatibility.

``DistillationLoss`` computes sCM, MeanFlow, DMD2, or f-distill losses for
PhysicsNeMo models. Each call constructs training targets and returns a
differentiable ``total_loss`` with detached values for logging.

Use it in your training loop with initialized models and a noise scheduler.
Your loop controls backpropagation, optimizer updates, and checkpoints.
Use PhysicsNeMo samplers for inference.

Install the backend
-------------------

Install the optional ``diffusion-fastgen`` extra from this checkout:

.. code-block:: bash

   pip install -e '.[diffusion-fastgen]'

This extra installs the FastGen version required by the built-in losses.
You do not need FastGen to write your own loss using ``BaseDistillationLoss``.

Train a student
---------------

Give your student the :class:`~physicsnemo.diffusion.base.DiffusionModel`
interface. For MeanFlow, it receives ``(t, DiffusionExtraTimes(r=r, batch_size=[B]))``
and predicts average velocity between these times. A teacher supplies
instantaneous velocity at ``t``. Set its ``prediction_type`` if it predicts
clean data, noise, or score instead.

.. code-block:: python

   from physicsnemo.diffusion.distillation import DistillationLoss
   from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler

   loss_fn = DistillationLoss(
       student,
       RectifiedFlowNoiseScheduler(),
       method="MeanFlow",
       teacher=teacher,
       model_options={"teacher": {"prediction_type": "flow"}},
   )
   optimizer.zero_grad()
   losses = loss_fn(x0, condition=condition, step=step)
   losses["total_loss"].backward()
   optimizer.step()

Omit ``teacher`` to learn from the conditional velocity of paired data and noise.
Backpropagate through ``total_loss`` only. Other entries are for logging and
may duplicate parts of the total. ``return_outputs=True`` also returns
detached predictions and times without drawing extra samples.

For sCM, select ``method="sCM"`` and return a prediction tensor paired with
``DiffusionExtraOutputs(logvar=logvar, batch_size=[B])``. The prediction is
always required. The ``logvar`` tensor has shape ``[B]`` and controls
adaptive loss weighting. It does not describe diffusion posterior variance.
The default student prediction type is ``"x0"``. A
:class:`~physicsnemo.diffusion.preconditioners.TrigFlowPreconditioner` can provide
this prediction from a normalized-velocity backbone.

Control models and times
------------------------

``model_options`` accepts role-specific mappings for ``student`` and ``teacher``.
Use ``prediction_type="x0"``, ``"epsilon"``, ``"score"``, or ``"flow"`` to
describe the model output. ``model_kwargs`` forwards extra model controls.
For example, a model may need ``{"return_logvar": True}`` to produce ``logvar``.

MeanFlow students receive both times in a tuple. Other model calls use a
single time tensor. The extra-time container supplies only the
reserved ``r`` field. General custom-field propagation remains future work.

To use a different noise schedule for the teacher, set
``model_options["teacher"]["noise_scheduler"]``. The loss matches noise levels
and converts predictions between schedules for you. The teacher's scheduler
must provide ``snr_inv`` for this conversion.

``method_options["guidance_scale"]`` enables classifier-free guidance. MeanFlow
applies it at all diffusion times by default. Use
``guidance_time_bounds=(lo, hi)`` to restrict it to an interval.

By default, the loss uses the scheduler's ``sample_time`` distribution. Pass
``t=...`` to choose diffusion times directly. For MeanFlow, pass
``(t, DiffusionExtraTimes(r=r, batch_size=[B]))``; the loss uses this pair as given.
You can also supply ``sample_t_fn(N, *, device, dtype)`` at call time. This
callback returns the same time structure and applies only to that call.

To request a bounded distribution, set
``method_options={"sampling": {"time_dist_type": "uniform"}}``. Other choices
are ``"lognormal"``, ``"logitnormal"``, and ``"log_t"``. Bounds follow live
attributes of built-in schedulers. For custom schedulers, provide training
limits through ``model_options["student"]["scheduler_options"]["time_bounds"]``
when the selected method or sampling policy needs them. sCM also needs a differentiable
``snr_inv`` implementation.

Mask targets and weight residuals
--------------------------------

``WeightedDistillationLoss`` adds masks and element weights to the standard
loss. Zero weights exclude entries from residuals, adaptive norms, and
effective dimensions. Positive weights multiply the final squared residual
once and leave the adaptive statistics unchanged. Mean reductions keep the
full element count, matching ``WeightedMSEDSMLoss``. Samples with no positive
weights contribute zero, including the sCM log-variance term.

Use the same constructor as ``DistillationLoss`` and pass ``weight`` at call
time. Weights must be finite, nonnegative, and broadcast to the data shape.

.. code-block:: python

   from physicsnemo.diffusion.distillation import WeightedDistillationLoss

   loss_fn = WeightedDistillationLoss(student, scheduler, method="MeanFlow")
   losses = loss_fn(x0, weight=valid_pixels, step=step)
   losses["total_loss"].backward()

Replace invalid input values before calling the model. A loss mask cannot
prevent corrupted inputs from affecting predictions at other locations.

Compilation and state
---------------------

To compile sCM or MeanFlow training, apply ``torch.compile`` to the enclosing
loss calculation. Compiling only the model can fail when the loss computes
an exact Jacobian-vector product (JVP). If you compile only the model, select
``method_options={"use_jvp_finite_diff": True}`` to approximate the derivative
with finite differences instead.

With exact derivatives, MeanFlow updates stateful layers once per prediction
and shares random draws between the prediction and its derivative. These calculations use
the math attention backend because fused kernels may not support these
derivatives. Check your combination of compilation, mixed precision, and
distributed wrapping on your target hardware before a long training run.

Save your PhysicsNeMo models and optimizers as usual. Recreate the loss with
the same noise schedules and settings when resuming. sCM and MeanFlow have no
extra running statistics, so their ``state_dict`` results are empty.

Alternate distribution-matching objectives
-----------------------------------------

DMD2 and f-distill match a student's generated distribution to a fixed
diffusion teacher. Unlike an ordinary denoising loss that trains a model on
clean data and noise, these objectives also train a fake-score model on
student-generated samples.

Initialize the models and pass the fake-score model in ``auxiliary_models``.
Choose ``phase="student"`` or ``phase="auxiliary"`` on each call and step
the matching optimizer. Pass the training step separately as ``step``.

.. code-block:: python

   loss_fn = DistillationLoss(
       student, scheduler, method="DMD2", teacher=teacher,
       auxiliary_models={"fake_score": fake_score},
   )
   for phase, optimizer in (("auxiliary", fake_optimizer),
                            ("student", student_optimizer)):
       optimizer.zero_grad()
       loss_fn(x0, phase=phase, step=step)["total_loss"].backward()
       optimizer.step()

The student phase trains only the student. The auxiliary phase trains the
fake-score model and any discriminator. The teacher stays fixed in both phases.
Use separate parameters for each model. The loss preserves your choices of
training mode and frozen parameters.

DMD2 trains the student using teacher and fake-score predictions, with an
optional adversarial loss. f-distill also weights this score
matching by a selected f-divergence and requires a discriminator.

Supply ``discriminator`` in ``auxiliary_models`` to enable adversarial training.
The generator GAN coefficient defaults to ``0.001``. Set
``gan_loss_weight_gen`` to change it. f-distill requires a positive coefficient.
The discriminator accepts a TensorDict of named teacher features and returns
scores with shape :math:`(B, L)`, where :math:`L` counts scores per sample.
Supply a ``feature_fn(prediction, extras)`` callback to select those features
from your teacher output. The teacher must still return its prediction.
Feature fields are model-specific, not reserved diffusion outputs. The callback
interprets them. For example, if your teacher adds a
``bottleneck`` field to a subclass of ``DiffusionExtraOutputs``:

.. code-block:: python

   from tensordict import TensorDict

   def teacher_features(prediction, extras):
       return TensorDict({"bottleneck": extras.bottleneck}, batch_size=extras.batch_size)

   loss_fn = DistillationLoss(
       student, scheduler, method="f-distill", teacher=teacher,
       auxiliary_models={"fake_score": fake_score,
                         "discriminator": discriminator},
       model_options={"teacher": {"feature_names": ("bottleneck",),
                                  "feature_fn": teacher_features}},
       method_options={"f_distill": {"f_div": "js"}},
   )

Supported divergence names are ``rkl``, ``kl``, ``js``, ``sf``, ``neyman``,
``sh``, and ``jf``. f-distill tracks running ratio statistics in the loss object.
Save and restore ``loss_fn.state_dict()`` alongside model and optimizer state.
Pass ``update_state=False`` to compute a loss without changing these statistics.

``model_options["fake_score"]`` configures its prediction type or scheduler.
The loss converts predictions when these models use different noise schedules.
Use ``fake_score_pred_type`` to select the fake-score training target and
``fake_score_sampling`` to choose its training-time distribution.

One-step training draws student inputs at the scheduler's upper training time.
Custom schedulers without that bound need explicit ``input_student`` and
``t_student`` on the loss call, or configured training bounds. Multi-step training
uses ``student_sample_steps`` and an explicit ``training_times`` grid under the
student's ``scheduler_options``. Specify training times independently of your
inference grid.
Pass ``noise`` to control the score perturbation independently.

Adversarial masking
~~~~~~~~~~~~~~~~~~~

Adversarial masks exclude corrupted inputs during feature extraction and
discriminator pooling. The loss needs a callback that maps the input mask to
valid discriminator scores.

For masked adversarial training, supply a callback with this signature:

.. code-block:: python

   def adversarial_fn(x_t, t, condition, weight):
       ...
       return logits, logit_weight

Return ``logits`` with shape :math:`(B, L)` and finite,
nonnegative ``logit_weight`` values that broadcast to that shape.

Use the same teacher and discriminator you passed to the loss, and keep
gradients through their inputs for student training. You may need a masked
backbone or feature extractor. Masking only the final scores cannot remove
corruption that has already spread through features or pooled statistics.

The loss applies the returned weights to generator, discriminator, and approximate
R1 residuals. Zero weights also exclude scores and empty samples from f-distill
ratio statistics. Positive residual weights remain linear. The callback receives
both real and generated perturbations, including the R1 query when enabled.

Distributed statistics and compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

f-distill reduces ratio statistics over the default process group for ordinary
data parallelism. For a domain-parallel scheduler, explicitly supply the data
group through ``method_options["f_distill"]["statistics_group"]``. It must
group corresponding spatial partitions across independent data replicas, not
different partitions of the same sample. Each rank participates even when its
local batch has no valid entries. The callback must produce consistent global scores
across spatial partitions when the architecture needs spatial aggregation.

For DMD2, you can compile the individual models with ``torch.compile``.
Keep phase selection and f-distill statistics updates outside compilation.
Check combined domain parallelism and FSDP on your target hardware before
a long training run.

Custom implementations
----------------------

``BaseDistillationLoss`` defines the callable loss interface independently
of the built-in objectives. A custom loss can be a function or callable class.
Inheritance and FastGen are optional.

Define ``__call__(x0, *, condition=None, **kwargs)`` and return a dictionary
containing a differentiable ``total_loss``. Add named entries for logging or
extra arguments for your method as needed. The training loop computes gradients
through ``total_loss`` and updates the model.

API reference
-------------

.. automodule:: physicsnemo.diffusion.distillation
   :members:
   :show-inheritance:

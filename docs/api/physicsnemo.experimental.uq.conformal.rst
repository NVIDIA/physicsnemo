PhysicsNeMo Conformal Prediction
================================

.. currentmodule:: physicsnemo.experimental.uq.conformal

This module quantifies uncertainty after training, with conformal
prediction. It turns the outputs of a trained model into calibrated
prediction intervals. You do not need to retrain the model.

.. warning::

    This module is experimental. Its API can change without a deprecation
    period.

.. seealso::

    The `Uncertainty Quantification <https://docs.nvidia.com/physicsnemo/latest/user-guide/uncertainty_quantification.html>`_ user guide covers the
    PhysicsNeMo models that estimate their own uncertainty, such as Gaussian
    process heads, concrete dropout, and ensembles. Conformal prediction can
    calibrate their intervals after training.

Overview
--------

A trained model returns a prediction, such as a scalar, a vector, or a
tensor at each of many points. It does not say how far that prediction can
be from the truth. Conformal
prediction measures the model's errors on a held-out *calibration set* and
picks a threshold from those errors. It then turns each new prediction into
bounds ``(lo, hi)`` that contain the true value with a stated probability.

The guarantee holds for any model and any data distribution. It needs one
assumption. The calibration samples and the new samples are
*exchangeable*, for example drawn independently from the same distribution.
The guarantee no longer holds if the deployment data comes from a
different distribution than the calibration data, for example geometries or
operating conditions outside the calibration range.

The workflow has three steps.

1. Pass each calibration sample, the prediction and the target, to a
   calibrator's ``update``.
2. Call ``finalize`` to fit the thresholds and get a
   :class:`ConformalPredictor`.
3. Call ``predictor.predict_interval`` on each new model output to get
   ``(lo, hi)``.

You can save the predictor, load it elsewhere, and check it on held-out
data with a :class:`CoverageAccumulator`.


Quick Start
-----------

.. code:: python

    from physicsnemo.experimental.uq.conformal import (
        AbsoluteErrorScore,
        RiskControlCalibrator,
    )

    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.1)
    for prediction, target in calibration_set:  # samples can differ in size
        calibrator.update(prediction, target)
    predictor = calibrator.finalize()

    lo, hi = predictor.predict_interval(model(new_input))

With ``alpha=0.1``, on average at most 10% of a new sample's points fall
outside ``[lo, hi]``. ``prediction`` and ``target`` share a shape
:math:`(n_{\text{points}}, *\text{dims})`. A scalar observable has shape
``(1,)`` and a surface field with four channels has shape ``(n_points, 4)``.


Choosing a Calibrator
---------------------

The calibrator sets what the guarantee covers and how the band width varies
across the output. Pick it by the guarantee your application needs and by
whether every sample has the same output layout.

.. list-table:: Calibrator Guarantees
   :header-rows: 1
   :widths: 18 30 34 18

   * - Calibrator
     - With ``alpha=0.1``
     - Band width across the output
     - Layout
   * - :class:`PointwiseCalibrator`
     - Each output element is covered with probability at least 90%, one
       element at a time.
     - One width per element, fitted from that element's calibration
       errors, so the band follows the model's local error.
     - One layout, and the same mesh if ``points`` is given.
   * - :class:`FunctionalBandCalibrator`
     - The whole sample is covered at once with probability at least 90%.
     - One threshold times the per-point scale from the score (sigma for
       ``NormalizedErrorScore``) or from an ``AuxScale``. Constant with
       ``AbsoluteErrorScore`` alone.
     - Can differ.
   * - :class:`RiskControlCalibrator`
     - The fraction of missed points is at most 10% in expectation. A
       single case can exceed it.
     - Same as the functional band, and narrower.
     - Can differ.

Start with :class:`RiskControlCalibrator` when the output size changes
between samples, such as CFD results on different geometries. Use
:class:`FunctionalBandCalibrator` when a single miss anywhere in the sample
matters, for example to bound the worst-case error over a surface. Both
treat a sample as a whole. The functional band must hold at every point and
every channel. Risk control counts a point as missed when any of its
channels falls outside the band. Separate ``TensorDict`` fields get separate
thresholds and separate guarantees.

Use :class:`PointwiseCalibrator` when every sample has the same output
layout and you need an interval for each output element. The layout can be
a grid, a mesh, or a fixed set of outputs such as drag, lift, and moment
coefficients. For a mesh, pass the coordinates as ``points=`` on every
call. The calibrator and predictor then reject any other mesh.

Non-Uniform Meshes
^^^^^^^^^^^^^^^^^^

Risk control counts every point equally, so on a mesh that is fine in one
region and coarse in another, the fraction of missed points is not the
fraction of missed area. Pass ``weights=`` to give each point its own
weight. Any nonnegative per-point weights work, for example cell areas or
volumes, or an importance you choose for each region.

.. code:: python

    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.1)
    for prediction, target, areas in calibration_set:
        calibrator.update(prediction, target, weights=areas)
    predictor = calibrator.finalize()

``weights`` has shape ``(n_points,)`` and holds nonnegative, finite values
that are not all zero. The calibrator normalizes them within each sample.
The weights change what counts as missed, not the band itself. The
guarantee then bounds the expected weighted fraction of missed points by
``alpha``.

If you use ``physicsnemo.mesh``, its per-point area or volume makes a good
set of weights. ``physicsnemo.mesh.calculus.point_measures(mesh)`` returns
it when the mesh already stores point measures.
``physicsnemo.mesh.calculus.lumped_point_measures(mesh)`` computes it from
the cells.

Pass ``weights`` on every call or on none. The predictor records
that it was calibrated with weights, and ``save`` and ``load`` keep that
record. Its ``coverage_accumulator()`` then requires ``weights`` on every
``update(lo, hi, target, weights=...)`` so the report measures the weighted
risk.

Several Coverage Levels
^^^^^^^^^^^^^^^^^^^^^^^

``finalize(alpha=...)`` fits the thresholds at another level from the stored
calibration scores, so one pass over the calibration set gives predictors at
several levels.

.. code:: python

    predictor_80 = calibrator.finalize(alpha=0.2)
    predictor_90 = calibrator.finalize()  # the constructor's alpha
    predictor_95 = calibrator.finalize(alpha=0.05)

Each predictor has its own ``alpha`` and its own guarantee. Choose the
levels before looking at the fitted intervals. Picking ``alpha`` from the
results weakens the guarantee. The calibration set size rule in Practical
Notes applies to each level, so ``alpha=0.05`` needs at least 19 samples.


Choosing a Score
----------------

The score sets the shape of the interval. Pick it by what the model
outputs. The score reads any extra model outputs from the ``aux``
argument.

.. list-table:: Built-In Scores
   :header-rows: 1
   :widths: 28 30 22 20

   * - Score
     - Model output
     - ``aux`` entries
     - Interval
   * - :class:`AbsoluteErrorScore`
     - A point prediction :math:`\hat y`
     - None
     - :math:`[\hat y - t,\ \hat y + t]`
   * - :class:`NormalizedErrorScore`
     - A mean :math:`\mu` and a standard deviation :math:`\sigma`
     - ``"sigma"``
     - :math:`[\mu - t\sigma,\ \mu + t\sigma]`
   * - :class:`QuantileRegressionScore`
     - Lower and upper quantile heads
     - ``"lo"``, ``"hi"``
     - :math:`[q_{lo} - t,\ q_{hi} + t]`

:math:`t` is the threshold the calibrator fits. :class:`NormalizedErrorScore`
gives narrow intervals where the model is confident and wide intervals
where it is not. It is the better choice when the model has a
:math:`\sigma` head, an MC-dropout spread, or an ensemble spread.

Pass ``sigma`` the same way at calibration and at prediction.

.. code:: python

    from physicsnemo.experimental.uq.conformal import (
        NormalizedErrorScore,
        RiskControlCalibrator,
    )

    calibrator = RiskControlCalibrator(NormalizedErrorScore(), alpha=0.1)
    for prediction, target, std in calibration_set:
        calibrator.update(prediction, target, aux={"sigma": std})
    predictor = calibrator.finalize()

    lo, hi = predictor.predict_interval(new_prediction, aux={"sigma": new_std})

:class:`FunctionalBandCalibrator` and :class:`RiskControlCalibrator` also
accept an :class:`AuxScale` as ``scale=``, which scales the
threshold by a per-point uncertainty read from ``aux``. Pair it with
:class:`AbsoluteErrorScore`. Pairing ``AuxScale("sigma")`` with
:class:`NormalizedErrorScore` would scale by :math:`\sigma` twice, so the
calibrators reject that pair.


Calibrating Existing Uncertainty
--------------------------------

The score decides what conformal prediction calibrates. A deterministic
model has no spread of its own, so use :class:`AbsoluteErrorScore`. The
interval is then the prediction plus or minus a fitted width.

Some models estimate their own uncertainty, such as a Gaussian process
head, concrete dropout, or an ensemble. These models supply a standard
deviation for each output. Pass it as ``aux["sigma"]`` with
:class:`NormalizedErrorScore`. With the functional band or risk-control
calibrator, you can instead pass it to an ``AuxScale`` and use
:class:`AbsoluteErrorScore`. The interval keeps the model's shape and stays
wide where the model is unsure. The calibrator fits how much to widen or
shrink it. For a ``FieldVariationalGPHead``, ``sigma`` is
``output.variance.sqrt()`` from the ``FieldVariationalGPPrediction`` it
returns. For an ensemble or repeated dropout passes, ``sigma`` is the
standard deviation across members or passes. The
`Uncertainty Quantification <https://docs.nvidia.com/physicsnemo/latest/user-guide/uncertainty_quantification.html>`_ user guide notes that these
spreads often fall short of the observed error. Conformal calibration
corrects that shortfall on held-out data.

A simpler post hoc option is standard-deviation rescaling. It multiplies
every ``sigma`` by one factor, chosen so that the errors divided by
``sigma`` have unit spread on held-out data. Both methods use held-out data,
but they promise different things. Rescaling matches the spread on average.
For example, 90% of true values fall inside ``mu +/- 1.645 * sigma`` only if
the errors are Gaussian with that spread. A conformal interval contains the
true value with probability at least :math:`1 - \alpha` for any model and
any data distribution, given exchangeable data. For risk control, it keeps
the expected missed fraction at or below :math:`\alpha`.


Custom Scores
-------------

Write a custom score when no built-in score gives the interval shape you
need, for example a band proportional to the prediction. You do not need
one to widen the band where the model is uncertain.
:class:`NormalizedErrorScore` divides the error by a predicted ``sigma``,
and ``scale=AuxScale(key)`` multiplies a functional or risk-control band by
any per-point ``aux`` entry. A custom score can shape the interval element
by element and works with every calibrator.

The Contract
^^^^^^^^^^^^

Subclass :class:`NonconformityScore`, implement two methods, and register
the class with :func:`register_score`. The two methods are the following:

- ``score(prediction, target, *, aux=None)`` returns the per-element score,
  where larger is worse.
- ``bounds(prediction, threshold, *, aux=None)`` returns ``(lo, hi)``, the
  targets whose score is at most ``threshold``.

The coverage guarantee holds only if the two methods agree. For every
element, the targets with ``score <= t`` are exactly the values in
``bounds(t)``, and that interval grows with ``t``. The example below is a
band proportional to the prediction.

.. code:: python

    from physicsnemo.experimental.uq.conformal import (
        NonconformityScore,
        RiskControlCalibrator,
        check_score,
        register_score,
    )

    @register_score("relative_error")
    class RelativeErrorScore(NonconformityScore):
        config_fields = ("floor",)  # saved with the predictor

        def __init__(self, floor=1e-6):
            self.floor = floor

        def score(self, prediction, target, *, aux=None):
            scale = prediction.abs().clamp_min(self.floor)
            return (target - prediction).abs() / scale

        def bounds(self, prediction, threshold, *, aux=None):
            half = threshold * prediction.abs().clamp_min(self.floor)
            return prediction - half, prediction + half

    check_score(RelativeErrorScore(), prediction, target)
    calibrator = RiskControlCalibrator(RelativeErrorScore(floor=1e-3), alpha=0.1)

Class Attributes
^^^^^^^^^^^^^^^^

Four class attributes tell the calibrators and predictors how to treat the
score. All default to empty or ``False``.

.. list-table:: Score Class Attributes
   :header-rows: 1
   :widths: 20 46 34

   * - Attribute
     - Meaning
     - Set it when
   * - ``aux_keys``
     - The ``aux`` entries the score reads. Calibrators and predictors
       require them and check that each is a finite tensor with the
       prediction's shape.
     - ``score`` or ``bounds`` reads anything from ``aux``.
   * - ``config_fields``
     - Constructor arguments, each stored as an attribute of the same name.
       A saved predictor records their values and passes them back to the
       constructor on load.
     - The constructor takes arguments. Their values must be ``bool``,
       ``int``, ``float``, or ``str``.
   * - ``signed_threshold``
     - Scores, and so the fitted threshold, can be negative. Otherwise
       predictors reject a negative threshold.
     - The score is signed, as in :class:`QuantileRegressionScore`, where a
       negative threshold shrinks the band.
   * - ``scale_aux_keys``
     - The ``aux`` entries the score divides by. Calibrators reject an
       :class:`AuxScale` that reads one of them, since the band would
       scale by it twice.
     - ``bounds`` multiplies the threshold by an ``aux`` entry.

A score that reads an ``aux`` entry lists it in ``aux_keys``, and also in
``scale_aux_keys`` if it divides by it. The example below scores the error
in units of a per-point tolerance set by the application.

.. code:: python

    import torch

    from physicsnemo.experimental.uq.conformal import (
        NonconformityScore,
        PointwiseCalibrator,
        register_score,
    )

    @register_score("tolerance_error")
    class ToleranceScore(NonconformityScore):
        aux_keys = ("tolerance",)
        scale_aux_keys = ("tolerance",)  # bounds multiply the threshold by it

        def score(self, prediction, target, *, aux=None):
            return (target - prediction).abs() / aux["tolerance"]

        def bounds(self, prediction, threshold, *, aux=None):
            half = threshold * aux["tolerance"].to(torch.float64)
            return prediction - half, prediction + half

    calibrator = PointwiseCalibrator(ToleranceScore(), alpha=0.1)
    for prediction, target, tolerance in calibration_set:
        calibrator.update(prediction, target, aux={"tolerance": tolerance})

What the Base Class Handles
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do not override ``interval``. It checks that every entry in ``aux_keys`` is
present, converts the prediction and the threshold to float64, calls
``bounds``, and rounds the interval outward to the prediction's dtype.
Low-precision inputs therefore keep the guarantee without any work in
``bounds``. The ``aux`` entries reach ``bounds`` in their original dtype.
Repeat any dtype-dependent step of ``score``, such as a clamp, before
converting them to float64, so the two methods agree.

Checking a Score
^^^^^^^^^^^^^^^^

:func:`check_score` uses each element's own score as its threshold and
raises ``ValueError`` if any target falls outside the resulting interval,
which means ``bounds`` does not invert ``score``. Every calibrator runs the
same check on its first sample and rejects a score that fails it. Call
``check_score`` in your tests on representative data, in each dtype you use,
with the ``aux`` entries the score reads.

Registration and Saved Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calibrators and predictors accept only registered classes, and a subclass
needs its own registration. A saved predictor stores the registered name
and the ``config_fields`` values, which must be ``bool``, ``int``, ``float``,
or ``str`` so the file loads with ``weights_only=True``. Import the module
that registers the class before :meth:`ConformalPredictor.load`. Otherwise
``load`` raises and lists the registered names. You can register the same
class again under the same name, for example when you rerun a notebook
cell. Registering a different class under a name in use raises
``ValueError``.


Multiple Fields
---------------

Pass a ``TensorDict`` to calibrate several fields at once. Each field gets
its own threshold and its own guarantee. Nest ``aux`` by field name.

.. code:: python

    from tensordict import TensorDict

    from physicsnemo.experimental.uq.conformal import (
        NormalizedErrorScore,
        RiskControlCalibrator,
    )

    calibrator = RiskControlCalibrator(NormalizedErrorScore(), alpha=0.1)
    for sample in calibration_set:
        n_points = sample["pressure"].shape[0]
        prediction = TensorDict(
            {"pressure": sample["pressure"], "wss": sample["wss"]},
            batch_size=[n_points],
        )
        target = TensorDict(
            {"pressure": sample["pressure_true"], "wss": sample["wss_true"]},
            batch_size=[n_points],
        )
        aux = {
            "pressure": {"sigma": sample["pressure_std"]},
            "wss": {"sigma": sample["wss_std"]},
        }
        calibrator.update(prediction, target, aux=aux)
    predictor = calibrator.finalize()

``predict_interval`` then returns ``lo`` and ``hi`` as ``TensorDict``
containers with the same fields. To calibrate only some fields, pass
``keys=["pressure"]`` to the calibrator.


Saving and Checking a Predictor
-------------------------------

A fitted predictor saves to a single file that loads with
``torch.load(..., weights_only=True)``. Check its coverage on held-out data
that you did not use for calibration. The coverage report measures the same
quantity the calibrator guarantees. This example continues the Quick Start
predictor, which uses :class:`AbsoluteErrorScore`. A predictor fitted with
:class:`NormalizedErrorScore` also needs ``aux={"sigma": ...}`` on every
``predict_interval`` call.

.. code:: python

    from physicsnemo.experimental.uq.conformal import ConformalPredictor

    predictor.save("predictor.pt", provenance={"dataset": "holdout-v1"})
    predictor = ConformalPredictor.load("predictor.pt", map_location="cuda")

    accumulator = predictor.coverage_accumulator()
    for prediction, target in test_set:
        lo, hi = predictor.predict_interval(prediction)
        accumulator.update(lo, hi, target)
    report = accumulator.finalize()

For a risk-control predictor, ``report["fields"]["tensor"]["empirical_mean_risk"]``
is the mean fraction of points per sample that fell outside the band. It
should be close to or below ``alpha``. The report lists a plain tensor
under the key ``"tensor"`` and ``TensorDict`` fields under their own names.


Practical Notes
---------------

**Calibration set size.**
    ``finalize`` needs at least :math:`(1 - \alpha) / \alpha` samples at the
    level it fits, for example 9 for ``alpha=0.1`` and 19 for
    ``alpha=0.05``. With more samples the fitted threshold varies less from
    one calibration set to the next.

**One call per sample.**
    Each ``update`` call adds one calibration sample, and the guarantee
    counts samples. If the model returns a batch, call ``update`` once for
    each item, not once for the batch.

    .. code:: python

        for prediction, target in zip(batch_prediction, batch_target):
            calibrator.update(prediction, target)

    The same applies to ``CoverageAccumulator.update``.

**Single process.**
    Calibration does not communicate across ranks. Predictions can come
    from many GPUs, but gather every calibration sample onto the process
    that calls ``finalize``, so there is one threshold. A rank that
    calibrates on its own share still gets a valid guarantee when the
    share was chosen independently of the data, for example by shard index.
    That guarantee rests on fewer samples, and every rank gets a different
    threshold. A share chosen by looking at the data, such as the hardest
    cases, has no guarantee.

**Memory.**
    :class:`PointwiseCalibrator` and :class:`RiskControlCalibrator` keep the
    calibration scores on the CPU until ``finalize``. Pointwise memory grows
    as :math:`n_{cal}` times the output size. Risk control stores one
    float64 per calibration point, two with ``weights``.
    :class:`FunctionalBandCalibrator` stores one number per sample.

**Devices and dtypes.**
    Inputs can live on any device and use any floating-point dtype,
    including ``float16`` and ``bfloat16``. Interval endpoints are computed
    in float64 and rounded outward to the input dtype, so low-precision
    inputs do not break the guarantee.

**Separate data for the uncertainty estimate.**
    Whatever produces ``sigma`` or the ``AuxScale`` values must not see
    the calibration samples. Fit it on training data or a separate split.


References
----------

- J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman,
  `Distribution-Free Predictive Inference for Regression
  <https://arxiv.org/abs/1604.04173>`_, 2018. Split conformal prediction.
- Y. Romano, E. Patterson, and E. Candes, `Conformalized Quantile
  Regression <https://arxiv.org/abs/1905.03222>`_, 2019.
- J. Diquigiovanni, M. Fontana, and S. Vantini, `Conformal Prediction Bands
  for Multivariate Functional Data <https://arxiv.org/abs/2106.01792>`_,
  2021. Functional bands.
- A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster,
  `Conformal Risk Control <https://arxiv.org/abs/2208.02814>`_, 2022.
- V. Gopakumar et al., `Uncertainty Quantification of
  Surrogate Models using Conformal Prediction
  <https://arxiv.org/abs/2408.09881>`_, 2024. Pointwise intervals on
  fields.


API Reference
-------------

Calibrators
^^^^^^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.PointwiseCalibrator
    :members:
    :inherited-members:

.. autoclass:: physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator
    :members:
    :inherited-members:

.. autoclass:: physicsnemo.experimental.uq.conformal.RiskControlCalibrator
    :members:
    :inherited-members:

Scores
^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.AbsoluteErrorScore
    :members: score, interval

.. autoclass:: physicsnemo.experimental.uq.conformal.NormalizedErrorScore
    :members: score, interval

.. autoclass:: physicsnemo.experimental.uq.conformal.QuantileRegressionScore
    :members: score, interval

.. autoclass:: physicsnemo.experimental.uq.conformal.AuxScale
    :members: __call__

Custom Score API
^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.NonconformityScore
    :members: score, bounds, interval

.. autofunction:: physicsnemo.experimental.uq.conformal.register_score

.. autofunction:: physicsnemo.experimental.uq.conformal.check_score

Predictor
^^^^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.ConformalPredictor
    :members:

Coverage Diagnostics
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.CoverageAccumulator
    :members:

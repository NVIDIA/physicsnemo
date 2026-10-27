PhysicsNeMo Conformal Prediction
================================

.. currentmodule:: physicsnemo.experimental.uq.conformal

Calibrated prediction intervals for the field outputs of a trained model,
without retraining it.

.. warning::

    This module is experimental. Its API may change without a deprecation
    period.

Overview
--------

A surrogate model returns a field, for example surface pressure on a car,
but not how far that field can be from the truth. Conformal prediction
answers that question after training. It measures the model's errors on a
held-out *calibration set*, picks a threshold from those errors, and turns
every new prediction into a lower and an upper bound, ``(lo, hi)``, with a
stated probability of containing the true field.

The guarantee holds for any model and any data distribution. It needs only
one assumption: the calibration samples and the new samples are
*exchangeable*, for example drawn independently from the same distribution.
If the deployment data drifts away from the calibration data, the
guarantee no longer holds.

The workflow has three steps:

1. Pass each calibration sample, the prediction and the target, to a
   calibrator's ``update``.
2. Call ``finalize`` to fit the thresholds and get a
   :class:`ConformalPredictor`.
3. Call ``predictor.predict_interval`` on each new model output to get
   ``(lo, hi)``.

The predictor can be saved, loaded elsewhere, and checked on held-out data
with a :class:`CoverageAccumulator`.


Quick Start
-----------

.. code:: python

    from physicsnemo.experimental.uq.conformal import (
        AbsoluteErrorScore,
        RiskControlCalibrator,
    )

    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.1)
    for prediction, target in calibration_set:  # meshes may differ per sample
        calibrator.update(prediction, target)
    predictor = calibrator.finalize()

    lo, hi = predictor.predict_interval(model(new_input))

With ``alpha=0.1``, a new field has on average at most 10% of its points
outside ``[lo, hi]``. ``prediction`` and ``target`` have shape
:math:`(n_{\text{points}}, *\text{dims})`, for example
``(n_points, 4)`` for pressure and wall shear stress.


Choosing a Calibrator
---------------------

The calibrator sets what the guarantee covers and how the band width varies
across the field. Pick it by the guarantee your application needs and by
whether every sample uses the same mesh.

.. list-table::
   :header-rows: 1
   :widths: 18 30 34 18

   * - Calibrator
     - With ``alpha=0.1``
     - Band width across the field
     - Meshes
   * - :class:`PointwiseCalibrator`
     - Each output element is covered with probability at least 90%, one
       element at a time.
     - One width per element, fitted from that element's calibration
       errors, so the band follows the model's local error.
     - One layout; the same mesh if ``points`` is given.
   * - :class:`FunctionalBandCalibrator`
     - The whole field is covered at once with probability at least 90%.
     - One threshold times the per-point scale from the score (sigma for
       ``NormalizedErrorScore``) or from an ``AuxScale``; constant
       across the field with ``AbsoluteErrorScore`` alone.
     - May differ.
   * - :class:`RiskControlCalibrator`
     - The fraction of missed points is at most 10% in expectation; a
       single case may exceed it.
     - Same as the functional band, and narrower.
     - May differ.

For fields on meshes that change between samples, such as CFD results on
different geometries, start with :class:`RiskControlCalibrator`. Use
:class:`FunctionalBandCalibrator` when a single miss anywhere on the field
matters, for example to bound the worst-case error over a surface. Both
treat a field as a whole: the functional band must hold at every point and
every channel, and risk control counts a point as missed when any of its
channels falls outside the band. Separate ``TensorDict`` fields get separate
thresholds and separate guarantees. Use :class:`PointwiseCalibrator` when
every sample has the same output layout and you need an interval for each
output element. The layout can be one grid or mesh, or a fixed set of
outputs such as drag, lift, and moment coefficients. For a mesh, pass the
coordinates as ``points=`` on every call, and the calibrator and predictor
reject any other mesh.

Non-Uniform Meshes
^^^^^^^^^^^^^^^^^^

Risk control counts every point equally, so on a mesh that is fine in one
region and coarse in another, the fraction of missed points is not the
fraction of missed area. Pass ``weights=`` to count each point by its cell
area or volume:

.. code:: python

    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.1)
    for prediction, target, areas in calibration_set:
        calibrator.update(prediction, target, weights=areas)
    predictor = calibrator.finalize()

``weights`` has shape ``(n_points,)`` and holds nonnegative, finite values
that are not all zero; the calibrator normalizes them within each sample.
The guarantee becomes: the expected weighted fraction of missed points is at
most ``alpha``. With ``physicsnemo.mesh``, ``point_measures(mesh)`` from
``physicsnemo.mesh.calculus`` returns these weights for a mesh that stores
point measures, and ``lumped_point_measures(mesh)`` builds them from the
cells. Pass ``weights`` on every call or on none. The predictor remembers
that it was calibrated with weights, also across ``save`` and ``load``, and
the accumulator from ``predictor.coverage_accumulator()`` then requires
``weights`` on every ``update(lo, hi, target, weights=...)`` so the report
measures the weighted risk.

Several Coverage Levels
^^^^^^^^^^^^^^^^^^^^^^^

``finalize(alpha=...)`` fits the thresholds at another level from the stored
calibration scores, so one pass over the calibration set gives predictors at
several levels:

.. code:: python

    predictor_80 = calibrator.finalize(alpha=0.2)
    predictor_90 = calibrator.finalize()  # the constructor's alpha
    predictor_95 = calibrator.finalize(alpha=0.05)

Each predictor has its own ``alpha`` and its own guarantee. Choose the
levels before looking at the fitted intervals: picking ``alpha`` from the
results weakens the guarantee. The calibration set size rule in Practical
Notes applies to each level, so ``alpha=0.05`` needs at least 19 samples.


Choosing a Score
----------------

The score sets the shape of the interval. Pick it by what the model
outputs. The score reads any extra model outputs from the ``aux``
argument.

.. list-table::
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
where it is not, so it is the better choice when the model has a
:math:`\sigma` head, an MC-dropout spread, or an ensemble spread.

Pass ``sigma`` the same way at calibration and at prediction:

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


Custom Scores
-------------

Write a custom score when no built-in score gives the interval shape you
need, for example a band proportional to the prediction. You do not need
one to widen the band where the model is uncertain:
:class:`NormalizedErrorScore` divides the error by a predicted ``sigma``,
and ``scale=AuxScale(key)`` multiplies a functional or risk-control
band by any per-point ``aux`` entry without a new score. A custom score
can shape the interval element by element, and works with every
calibrator.

The Contract
^^^^^^^^^^^^

Subclass :class:`NonconformityScore`, implement two methods, and register
the class with :func:`register_score`:

- ``score(prediction, target, *, aux=None)`` returns the per-element score;
  larger is worse.
- ``bounds(prediction, threshold, *, aux=None)`` returns ``(lo, hi)``, the
  targets whose score is at most ``threshold``.

The coverage guarantee holds only if the two methods agree: for every
element, the targets with ``score <= t`` are exactly the values in
``bounds(t)``, and that interval grows with ``t``. For example, a band
proportional to the prediction:

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

.. list-table::
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
     - Scores, and so the fitted threshold, may be negative. Otherwise
       predictors reject a negative threshold.
     - The score is signed, as in :class:`QuantileRegressionScore`, where a
       negative threshold shrinks the band.
   * - ``scale_aux_keys``
     - The ``aux`` entries the score divides by. Calibrators reject an
       :class:`AuxScale` that reads one of them, since the band would
       scale by it twice.
     - ``bounds`` multiplies the threshold by an ``aux`` entry.

A score that reads an ``aux`` entry lists it in ``aux_keys``, and also in
``scale_aux_keys`` if it divides by it. For example, an error in units of a
per-point tolerance set by the application:

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
``bounds``, and rounds the interval outward to the prediction's dtype, so
low-precision inputs keep the guarantee without any work in ``bounds``. The
``aux`` entries reach ``bounds`` in their original dtype: repeat any
dtype-dependent step of ``score``, such as a clamp, before converting them
to float64, so the two methods still agree.

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
that registers the class before :meth:`ConformalPredictor.load`; otherwise
``load`` raises and lists the registered names. Registering the same class
again under the same name is allowed, for example when a notebook cell is
re-run; registering a different class under a name in use raises
``ValueError``.


Multiple Fields
---------------

Pass a ``TensorDict`` to calibrate several fields at once. Each field gets
its own threshold and its own guarantee. Nest ``aux`` by field name:

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
that was not used for calibration. The coverage report measures the same
quantity the calibrator guarantees. This example continues the Quick Start
predictor, which uses :class:`AbsoluteErrorScore`; a predictor fitted with
:class:`NormalizedErrorScore` also needs ``aux={"sigma": ...}`` on every
``predict_interval`` call:

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
should be close to or below ``alpha``. A plain-tensor field is reported
under the key ``"tensor"``; ``TensorDict`` fields under their own names.


Practical Notes
---------------

**Calibration set size.**
    ``finalize`` needs at least :math:`(1 - \alpha) / \alpha` samples at the
    level it fits, for example 9 for ``alpha=0.1`` and 19 for
    ``alpha=0.05``. With more samples the fitted threshold varies less from
    one calibration set to the next.

**Single process.**
    Calibration does not communicate across ranks. Predictions may come
    from many GPUs, but gather every calibration sample onto the process
    that calls ``finalize``, so there is one threshold. A rank that
    calibrates on its own share still gets a valid guarantee when the
    share was chosen independently of the data, for example by shard index,
    but from fewer samples and with a different threshold on every rank. A
    share chosen by looking at the data, such as the hardest cases, has no
    guarantee.

**Memory.**
    :class:`PointwiseCalibrator` and :class:`RiskControlCalibrator` keep the
    calibration scores on the CPU until ``finalize``. Pointwise memory grows
    as :math:`n_{cal}` times the field size; risk control stores one
    float64 per calibration point, two with ``weights``.
    :class:`FunctionalBandCalibrator` stores one number per sample.

**Devices and dtypes.**
    Inputs may live on any device and use any floating-point dtype,
    including ``float16`` and ``bfloat16``. Interval endpoints are computed
    in float64 and rounded outward to the input dtype, so low-precision
    inputs do not break the guarantee.

**Separate data for the uncertainty estimate.**
    Whatever produces ``sigma`` or the ``AuxScale`` values must not see
    the calibration samples: fit it on training data or a separate split.


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

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Calibrators that turn held-out model errors into conformal intervals.

Choose a calibrator by the guarantee you need:

- :class:`~physicsnemo.experimental.uq.conformal.PointwiseCalibrator`: an
  interval for each output element, when every sample has the same output
  layout.
- :class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator`:
  a functional band that contains every point of a field at once; meshes
  may vary.
- :class:`~physicsnemo.experimental.uq.conformal.RiskControlCalibrator`: a
  band that bounds the expected fraction of points it misses; meshes may
  vary.

Pass each calibration sample to ``update``, then call ``finalize``
to get a :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`.
Samples are plain tensors or ``TensorDict`` field containers. The first
sample fixes which of the two you use and which fields it has; later
samples must match. A sample that fails a check raises and leaves the
calibrator unchanged, so you can skip it and continue.

Calibration does not communicate across ranks. Predictions may come from
many GPUs, but gather every calibration sample onto the process that calls
``finalize``, so there is one threshold. A rank that calibrates on its own
share keeps a valid guarantee only if that share was chosen independently
of the data, and then from fewer samples and with a different threshold on
every rank.

The split conformal construction follows `Distribution-Free Predictive
Inference for Regression <https://arxiv.org/abs/1604.04173>`_ (Lei et al.,
2018): with :math:`n_{cal}` calibration scores the fitted threshold is the
:math:`k`-th smallest score, :math:`k = \lceil (n_{cal} + 1)(1 - \alpha)
\rceil`. The conformal risk control (CRC) tier instead follows `Conformal
Risk Control <https://arxiv.org/abs/2208.02814>`_ (Angelopoulos et al.,
2022).
"""

import copy
import struct
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from ._utils import (
    _field_label,
    broadcast_scale,
    check_aux,
    check_exact_shape,
    check_finite,
    check_point_alignment,
    check_points,
    check_real,
    check_weights,
    conformal_quantile_index,
    field_items,
    kth_smallest_of_samples,
    normalize_keys,
    pack_fields,
    require_feasible_alpha,
    require_matching_keys,
    require_mesh,
    slice_aux,
    validate_alpha,
)
from .predictors import ConformalPredictor
from .scores import (
    _SCALE_REGISTRY,
    _SCORE_REGISTRY,
    AuxScale,
    NonconformityScore,
    _check_no_double_scale,
    _inversion_message,
    _snapshot_strategy,
    _uncovered_count,
)

__all__ = [
    "PointwiseCalibrator",
    "FunctionalBandCalibrator",
    "RiskControlCalibrator",
]

# One field's staged record: (key, prediction, target, aux) -> stored value.
_Stage = Callable[[str, Tensor, Tensor, Mapping[str, Tensor] | None], object]


def _normalized_scores(raw: Tensor, scale: Tensor | None, key: str) -> Tensor:
    """Divide scores by the scale in float64; reject non-finite or tiny results."""
    raw64 = raw.to(torch.float64)
    if scale is None:
        return raw64
    scale64 = broadcast_scale(scale, raw, key).to(torch.float64)
    normalized = raw64 / scale64
    check_finite(key, "the scores divided by the per-point scale", normalized)
    underflow = (raw64 != 0) & (normalized.abs() < torch.finfo(torch.float64).tiny)
    if bool(underflow.any()):
        raise ValueError(
            f"{_field_label(key)}: {int(underflow.sum())} score(s) divided by "
            "the per-point scale are too small for float64. Rescale the "
            "data or the aux entry the AuxScale reads."
        )
    return normalized


class _SplitCalibratorBase:
    """Shared state, input checks, and threshold fitting for all calibrators."""

    def __init__(
        self,
        score: NonconformityScore,
        alpha: float,
        *,
        keys: Sequence[str] | None = None,
    ) -> None:
        self._score = _snapshot_strategy(score, _SCORE_REGISTRY, "score")
        self._alpha = validate_alpha(alpha)
        self._keys = normalize_keys(keys)
        self._n = 0
        self._tensor_mode: bool | None = None
        self._schema: tuple[str, ...] | None = None
        self._scores: dict[str, list[Tensor]] = {}

    @property
    def score(self) -> NonconformityScore:
        r"""Copy of the score set at construction; editing it has no effect."""
        return copy.deepcopy(self._score)

    @property
    def alpha(self) -> float:
        r"""Target miscoverage level (risk level for risk control)."""
        return self._alpha

    @property
    def keys(self) -> list[str] | None:
        r"""Fields being calibrated, or ``None`` for every field."""
        return list(self._keys) if self._keys is not None else None

    @property
    def n_cal(self) -> int:
        r"""Number of calibration samples accepted so far."""
        return self._n

    def _validated_fields(
        self,
        prediction: Tensor | TensorDict,
        target: Tensor | TensorDict,
        aux: Mapping | None,
    ) -> tuple[bool, tuple[str, ...], list[tuple[str, Tensor, Tensor, Mapping | None]]]:
        """Check one sample against the fields seen so far, without changing state."""
        prediction_items = field_items(prediction, self._keys)
        target_items = dict(field_items(target, self._keys))
        _tensor_mode = isinstance(prediction, Tensor)
        if _tensor_mode != isinstance(target, Tensor):
            raise TypeError(
                "prediction and target must both be plain tensors or both be "
                "TensorDict field containers."
            )
        if self._tensor_mode is not None and self._tensor_mode != _tensor_mode:
            raise TypeError(
                "Cannot mix plain-tensor and TensorDict updates in one calibrator."
            )
        prediction_keys = tuple(key for key, _ in prediction_items)
        require_matching_keys(
            prediction_keys, target_items, "prediction/target field mismatch"
        )
        if self._schema is not None and prediction_keys != self._schema:
            raise KeyError(
                f"Fields changed: the first sample had {list(self._schema)}, "
                f"this one has {list(prediction_keys)}."
            )

        fields = []
        for key, prediction_field in prediction_items:
            target_field = target_items[key]
            if prediction_field.numel() == 0:
                raise ValueError(
                    f"{_field_label(key)}: empty sample; each sample must hold at "
                    "least one value."
                )
            check_exact_shape(
                key, "prediction", prediction_field, "target", target_field
            )
            check_real(key, "prediction", prediction_field)
            check_real(key, "target", target_field)
            aux_field = slice_aux(aux, key)
            check_aux(key, self._score, prediction_field, aux_field)
            fields.append((key, prediction_field, target_field, aux_field))
        return _tensor_mode, prediction_keys, fields

    def _collect(
        self,
        prediction: Tensor | TensorDict,
        target: Tensor | TensorDict,
        aux: Mapping | None,
        points: Tensor | None,
        stage: _Stage,
    ) -> None:
        """Check every field first so that a rejected sample is not stored."""
        with torch.no_grad():
            _tensor_mode, schema, fields = self._validated_fields(
                prediction, target, aux
            )
            staged: dict[str, object] = {}
            for key, prediction_field, target_field, aux_field in fields:
                if points is not None:
                    check_point_alignment(key, prediction_field, points, "prediction")
                if self._n == 0:
                    self._check_score_inverts(
                        key, prediction_field, target_field, aux_field
                    )
                staged[key] = stage(key, prediction_field, target_field, aux_field)
            self._tensor_mode, self._schema = _tensor_mode, schema
            for key, record in staged.items():
                self._scores.setdefault(key, []).append(record)
            self._n += 1

    def _check_score_inverts(
        self,
        key: str,
        prediction: Tensor,
        target: Tensor,
        aux: Mapping[str, Tensor] | None,
    ) -> None:
        """Reject a score whose ``bounds()`` does not invert ``score()`` on this sample."""
        uncovered = int(_uncovered_count(self._score, prediction, target, aux))
        if uncovered:
            raise ValueError(
                f"{_field_label(key)}: "
                + _inversion_message(self._score, uncovered, target.numel())
            )

    def _require_finalizable(self) -> None:
        if self._n == 0:
            raise RuntimeError("No calibration samples collected.")

    def _resolve_alpha(self, alpha: float | None) -> float:
        return self._alpha if alpha is None else validate_alpha(alpha)

    def _conformal_thresholds(self, alpha: float) -> Tensor | TensorDict:
        k = conformal_quantile_index(self._n, alpha)
        return pack_fields(
            {
                key: kth_smallest_of_samples(per_sample, k)
                for key, per_sample in self._scores.items()
            }
        )

    def _build_predictor(
        self, tier: str, thresholds: Tensor | TensorDict, *, alpha: float, **state
    ) -> ConformalPredictor:
        return ConformalPredictor._from_state(
            tier=tier,
            score=self._score,
            alpha=alpha,
            n_cal=self._n,
            thresholds=thresholds,
            **state,
        )


class PointwiseCalibrator(_SplitCalibratorBase):
    r"""Calibrate a separate interval width for every output element of a fixed layout.

    Use this calibrator when every calibration and deployment sample has the
    same output layout and you want an interval for each output element, as
    in the field-level conformal prediction of `Gopakumar et al., 2024
    <https://arxiv.org/abs/2408.09881>`_. The layout can be a mesh or grid
    with the points in the same order, or a fixed set of outputs such as
    scalar observables. The calibrator requires the same output shape on
    every call. If the mesh changes between samples, use
    :class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator`
    or :class:`~physicsnemo.experimental.uq.conformal.RiskControlCalibrator`.

    For outputs on a mesh, pass the coordinates as ``points=`` on every call
    to also check that every sample, and later every prediction, uses the
    same mesh. Two meshes count as the same only when their coordinate
    values, order, shape, and dtype all match; the same point count is not
    enough.

    Guarantee: each output element is covered with probability at least
    :math:`1 - \alpha`, one element at a time, not the whole field at once.

    Parameters
    ----------
    score : NonconformityScore
        Defines the interval: ``AbsoluteErrorScore`` for a fixed-width band
        around the prediction, ``NormalizedErrorScore`` for a band scaled by
        ``aux["sigma"]``, ``QuantileRegressionScore`` to adjust the model's
        own ``aux["lo"]`` and ``aux["hi"]`` bounds, or a custom score
        registered with
        :func:`~physicsnemo.experimental.uq.conformal.register_score`. The calibrator keeps a
        copy, so later edits to ``score`` have no effect.
    alpha : float
        Target miscoverage level in :math:`(0, 1)`; ``alpha=0.1`` means
        90% coverage.
    keys : Sequence[str], optional
        Names of the ``TensorDict`` fields to calibrate. The fitted
        :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
        remembers them, so its ``predict_interval`` accepts a container
        with extra fields and uses only these. By default every field is
        calibrated.

    Notes
    -----
    Each element's threshold is the :math:`k`-th smallest of its
    :math:`n_{cal}` calibration scores,
    :math:`k = \lceil (n_{cal} + 1)(1 - \alpha) \rceil`. The calibrator
    keeps every sample's scores on the CPU until :meth:`finalize`, so
    memory grows as :math:`n_{cal}` times the field size.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import (
    ...     AbsoluteErrorScore, PointwiseCalibrator,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> points = torch.rand(50, 3)
    >>> calibrator = PointwiseCalibrator(AbsoluteErrorScore(), alpha=0.1)
    >>> for _ in range(20):
    ...     prediction = torch.randn(50, 2)
    ...     target = prediction + 0.1 * torch.randn(50, 2)
    ...     calibrator.update(prediction, target, points=points)
    >>> predictor = calibrator.finalize()
    >>> lo, hi = predictor.predict_interval(torch.randn(50, 2), points=points)
    >>> lo.shape
    torch.Size([50, 2])
    """

    def __init__(
        self,
        score: NonconformityScore,
        alpha: float,
        *,
        keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(score, alpha, keys=keys)
        self._mesh_fingerprint: str | None = None

    def update(
        self,
        prediction: Float[Tensor, "*dims"] | TensorDict,
        target: Float[Tensor, "*dims"] | TensorDict,
        *,
        aux: Mapping[str, Float[Tensor, "*dims"]] | Mapping[str, Mapping] | None = None,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
    ) -> None:
        r"""Add one calibration sample.

        Parameters
        ----------
        prediction : torch.Tensor | TensorDict
            Model output of shape :math:`(n_{\text{points}}, *\text{dims})`,
            or a field container of such tensors.
        target : torch.Tensor | TensorDict
            Observed values, same shape and container type as ``prediction``.
        aux : Mapping[str, torch.Tensor] | Mapping[str, Mapping], optional
            Extra tensors the score reads, each the same shape as
            ``prediction``: ``{"sigma": ...}`` for ``NormalizedErrorScore``,
            ``{"lo": ..., "hi": ...}`` for ``QuantileRegressionScore``. For
            ``TensorDict`` inputs, nest by field name, as in
            ``{"pressure": {"sigma": ...}}``.
        points : torch.Tensor, optional
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`, identical
            on every call. Pass them on every call or on none.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``points`` differs from the first sample's mesh, if the
            output shape differs from the first sample's, if ``prediction``
            and ``target`` shapes differ, or if any value is non-finite.
        TypeError
            If ``points`` is passed on some calls but not others, or if
            plain tensors and ``TensorDict`` inputs are mixed.
        KeyError
            If the set of fields differs from the first sample.

        Notes
        -----
        A sample that raises is not stored.
        """
        if self._n and (points is None) != (self._mesh_fingerprint is None):
            first = "had" if self._mesh_fingerprint is not None else "did not have"
            raise TypeError(
                f"The first calibration sample {first} points=; pass points= "
                "on every call or on none."
            )
        fingerprint = (
            None if points is None else require_mesh(points, self._mesh_fingerprint)
        )

        def stage(key, prediction_field, target_field, aux_field):
            score = check_finite(
                key,
                "the nonconformity scores",
                self._score.score(prediction_field, target_field, aux=aux_field),
            )
            if key in self._scores and score.shape != self._scores[key][0].shape:
                raise ValueError(
                    f"{_field_label(key)}: score shape {tuple(score.shape)} differs "
                    f"from the first sample's {tuple(self._scores[key][0].shape)}; "
                    "PointwiseCalibrator needs the same shape on every call."
                )
            return score.detach().cpu().contiguous()

        self._collect(prediction, target, aux, points, stage)
        self._mesh_fingerprint = fingerprint

    def finalize(self, *, alpha: float | None = None) -> ConformalPredictor:
        r"""Fit the per-element thresholds and return the predictor.

        Parameters
        ----------
        alpha : float, optional
            Miscoverage level for this predictor, in :math:`(0, 1)`. Defaults
            to the constructor's ``alpha``. The stored calibration scores are
            reused, so one calibrator can produce predictors at several
            levels from one pass over the calibration set. Choose the levels
            before looking at the fitted intervals: picking ``alpha`` from
            the results weakens the guarantee.

        Returns
        -------
        ConformalPredictor
            A pointwise
            :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
            with one threshold per output element. If calibration used
            ``points``, its ``predict_interval`` requires the same ``points``.

        Raises
        ------
        RuntimeError
            If no samples were collected.
        TypeError
            If ``alpha`` is not a real number.
        ValueError
            If ``alpha`` is outside :math:`(0, 1)`, or if
            :math:`n_{cal} < (1 - \alpha) / \alpha`, for example fewer
            than 9 samples for ``alpha=0.1``. Collect more or raise ``alpha``.
        """
        alpha = self._resolve_alpha(alpha)
        self._require_finalizable()
        return self._build_predictor(
            "pointwise",
            self._conformal_thresholds(alpha),
            alpha=alpha,
            mesh_fingerprint=self._mesh_fingerprint,
        )


class _ScaledCalibratorBase(_SplitCalibratorBase):
    """Shared ``update`` and scaled scores for calibrators with an ``AuxScale``."""

    _reduce: Callable[[Tensor], Tensor]

    def __init__(
        self,
        score: NonconformityScore,
        alpha: float,
        *,
        scale: AuxScale | None = None,
        keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(score, alpha, keys=keys)
        self._scale = (
            None
            if scale is None
            else _snapshot_strategy(scale, _SCALE_REGISTRY, "scale")
        )
        _check_no_double_scale(self._score, self._scale)

    @property
    def scale(self) -> AuxScale | None:
        r"""Copy of the ``AuxScale`` set at construction, or ``None``."""
        return copy.deepcopy(self._scale)

    def update(
        self,
        prediction: Float[Tensor, "*dims"] | TensorDict,
        target: Float[Tensor, "*dims"] | TensorDict,
        *,
        aux: Mapping[str, Float[Tensor, "*dims"]] | Mapping[str, Mapping] | None = None,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
    ) -> None:
        r"""Add one calibration sample; its point count may differ from other samples.

        Parameters
        ----------
        prediction : torch.Tensor | TensorDict
            Model output of shape :math:`(n_{\text{points}}, *\text{dims})`
            (any shape :math:`(*\text{dims})` when neither ``points`` nor
            ``scale`` is used), or a field container of such tensors.
        target : torch.Tensor | TensorDict
            Observed values, same shape and container type as ``prediction``.
        aux : Mapping[str, torch.Tensor] | Mapping[str, Mapping], optional
            Extra tensors read by the score or ``AuxScale``, each the
            same shape as ``prediction``, such as ``{"sigma": ...}`` for
            ``AuxScale("sigma")``. For ``TensorDict`` inputs, nest by
            field name, as in ``{"pressure": {"sigma": ...}}``.
        points : torch.Tensor, optional
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`; may differ
            between samples. Only checks that each field has one leading
            entry per point; it does not change the band.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``prediction`` and ``target`` shapes differ, if any value or
            scale is non-finite, or if a score divided by its scale is too
            small for float64.
        TypeError
            If plain tensors and ``TensorDict`` inputs are mixed.
        KeyError
            If the set of fields differs from the first sample.

        Notes
        -----
        A sample that raises is not stored.
        """

        def stage(key, prediction_field, target_field, aux_field):
            return self._reduce(
                self._scaled_scores(key, prediction_field, target_field, aux_field)
            )

        if points is not None:
            check_points(points)
        self._collect(prediction, target, aux, points, stage)

    def _scaled_scores(
        self,
        key: str,
        prediction: Tensor,
        target: Tensor,
        aux: Mapping[str, Tensor] | None,
    ) -> Tensor:
        """One field's finite scores divided by the per-point scale, in float64."""
        scale = None if self._scale is None else self._scale(aux)
        raw = check_finite(
            key,
            "the nonconformity scores",
            self._score.score(prediction, target, aux=aux),
        )
        return _normalized_scores(raw, scale, key)


class FunctionalBandCalibrator(_ScaledCalibratorBase):
    r"""Calibrate a band that contains every point of each new field at once.

    Use this calibrator when a single band must cover every point of a field
    at the same time, for example to bound the worst-case error over a mesh.
    Meshes may differ between calibration and deployment. At point
    :math:`x` the score turns :math:`\text{threshold} \cdot s(x)` into the
    band, where :math:`s` is the per-point scale from the optional
    ``AuxScale``. Without one the width is whatever the score gives: constant for
    ``AbsoluteErrorScore``, proportional to ``aux["sigma"]`` for
    ``NormalizedErrorScore``, and following the quantile heads for
    ``QuantileRegressionScore``. If bounding the expected fraction of missed
    points is enough,
    :class:`~physicsnemo.experimental.uq.conformal.RiskControlCalibrator`
    gives tighter bands.

    Guarantee: a new field lies inside the band at every point with
    probability at least :math:`1 - \alpha`. For a field of shape
    :math:`(n_{\text{points}}, C)` the band covers all points and all
    :math:`C` channels at the same time, because each sample's score is its
    maximum over every element. Each ``TensorDict`` field gets its own
    threshold and its own guarantee; there is no joint guarantee across
    fields.

    Parameters
    ----------
    score : NonconformityScore
        Defines the interval (see
        :class:`~physicsnemo.experimental.uq.conformal.PointwiseCalibrator`).
        The calibrator keeps a copy.
    alpha : float
        Target miscoverage level in :math:`(0, 1)`.
    scale : AuxScale, optional
        Per-point scale :math:`s(x)` that widens the band where it is large.
        With ``None`` the score sets the width, for example constant for
        ``AbsoluteErrorScore`` and proportional to ``aux["sigma"]`` for
        ``NormalizedErrorScore``.
    keys : Sequence[str], optional
        Names of the ``TensorDict`` fields to calibrate (see
        :class:`~physicsnemo.experimental.uq.conformal.PointwiseCalibrator`).
        By default every field is calibrated.

    Raises
    ------
    ValueError
        If ``score`` already divides by the aux key ``scale`` reads.

    Notes
    -----
    Each calibration sample contributes its largest scaled score,
    :math:`\max_x \text{score}(x) / s(x)`, as in `Conformal prediction bands
    for multivariate functional data <https://arxiv.org/abs/2106.01792>`_
    (Diquigiovanni, Fontana and Vantini, 2021). The threshold is the
    :math:`k`-th smallest of the :math:`n_{cal}` per-sample maxima,
    :math:`k = \lceil (n_{cal} + 1)(1 - \alpha) \rceil`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import (
    ...     AbsoluteErrorScore, FunctionalBandCalibrator,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> calibrator = FunctionalBandCalibrator(AbsoluteErrorScore(), alpha=0.1)
    >>> for n_points in range(40, 60):
    ...     prediction = torch.randn(n_points, 2)
    ...     target = prediction + 0.1 * torch.randn(n_points, 2)
    ...     calibrator.update(prediction, target)
    >>> predictor = calibrator.finalize()
    >>> lo, hi = predictor.predict_interval(torch.randn(80, 2))
    >>> hi.shape
    torch.Size([80, 2])
    """

    _reduce = staticmethod(lambda normalized: normalized.amax().detach().cpu())

    def finalize(self, *, alpha: float | None = None) -> ConformalPredictor:
        r"""Fit the band threshold and return the predictor.

        Parameters
        ----------
        alpha : float, optional
            Miscoverage level for this predictor, in :math:`(0, 1)`. Defaults
            to the constructor's ``alpha``. The stored per-sample maxima are
            reused, so one calibrator can produce predictors at several
            levels from one pass over the calibration set. Choose the levels
            before looking at the fitted bands: picking ``alpha`` from the
            results weakens the guarantee.

        Returns
        -------
        ConformalPredictor
            A functional
            :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
            with one float64 scalar threshold per field. It applies the same
            ``AuxScale``, so pass the same ``aux`` keys at prediction.

        Raises
        ------
        RuntimeError
            If no samples were collected.
        TypeError
            If ``alpha`` is not a real number.
        ValueError
            If ``alpha`` is outside :math:`(0, 1)`, or if
            :math:`n_{cal} < (1 - \alpha) / \alpha`, for example fewer
            than 9 samples for ``alpha=0.1``. Collect more or raise ``alpha``.
        """
        alpha = self._resolve_alpha(alpha)
        self._require_finalizable()
        return self._build_predictor(
            "functional",
            self._conformal_thresholds(alpha),
            alpha=alpha,
            scale=self._scale,
        )


def _ordered_bits(x: float) -> int:
    """Integer that sorts like the float64 ``x``; ``-0.0`` sits just below ``0.0``."""
    bits = struct.unpack("<q", struct.pack("<d", x))[0]
    return bits if bits >= 0 else -(bits & 0x7FFFFFFFFFFFFFFF) - 1


def _from_ordered_bits(i: int) -> float:
    """Inverse of ``_ordered_bits``."""
    bits = i if i >= 0 else (-(i + 1)) | -0x8000000000000000
    return struct.unpack("<d", struct.pack("<q", bits))[0]


def _crc_threshold(
    sorted_scores: list[Tensor | tuple[Tensor, Tensor]], alpha: float
) -> float:
    """Smallest observed ``lambda`` with ``(R(lambda) + 1) / (n + 1) <= alpha``.

    Each entry is one sample's sorted per-point scores, or a ``(scores, tail)``
    pair from :func:`_sorted_point_scores` for a weighted sample, where
    ``tail[j]`` is the normalized weight of the points of sorted rank ``j`` or
    above.

    ``R`` only drops at observed scores, so the smallest float64 that meets the
    bound is an observed score. The search bisects over float64 bit patterns
    rather than numeric midpoints or sorted pooled candidates:

    - Ordered bit patterns are a total order over every float64, signed zeros
      included, so bisecting them lands exactly on an observed score in at
      most 64 probes.
    - Numeric midpoints can stall when ``lo`` and ``hi`` are adjacent floats,
      and need thousands of halvings to cross a range such as 1e-300 to 1e300.
    - Sorting the pooled candidates would materialize and sort
      ``n_cal * n_points`` scores (56 million for 140 samples of 400,000
      points); bit bisection needs only each sample's already-sorted scores
      and one ``searchsorted`` per sample per probe.

    An unweighted sample's miss fraction is an exact ``Fraction`` of its point
    count. A weighted sample's tail mass comes from a float64 normalization and
    cumulative sum, whose rounding moves it by at most ``numel * 2**-52``, so
    that margin is added: rounding can only raise the estimated risk, never
    lower it, and the threshold stays valid.
    """
    n = len(sorted_scores)
    alpha_exact = require_feasible_alpha(n, alpha)
    samples = [
        (record.to(torch.float64), None)
        if isinstance(record, Tensor)
        else (record[0].to(torch.float64), record[1])
        for record in sorted_scores
    ]

    def mass_above(scores: Tensor, tail: Tensor | None, rank: int) -> Fraction:
        numel = scores.numel()
        if tail is None:
            return Fraction(numel - rank, numel)
        if rank == 0:
            return Fraction(1)
        if rank == numel:
            return Fraction(0)
        return Fraction(float(tail[rank])) + Fraction(numel, 2**52)

    def within_budget(candidate: float) -> bool:
        probe = torch.tensor(candidate, dtype=torch.float64)
        total_loss = Fraction()
        for scores, tail in samples:
            rank = int(torch.searchsorted(scores, probe, right=True))
            total_loss += mass_above(scores, tail, rank)
        return (total_loss + 1) / (n + 1) <= alpha_exact

    # Below the smallest score every point is missed; at the largest none is.
    lo = _ordered_bits(min(float(scores[0]) for scores, _ in samples)) - 1
    hi = _ordered_bits(max(float(scores[-1]) for scores, _ in samples))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if within_budget(_from_ordered_bits(mid)):
            hi = mid
        else:
            lo = mid
    return _from_ordered_bits(hi)


def _sorted_point_scores(
    normalized: Tensor, weights: Tensor | None = None
) -> Tensor | tuple[Tensor, Tensor]:
    """Max over trailing dims to get one score per point, sorted before the CPU copy.

    With ``weights``, also return the normalized weights' tail mass in score
    order: ``tail[j]`` is the weight of the points of sorted rank ``j`` or above.
    """
    if normalized.ndim > 1:
        normalized = normalized.amax(dim=tuple(range(1, normalized.ndim)))
    point_scores = normalized.reshape(-1).detach()
    if weights is None:
        return point_scores.sort().values.cpu()
    sorted_scores, perm = point_scores.sort()
    w = weights.to(device=point_scores.device, dtype=torch.float64)
    tail = (w / w.sum())[perm].flip(0).cumsum(0).flip(0)
    return sorted_scores.cpu(), tail.cpu()


class RiskControlCalibrator(_ScaledCalibratorBase):
    r"""Calibrate a band that bounds the expected fraction of points it misses.

    Use this calibrator when meshes vary between samples and it is enough to
    control the average fraction of points outside the band, rather than
    cover every point at once. It gives much tighter bands than
    :class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator`
    in exchange for this weaker guarantee. A point counts as missed when any
    of its trailing components is outside the band, for example any channel
    of an :math:`(n_{\text{points}}, C)` field. Every sample has equal
    weight, whatever its point count; within a sample, ``weights=`` on
    :meth:`update` weights the points, for example by cell area or volume on
    a non-uniform mesh. Separate ``TensorDict`` fields get separate
    thresholds and separate guarantees, not a joint one. The threshold comes
    from conformal risk control (CRC), `Conformal Risk Control
    <https://arxiv.org/abs/2208.02814>`_ (Angelopoulos, Bates, Fisch, Lei
    and Schuster, 2022).

    Guarantee: the expected fraction of missed points in a new sample is at
    most :math:`\alpha`; with ``weights``, the expected weighted fraction of
    missed points is at most :math:`\alpha`.

    Parameters
    ----------
    score : NonconformityScore
        Defines the interval (see
        :class:`~physicsnemo.experimental.uq.conformal.PointwiseCalibrator`).
        The calibrator keeps a copy.
    alpha : float
        Target expected fraction of missed points, in :math:`(0, 1)`.
    scale : AuxScale, optional
        Per-point scale :math:`s(x)` that widens the band where it is large.
        Without one the width is whatever the score gives: constant for
        ``AbsoluteErrorScore``, proportional to ``aux["sigma"]`` for
        ``NormalizedErrorScore``, and following the quantile heads for
        ``QuantileRegressionScore``.
    keys : Sequence[str], optional
        Names of the ``TensorDict`` fields to calibrate (see
        :class:`~physicsnemo.experimental.uq.conformal.PointwiseCalibrator`).
        By default every field is calibrated.

    Raises
    ------
    ValueError
        If ``score`` already divides by the aux key ``scale`` reads.

    Notes
    -----
    The threshold :math:`\lambda` is the smallest observed score with
    :math:`(\hat R(\lambda) + 1)/(n_{cal} + 1) \le \alpha`, where
    :math:`\hat R` sums the per-sample fractions of missed points.

    The calibrator keeps every sample's per-point scores on the CPU until
    :meth:`finalize`, so memory grows with the total number of calibration
    points; with ``weights`` it doubles, one float64 tail mass per point.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import (
    ...     AbsoluteErrorScore, AuxScale, RiskControlCalibrator,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> calibrator = RiskControlCalibrator(
    ...     AbsoluteErrorScore(), alpha=0.1, scale=AuxScale("sigma")
    ... )
    >>> for n_points in range(40, 60):
    ...     prediction = torch.randn(n_points, 2)
    ...     sigma = torch.rand(n_points, 2) + 0.5
    ...     target = prediction + sigma * torch.randn(n_points, 2)
    ...     cell_areas = torch.rand(n_points) + 0.5
    ...     calibrator.update(
    ...         prediction, target, aux={"sigma": sigma}, weights=cell_areas
    ...     )
    >>> predictor = calibrator.finalize()
    >>> sigma = torch.rand(30, 2) + 0.5
    >>> lo, hi = predictor.predict_interval(torch.randn(30, 2), aux={"sigma": sigma})
    >>> lo.shape
    torch.Size([30, 2])
    """

    def __init__(
        self,
        score: NonconformityScore,
        alpha: float,
        *,
        scale: AuxScale | None = None,
        keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(score, alpha, scale=scale, keys=keys)
        self._weighted = False

    def update(
        self,
        prediction: Float[Tensor, "*dims"] | TensorDict,
        target: Float[Tensor, "*dims"] | TensorDict,
        *,
        aux: Mapping[str, Float[Tensor, "*dims"]] | Mapping[str, Mapping] | None = None,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
        weights: Float[Tensor, "n_points"] | None = None,
    ) -> None:
        r"""Add one calibration sample; its point count may differ from other samples.

        Parameters
        ----------
        prediction : torch.Tensor | TensorDict
            Model output of shape :math:`(n_{\text{points}}, *\text{dims})`
            (any shape :math:`(*\text{dims})` when none of ``points``,
            ``scale``, and ``weights`` is used), or a field container of
            such tensors.
        target : torch.Tensor | TensorDict
            Observed values, same shape and container type as ``prediction``.
        aux : Mapping[str, torch.Tensor] | Mapping[str, Mapping], optional
            Extra tensors read by the score or ``AuxScale``, each the
            same shape as ``prediction``, such as ``{"sigma": ...}`` for
            ``AuxScale("sigma")``. For ``TensorDict`` inputs, nest by
            field name, as in ``{"pressure": {"sigma": ...}}``.
        points : torch.Tensor, optional
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`; may differ
            between samples. Only checks that each field has one leading
            entry per point; it does not change the band.
        weights : torch.Tensor, optional
            Per-point weights of shape :math:`(n_{\text{points}},)`: floating
            point, finite, nonnegative, not all zero, on any device. For
            example cell areas or volumes on a non-uniform mesh (with
            :mod:`physicsnemo.mesh`,
            :func:`~physicsnemo.mesh.calculus.point_measures`). The sample's
            loss becomes its weighted fraction of missed points, normalized by
            the sum of the weights, and the guarantee becomes "the expected
            weighted fraction of missed points is at most :math:`\alpha`".
            For ``TensorDict`` inputs one vector applies to every calibrated
            field, so each field's leading dimension must equal
            ``len(weights)``. Pass ``weights`` on every call or on none.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``prediction`` and ``target`` shapes differ, if any value,
            scale, or weight is non-finite, if a score divided by its scale
            is too small for float64, or if ``weights`` has the wrong shape,
            a negative entry, or no positive entry.
        TypeError
            If plain tensors and ``TensorDict`` inputs are mixed, if
            ``weights`` is passed on some calls but not others, or if it is
            not a floating-point tensor.
        KeyError
            If the set of fields differs from the first sample.

        Notes
        -----
        A sample that raises is not stored.
        """
        if self._n and (weights is not None) != self._weighted:
            first = "had" if self._weighted else "did not have"
            raise TypeError(
                f"The first calibration sample {first} weights=; pass weights= "
                "on every call or on none."
            )
        if points is not None:
            check_points(points)
        if weights is not None:
            check_weights(weights)

        def stage(key, prediction_field, target_field, aux_field):
            if weights is not None:
                check_point_alignment(
                    key, prediction_field, weights, "prediction", "weights"
                )
            return _sorted_point_scores(
                self._scaled_scores(key, prediction_field, target_field, aux_field),
                weights,
            )

        self._collect(prediction, target, aux, points, stage)
        self._weighted = weights is not None

    def finalize(self, *, alpha: float | None = None) -> ConformalPredictor:
        r"""Fit the CRC threshold and return the predictor.

        Parameters
        ----------
        alpha : float, optional
            Risk level for this predictor, in :math:`(0, 1)`. Defaults to
            the constructor's ``alpha``. The stored per-point scores are
            reused, so one calibrator can produce predictors at several
            levels from one pass over the calibration set. Choose the levels
            before looking at the fitted bands: picking ``alpha`` from the
            results weakens the guarantee.

        Returns
        -------
        ConformalPredictor
            A risk-control
            :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
            with one float64 scalar threshold per field. It applies the same
            ``AuxScale``, so pass the same ``aux`` keys at prediction.

        Raises
        ------
        RuntimeError
            If no samples were collected.
        TypeError
            If ``alpha`` is not a real number.
        ValueError
            If ``alpha`` is outside :math:`(0, 1)`, or if
            :math:`n_{cal} < (1 - \alpha) / \alpha`, for example fewer
            than 9 samples for ``alpha=0.1``. Collect more or raise ``alpha``.
        """
        alpha = self._resolve_alpha(alpha)
        self._require_finalizable()
        thresholds = {
            key: torch.tensor(_crc_threshold(samples, alpha), dtype=torch.float64)
            for key, samples in self._scores.items()
        }
        return self._build_predictor(
            "risk_control",
            pack_fields(thresholds),
            alpha=alpha,
            scale=self._scale,
            weighted=self._weighted,
        )

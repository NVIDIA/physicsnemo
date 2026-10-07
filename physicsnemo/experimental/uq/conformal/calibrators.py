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

- :class:`~physicsnemo.experimental.uq.conformal.CellwiseCalibrator`: an
  interval for each output element, when every sample uses the same mesh.
- :class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator`:
  a functional band that contains every point of a field at once; meshes
  may vary.

Pass each calibration sample to ``update``, then call ``finalize``
to get a :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`.
Samples are plain tensors or ``TensorDict`` field containers. The first
sample fixes which of the two you use and which fields it has; later
samples must match. A sample that fails a check raises and leaves the
calibrator unchanged, so you can skip it and continue.

Calibration does not communicate across ranks. Predictions may come from
many GPUs, but send every calibration sample to the process that calls
``finalize``: a threshold fit on part of the samples has no guarantee for
the full set.

The split conformal construction follows `Distribution-Free Predictive
Inference for Regression <https://arxiv.org/abs/1604.04173>`_ (Lei et al.,
2018): with :math:`n_{cal}` calibration scores the fitted threshold is the
:math:`k`-th smallest score, :math:`k = \lceil (n_{cal} + 1)(1 - \alpha)
\rceil`.
"""

import copy
from collections.abc import Callable, Mapping, Sequence

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from ._utils import (
    _field_label,
    broadcast_difficulty,
    check_aux,
    check_exact_shape,
    check_finite,
    check_point_alignment,
    check_points,
    check_real,
    conformal_quantile_index,
    field_items,
    kth_smallest_of_samples,
    normalize_keys,
    pack_fields,
    require_matching_keys,
    require_mesh,
    slice_aux,
    validate_alpha,
)
from .predictors import ConformalPredictor
from .scores import (
    _DIFFICULTY_REGISTRY,
    _SCORE_REGISTRY,
    AuxDifficulty,
    _check_no_double_scale,
    _NonconformityScore,
    _Score,
    _snapshot_strategy,
)

__all__ = [
    "CellwiseCalibrator",
    "FunctionalBandCalibrator",
]

# One field's staged record: (key, prediction, target, aux) -> stored value.
_Stage = Callable[[str, Tensor, Tensor, Mapping[str, Tensor] | None], object]


def _normalized_scores(raw: Tensor, difficulty: Tensor | None, key: str) -> Tensor:
    """Divide scores by the difficulty in float64; reject non-finite or tiny results."""
    raw64 = raw.to(torch.float64)
    if difficulty is None:
        return raw64
    difficulty64 = broadcast_difficulty(difficulty, raw, key).to(torch.float64)
    normalized = raw64 / difficulty64
    check_finite(key, "the scores divided by the AuxDifficulty scale", normalized)
    underflow = (raw64 != 0) & (normalized.abs() < torch.finfo(torch.float64).tiny)
    if bool(underflow.any()):
        raise ValueError(
            f"{_field_label(key)}: {int(underflow.sum())} score(s) divided by "
            "the AuxDifficulty scale are too small for float64. Rescale the "
            "data or the difficulty aux entry."
        )
    return normalized


class _SplitCalibratorBase:
    """Shared state, input checks, and threshold fitting for all calibrators."""

    def __init__(
        self,
        score: _Score,
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
    def score(self) -> _NonconformityScore:
        r"""Copy of the score set at construction; editing it has no effect."""
        return copy.deepcopy(self._score)

    @property
    def alpha(self) -> float:
        r"""Target miscoverage level."""
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
                staged[key] = stage(key, prediction_field, target_field, aux_field)
            self._tensor_mode, self._schema = _tensor_mode, schema
            for key, record in staged.items():
                self._scores.setdefault(key, []).append(record)
            self._n += 1

    def _require_finalizable(self) -> None:
        if self._n == 0:
            raise RuntimeError("No calibration samples collected.")

    def _conformal_thresholds(self) -> Tensor | TensorDict:
        k = conformal_quantile_index(self._n, self._alpha)
        return pack_fields(
            {
                key: kth_smallest_of_samples(per_sample, k)
                for key, per_sample in self._scores.items()
            }
        )

    def _build_predictor(
        self, tier: str, thresholds: Tensor | TensorDict, **state
    ) -> ConformalPredictor:
        return ConformalPredictor._from_state(
            tier=tier,
            score=self._score,
            alpha=self._alpha,
            n_cal=self._n,
            thresholds=thresholds,
            **state,
        )


class CellwiseCalibrator(_SplitCalibratorBase):
    r"""Calibrate a separate interval width for every output element on a fixed mesh.

    Use this calibrator when every calibration and deployment sample uses
    the same mesh or grid, with the points in the same order, and you want
    an interval for each output element, as in the field-level conformal
    prediction of `Gopakumar et al., 2024
    <https://arxiv.org/abs/2408.09881>`_. If the mesh changes between
    samples, use
    :class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator`.

    Pass the mesh coordinates as ``points=`` on every call. Two meshes count
    as the same only when their coordinate values, order, shape, and dtype
    all match; the same point count is not enough.

    Guarantee: each output element is covered with probability at least
    :math:`1 - \alpha`, one element at a time, not the whole field at once.

    Parameters
    ----------
    score : AbsoluteErrorScore | NormalizedErrorScore | QuantileRegressionScore
        Defines the interval: ``AbsoluteErrorScore`` for a fixed-width band
        around the prediction, ``NormalizedErrorScore`` for a band scaled by
        ``aux["sigma"]``, ``QuantileRegressionScore`` to adjust the model's
        own ``aux["lo"]`` and ``aux["hi"]`` bounds. The calibrator keeps a
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
    ...     AbsoluteErrorScore, CellwiseCalibrator,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> points = torch.rand(50, 3)
    >>> calibrator = CellwiseCalibrator(AbsoluteErrorScore(), alpha=0.1)
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
        score: _Score,
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
        points: Float[Tensor, "n_points n_spatial_dims"],
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
        points : torch.Tensor
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`, identical
            on every call.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``points`` differs from the first sample's mesh, if
            ``prediction`` and ``target`` shapes differ, or if any value is
            non-finite.
        TypeError
            If ``points`` is not passed, or if plain tensors and
            ``TensorDict`` inputs are mixed.
        KeyError
            If the set of fields differs from the first sample.

        Notes
        -----
        A sample that raises is not stored.
        """
        fingerprint = require_mesh(points, self._mesh_fingerprint)

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
                    "CellwiseCalibrator needs the same shape on every call."
                )
            return score.detach().cpu().contiguous()

        self._collect(prediction, target, aux, points, stage)
        self._mesh_fingerprint = fingerprint

    def finalize(self) -> ConformalPredictor:
        r"""Fit the per-element thresholds and return the predictor.

        Returns
        -------
        ConformalPredictor
            A cellwise
            :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
            with one threshold per output element. Its ``predict_interval``
            requires the same ``points`` used in calibration.

        Raises
        ------
        RuntimeError
            If no samples were collected.
        ValueError
            If :math:`n_{cal} < (1 - \alpha) / \alpha`, for example fewer
            than 9 samples for ``alpha=0.1``. Collect more or raise ``alpha``.
        """
        self._require_finalizable()
        return self._build_predictor(
            "cellwise",
            self._conformal_thresholds(),
            mesh_fingerprint=self._mesh_fingerprint,
        )


class _ScaledCalibratorBase(_SplitCalibratorBase):
    """Shared ``update`` for calibrators with an optional ``AuxDifficulty``."""

    _reduce: Callable[[Tensor], Tensor]

    def __init__(
        self,
        score: _Score,
        alpha: float,
        *,
        difficulty: AuxDifficulty | None = None,
        keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__(score, alpha, keys=keys)
        self._difficulty = (
            None
            if difficulty is None
            else _snapshot_strategy(difficulty, _DIFFICULTY_REGISTRY, "difficulty")
        )
        _check_no_double_scale(self._score, self._difficulty)

    @property
    def difficulty(self) -> AuxDifficulty | None:
        r"""Copy of the ``AuxDifficulty`` set at construction, or ``None``."""
        return copy.deepcopy(self._difficulty)

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
            ``difficulty`` is used), or a field container of such tensors.
        target : torch.Tensor | TensorDict
            Observed values, same shape and container type as ``prediction``.
        aux : Mapping[str, torch.Tensor] | Mapping[str, Mapping], optional
            Extra tensors read by the score or ``AuxDifficulty``, each the
            same shape as ``prediction``, such as ``{"sigma": ...}`` for
            ``AuxDifficulty("sigma")``. For ``TensorDict`` inputs, nest by
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
            difficulty = (
                None if self._difficulty is None else self._difficulty(aux_field)
            )
            raw = check_finite(
                key,
                "the nonconformity scores",
                self._score.score(prediction_field, target_field, aux=aux_field),
            )
            return self._reduce(_normalized_scores(raw, difficulty, key))

        if points is not None:
            check_points(points)
        self._collect(prediction, target, aux, points, stage)


class FunctionalBandCalibrator(_ScaledCalibratorBase):
    r"""Calibrate a band that contains every point of each new field at once.

    Use this calibrator when a single band must cover every point of a field
    at the same time, for example to bound the worst-case error over a mesh.
    Meshes may differ between calibration and deployment. The band's
    half-width at point :math:`x` is :math:`\text{threshold} \cdot s(x)`,
    where :math:`s` is the optional ``AuxDifficulty`` scale; without one the
    width is constant.

    Guarantee: a new field lies inside the band at every point with
    probability at least :math:`1 - \alpha`.

    Parameters
    ----------
    score : AbsoluteErrorScore | NormalizedErrorScore | QuantileRegressionScore
        Defines the interval (see
        :class:`~physicsnemo.experimental.uq.conformal.CellwiseCalibrator`).
        The calibrator keeps a copy.
    alpha : float
        Target miscoverage level in :math:`(0, 1)`.
    difficulty : AuxDifficulty, optional
        Per-point scale :math:`s(x)` that widens the band where it is large.
        ``None`` gives a constant width.
    keys : Sequence[str], optional
        Names of the ``TensorDict`` fields to calibrate (see
        :class:`~physicsnemo.experimental.uq.conformal.CellwiseCalibrator`).
        By default every field is calibrated.

    Raises
    ------
    ValueError
        If ``score`` already divides by the aux key ``difficulty`` reads.

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

    def finalize(self) -> ConformalPredictor:
        r"""Fit the band threshold and return the predictor.

        Returns
        -------
        ConformalPredictor
            A functional
            :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`
            with one float64 scalar threshold per field. It applies the same
            ``AuxDifficulty``, so pass the same ``aux`` keys at prediction.

        Raises
        ------
        RuntimeError
            If no samples were collected.
        ValueError
            If :math:`n_{cal} < (1 - \alpha) / \alpha`, for example fewer
            than 9 samples for ``alpha=0.1``. Collect more or raise ``alpha``.
        """
        self._require_finalizable()
        return self._build_predictor(
            "functional",
            self._conformal_thresholds(),
            difficulty=self._difficulty,
        )

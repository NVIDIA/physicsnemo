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

:class:`~physicsnemo.experimental.uq.conformal.PointwiseCalibrator` fits an
interval for each output element, when every sample has the same output
layout.

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
    check_aux,
    check_exact_shape,
    check_finite,
    check_point_alignment,
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
    _SCORE_REGISTRY,
    NonconformityScore,
    _inversion_message,
    _snapshot_strategy,
    _uncovered_count,
)

__all__ = ["PointwiseCalibrator"]

# One field's staged record: (key, prediction, target, aux) -> stored value.
_Stage = Callable[[str, Tensor, Tensor, Mapping[str, Tensor] | None], object]


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
    every call.

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

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

r"""Fitted conformal predictors that turn model outputs into intervals."""

import copy
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from ._utils import (
    TENSOR_KEY,
    TIERS,
    Tier,
    _field_label,
    check_aux,
    check_point_alignment,
    check_real,
    fetch_ints,
    field_items,
    pack_fields,
    points_fingerprint,
    require_container_kind,
    require_feasible_alpha,
    require_mesh,
    slice_aux,
    validate_provenance,
)
from .artifacts import _parse_artifact, _save_artifact
from .diagnostics import CoverageAccumulator
from .scores import _SCORE_REGISTRY, _NonconformityScore, _Score, _snapshot_strategy

__all__ = ["ConformalPredictor"]


def _validate_mesh_fingerprint(value: object) -> str:
    """Check the ``<dtype>-<n_points>x<n_dims>-<32 hex digits>`` mesh checksum."""
    if type(value) is not str or not re.fullmatch(
        r"[a-z0-9_]+-[1-9][0-9]*x[1-9][0-9]*-[0-9a-f]{32}", value
    ):
        raise ValueError(
            "mesh_fingerprint must be '<dtype>-<n_points>x<n_dims>-<checksum>' "
            "with a 32-digit lowercase hex checksum."
        )
    return value


def _validate_thresholds(
    thresholds: Tensor | TensorDict,
    score_type: type[_NonconformityScore],
) -> dict[str, Tensor]:
    """Check thresholds against the score, and return detached copies."""
    out: dict[str, Tensor] = {}
    for key, value in field_items(thresholds):
        check_real(key, "threshold", value)
        if value.numel() == 0:
            raise ValueError(f"{_field_label(key)}: empty threshold tensor.")
        if value.ndim == 0:
            raise ValueError(
                f"{_field_label(key)}: cellwise thresholds must have at least one "
                "dimension, got a scalar."
            )
        if not score_type._signed_threshold and bool((value < 0).any()):
            raise ValueError(
                f"{_field_label(key)}: negative threshold, but "
                f"{score_type.__name__} scores are never "
                "negative. Only QuantileRegressionScore allows negative thresholds."
            )
        out[key] = value.detach().clone()
    return out


class ConformalPredictor:
    r"""Turn new model outputs into intervals with a calibrated guarantee.

    You normally get a predictor from a calibrator's ``finalize()`` or from
    :meth:`load`, then call :meth:`predict_interval` on each new model
    output. Use :meth:`coverage_accumulator` to check coverage on held-out
    data and :meth:`save` to reuse the predictor later. Build one directly
    only to restore thresholds you computed elsewhere.

    The predictor has the guarantee of the calibrator that produced it and
    works only on the calibration mesh (same coordinates, dtype, and point
    order).

    Parameters
    ----------
    tier : {"cellwise"}
        Calibrator that produced ``thresholds``: ``"cellwise"`` for
        :class:`~physicsnemo.experimental.uq.conformal.CellwiseCalibrator`.
    score : AbsoluteErrorScore | NormalizedErrorScore | QuantileRegressionScore
        Score used during calibration. The predictor keeps its own copy, so
        later changes to ``score`` have no effect.
    alpha : float
        Target miscoverage level in :math:`(0, 1)`.
    n_cal : int
        Number of calibration samples.
    thresholds : torch.Tensor | TensorDict
        A tensor for one field, or a ``TensorDict`` keyed by field name, with
        one tensor of shape :math:`(*\text{dims})` per field. Values must be
        finite, and nonnegative unless ``score`` is a
        :class:`~physicsnemo.experimental.uq.conformal.QuantileRegressionScore`.
    points : torch.Tensor
        Calibration mesh coordinates of shape
        :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`. The predictor
        keeps only a checksum of them, and :meth:`predict_interval` accepts
        only the same coordinates, dtype, and point order.

    Raises
    ------
    ValueError
        If any of these hold:

        - ``tier`` is not ``"cellwise"``.
        - :math:`n_{cal} < (1 - \alpha) / \alpha`: collect more samples or
          raise ``alpha``.
        - ``points`` is missing, empty, not 2-D, or not finite.
        - A threshold is empty, not finite, or a scalar.
        - A threshold is negative and ``score`` is not
          ``QuantileRegressionScore``.
    TypeError
        If ``score`` is not a built-in class, or a threshold is not floating
        point.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import (
    ...     AbsoluteErrorScore, ConformalPredictor,
    ... )
    >>> _ = torch.manual_seed(0)
    >>> points = torch.rand(50, 3)
    >>> predictor = ConformalPredictor(
    ...     tier="cellwise", score=AbsoluteErrorScore(), alpha=0.1, n_cal=20,
    ...     thresholds=torch.full((50, 2), 0.3), points=points,
    ... )
    >>> lo, hi = predictor.predict_interval(torch.randn(50, 2), points=points)
    >>> lo.shape
    torch.Size([50, 2])
    """

    def __init__(
        self,
        *,
        tier: Tier,
        score: _Score,
        alpha: float,
        n_cal: int,
        thresholds: Float[Tensor, "*dims"] | TensorDict,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
    ) -> None:
        self._init(
            tier=tier,
            score=score,
            alpha=alpha,
            n_cal=n_cal,
            thresholds=thresholds,
            mesh_fingerprint=None if points is None else points_fingerprint(points),
        )

    @classmethod
    def _from_state(cls, **state) -> "ConformalPredictor":
        """Build a predictor from a stored mesh checksum and provenance."""
        predictor = cls.__new__(cls)
        predictor._init(**state)
        return predictor

    def _init(
        self,
        *,
        tier: Tier,
        score: _Score,
        alpha: float,
        n_cal: int,
        thresholds: Tensor | TensorDict,
        mesh_fingerprint: str | None = None,
        provenance: Mapping | None = None,
    ) -> None:
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}.")
        require_feasible_alpha(n_cal, alpha)
        alpha = float(alpha)
        score_snapshot = _snapshot_strategy(score, _SCORE_REGISTRY, "score")

        if mesh_fingerprint is None:
            raise ValueError(
                "A cellwise predictor requires points=, the calibration "
                "mesh coordinates."
            )
        mesh_snapshot = _validate_mesh_fingerprint(mesh_fingerprint)

        self._tier = tier
        self._score = score_snapshot
        self._alpha = alpha
        self._n_cal = n_cal
        self._thresholds_by_key = _validate_thresholds(thresholds, type(score_snapshot))
        self._mesh_fingerprint = mesh_snapshot
        self._provenance = {} if provenance is None else validate_provenance(provenance)

    @property
    def tier(self) -> Tier:
        r"""Calibrator that produced this predictor: ``"cellwise"``."""
        return self._tier

    @property
    def alpha(self) -> float:
        r"""Target miscoverage level."""
        return self._alpha

    @property
    def n_cal(self) -> int:
        r"""Number of calibration samples behind the thresholds."""
        return self._n_cal

    @property
    def score(self) -> _NonconformityScore:
        r"""A copy of the calibration score; editing it has no effect here."""
        return copy.deepcopy(self._score)

    @property
    def provenance(self) -> dict:
        r"""A copy of the metadata restored by :meth:`load`, or ``{}``."""
        return copy.deepcopy(self._provenance)

    @property
    def _tensor_mode(self) -> bool:
        """Whether calibration used a plain tensor instead of a ``TensorDict``."""
        return set(self._thresholds_by_key) == {TENSOR_KEY}

    @property
    def keys(self) -> list[str] | None:
        r"""Sorted calibrated field names, or ``None`` if calibrated on a tensor."""
        return None if self._tensor_mode else sorted(self._thresholds_by_key)

    @property
    def thresholds(self) -> Tensor | TensorDict:
        r"""A copy of the thresholds, as a tensor or ``TensorDict`` like the input."""
        return pack_fields(
            {
                key: value.detach().clone()
                for key, value in self._thresholds_by_key.items()
            }
        )

    def to(self, device: torch.device | str) -> "ConformalPredictor":
        r"""Move the thresholds to ``device`` in place, like ``nn.Module.to``.

        Call it once with the device of your model outputs so
        :meth:`predict_interval` does not copy the thresholds on every call.

        Parameters
        ----------
        device : torch.device | str
            Target device for the threshold tensors.

        Returns
        -------
        ConformalPredictor
            ``self``, for chaining.
        """
        self._thresholds_by_key = {
            key: value.to(device=device)
            for key, value in self._thresholds_by_key.items()
        }
        return self

    def _threshold_for(self, key: str, prediction: Tensor) -> Tensor:
        threshold = self._thresholds_by_key[key]
        if threshold.shape != prediction.shape:
            raise ValueError(
                f"{_field_label(key)}: prediction shape {tuple(prediction.shape)} "
                f"differs from calibrated shape {tuple(threshold.shape)}."
            )
        return threshold

    def _bounds(
        self,
        items: list[tuple[str, Tensor]],
        aux: Mapping | None,
        points: Tensor | None,
        pending: list[Tensor] | None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Validate the inputs and compute bounds; ``pending`` as in ``check_finite``."""
        require_mesh(points, self._mesh_fingerprint, pending)

        lo_out: dict[str, Tensor] = {}
        hi_out: dict[str, Tensor] = {}
        for key, prediction_field in items:
            check_point_alignment(key, prediction_field, points, "prediction")
            aux_field = slice_aux(aux, key)
            check_real(key, "prediction", prediction_field, pending)
            check_aux(key, self._score, prediction_field, aux_field, pending)
            threshold = self._threshold_for(key, prediction_field)
            lo_out[key], hi_out[key] = self._score.interval(
                prediction_field,
                threshold.to(device=prediction_field.device),
                aux=aux_field,
            )
        return lo_out, hi_out

    def predict_interval(
        self,
        prediction: Float[Tensor, "*dims"] | TensorDict,
        *,
        aux: Mapping[str, Float[Tensor, "*dims"]] | Mapping[str, Mapping] | None = None,
        points: Float[Tensor, "n_points n_spatial_dims"] | None = None,
    ) -> tuple[
        Float[Tensor, "*dims"] | TensorDict, Float[Tensor, "*dims"] | TensorDict
    ]:
        r"""Return lower and upper bounds for one model output.

        Pass the same kind of input used at calibration. A ``TensorDict``
        may contain extra fields (for example when calibration used
        ``keys=``); the bounds contain only the calibrated fields.

        Parameters
        ----------
        prediction : torch.Tensor | TensorDict
            Model output for one sample, of shape :math:`(*\text{dims})`, or
            a ``TensorDict`` of such tensors. The shape must equal the
            calibration shape, with leading dimension
            :math:`n_{\text{points}}`.
        aux : Mapping[str, torch.Tensor] | Mapping[str, Mapping], optional
            Extra tensors the score reads, each with the shape of the
            prediction: ``"sigma"`` for
            :class:`~physicsnemo.experimental.uq.conformal.NormalizedErrorScore`,
            ``"lo"`` and ``"hi"`` for
            :class:`~physicsnemo.experimental.uq.conformal.QuantileRegressionScore`.
            For ``TensorDict`` inputs, nest by field name, for example
            ``aux={"pressure": {"sigma": s}}``.
        points : torch.Tensor
            Mesh coordinates of shape
            :math:`(n_{\text{points}}, n_{\text{spatial\_dims}})`. They must
            match the calibration coordinates, dtype, and point order.

        Returns
        -------
        tuple[torch.Tensor | TensorDict, torch.Tensor | TensorDict]
            Lower and upper bounds ``(lo, hi)`` with the container type,
            dtype, and shape :math:`(*\text{dims})` of ``prediction``.

        Raises
        ------
        TypeError
            If ``prediction`` is a plain tensor for a predictor calibrated on
            ``TensorDict`` fields, or the reverse.
        ValueError
            If a shape differs from calibration, values are not finite, or
            ``points`` is missing or describes a different mesh.
        KeyError
            If a ``TensorDict`` prediction lacks a calibrated field.

        Notes
        -----
        Bounds are rounded outward so the stated coverage holds in the
        prediction dtype. Near the dtype's limits this can give infinite
        endpoints. If you need finite bounds, upcast the prediction first or
        rescale the model outputs and recalibrate. Do not clip bounds
        inward: that can remove coverage.

        Each call checks every input value and the ``points`` checksum on the
        inputs' device, then synchronizes with the GPU once to read the
        results. This method is not intended for use inside ``torch.compile``
        regions.
        """
        selection = self.keys
        require_container_kind(prediction, selection, "This predictor", "prediction")
        items = field_items(prediction, selection)
        pending: list[Tensor] = []
        lo_out, hi_out = self._bounds(items, aux, points, pending)
        if not all(fetch_ints(pending)):
            # Repeat with one sync per check to raise the first error.
            lo_out, hi_out = self._bounds(items, aux, points, None)

        if isinstance(prediction, TensorDict):
            lo = prediction.empty()
            hi = prediction.empty()
            lo.update(lo_out)
            hi.update(hi_out)
            return lo, hi
        return pack_fields(lo_out), pack_fields(hi_out)

    def coverage_accumulator(self) -> CoverageAccumulator:
        r"""Return an accumulator that checks this predictor on held-out data.

        Feed it the ``(lo, hi)`` from :meth:`predict_interval` together with
        the matching targets; its report measures the same guarantee this
        predictor states.

        Returns
        -------
        CoverageAccumulator
            A new, empty
            :class:`~physicsnemo.experimental.uq.conformal.CoverageAccumulator`
            set to this predictor's tier, ``alpha``, ``n_cal``, and fields.
        """
        return CoverageAccumulator(
            tier=self._tier,
            alpha=self._alpha,
            n_cal=self._n_cal,
            keys=self.keys,
        )

    def save(self, path: Path | str, *, provenance: Mapping | None = None) -> None:
        r"""Write the predictor to ``path`` so :meth:`load` can restore it.

        The file holds only tensors and plain metadata, so it loads with
        ``torch.load(..., weights_only=True)``. The saved file is checked
        before it replaces ``path``, so a failed save leaves any existing
        file unchanged.

        Parameters
        ----------
        path : Path | str
            Destination file. Missing parent directories are created.
        provenance : Mapping, optional
            Strict-JSON metadata to save with the predictor, for example
            ``{"dataset": "holdout-v1"}``. ``None`` saves the current
            :attr:`provenance`.

        Returns
        -------
        None
            The artifact is written to ``path``.
        """
        state = {
            "tier": self._tier,
            "score": self._score,
            "alpha": self._alpha,
            "n_cal": self._n_cal,
            "thresholds": self._thresholds_by_key,
            "mesh_fingerprint": self._mesh_fingerprint,
            "provenance": self._provenance if provenance is None else provenance,
        }
        _save_artifact(state, path, lambda written: self._read(written, path))

    @classmethod
    def load(
        cls, path: Path | str, map_location: str | torch.device = "cpu"
    ) -> "ConformalPredictor":
        r"""Restore a predictor written by :meth:`save`.

        Parameters
        ----------
        path : Path | str
            Artifact written by :meth:`save`.
        map_location : str | torch.device, optional, default="cpu"
            Device for the loaded thresholds.

        Returns
        -------
        ConformalPredictor
            The restored predictor.

        Raises
        ------
        ValueError
            If the file is not a conformal predictor artifact, was written by
            an incompatible version (recalibrate and save again), or holds
            invalid predictor state.
        """
        return cls._read(path, path, map_location)

    @classmethod
    def _read(
        cls,
        source: Path | str,
        path: Path | str,
        map_location: str | torch.device = "cpu",
    ) -> "ConformalPredictor":
        """Load ``source``; name ``path`` (the user's file) in validation errors."""
        payload = torch.load(source, map_location=map_location, weights_only=True)
        try:
            return cls._from_state(**_parse_artifact(payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid conformal artifact at {path}: {exc}") from exc

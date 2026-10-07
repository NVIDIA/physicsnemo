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

r"""Nonconformity scores for conformal prediction.

A score measures, per element, how far a target falls from a prediction,
and turns a calibrated threshold into an interval ``(lo, hi)``. Pass a score
to a calibrator; you rarely need to call its methods yourself.

Choose the score by what your model outputs:

- :class:`AbsoluteErrorScore`: a point prediction only. The interval
  half-width comes only from the calibrated threshold.
- :class:`NormalizedErrorScore`: a mean and a standard deviation, with the
  standard deviation passed as ``aux["sigma"]``. Intervals widen where
  sigma is large.
- :class:`QuantileRegressionScore`: lower and upper quantile heads, passed
  as ``aux["lo"]`` and ``aux["hi"]``. Intervals follow the heads.

For tensor inputs, ``aux`` maps each key to a tensor; for ``TensorDict``
inputs, it maps each field name to such a mapping.

:class:`AuxDifficulty` is an optional per-point scale for the functional
band calibrator.
"""

import copy
from collections.abc import Mapping

import torch
from jaxtyping import Float
from torch import Tensor

from ._utils import cast_directed, check_real, clamp_min_floor, positive_finite_float

__all__ = [
    "AbsoluteErrorScore",
    "AuxDifficulty",
    "NormalizedErrorScore",
    "QuantileRegressionScore",
]


def _coarsest_finfo(*tensors: Tensor) -> torch.finfo:
    """``finfo`` of the least precise floating dtype among the inputs."""
    return max((torch.finfo(t.dtype) for t in tensors), key=lambda fi: fi.eps)


def _slack_threshold(threshold: Tensor, *dtype_sources: Tensor) -> Tensor:
    """Float64 threshold, inflated so rounding cannot exclude an admitted target."""
    t = threshold.to(torch.float64)
    fi = _coarsest_finfo(*dtype_sources)
    return t + (t.abs() * (4.0 * fi.eps) + 4.0 * fi.tiny)


def _outward_interval(
    prediction: Tensor, lo64: Tensor, hi64: Tensor
) -> tuple[Tensor, Tensor]:
    """Cast float64 bounds to the prediction dtype, rounding outward."""
    return (
        cast_directed(lo64, prediction.dtype, up=False),
        cast_directed(hi64, prediction.dtype, up=True),
    )


def _require_aux(
    aux: Mapping[str, Tensor] | None, keys: tuple[str, ...], score_name: str
) -> None:
    """Raise ``ValueError`` if ``aux`` lacks any of ``keys``."""
    missing = [k for k in keys if not isinstance(aux, Mapping) or k not in aux]
    if missing:
        raise ValueError(
            f"{score_name} requires aux entries {list(keys)}; missing {missing}. "
            "Pass aux={key: tensor} (or, for TensorDict inputs, "
            "aux={field: {key: tensor}})."
        )


class _NonconformityScore:
    """Base class for the built-in scores.

    ``aux_keys`` lists the aux entries a score reads; ``_scale_aux_keys`` lists
    the ones it divides by. ``_kind`` is the saved name, ``_saved_kwargs`` maps
    each saved constructor argument to its exact type, and ``_signed_threshold``
    allows negative thresholds.
    """

    aux_keys: tuple[str, ...] = ()
    _scale_aux_keys: tuple[str, ...] = ()
    _kind: str
    _saved_kwargs: dict[str, type] = {}
    _signed_threshold: bool = False

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        r"""Score how far target falls from prediction, per element; larger is worse.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output of shape :math:`(*\text{dims})`.
        target : torch.Tensor
            Observed values, same shape as ``prediction``.
        aux : Mapping[str, torch.Tensor], optional
            Extra model outputs the score needs, each a finite real
            floating-point tensor of the same shape as ``prediction``:
            ``{"sigma": ...}`` for ``NormalizedErrorScore`` and
            ``{"lo": ..., "hi": ...}`` for ``QuantileRegressionScore``.
            ``AbsoluteErrorScore`` needs none.

        Returns
        -------
        torch.Tensor
            Nonconformity scores of shape :math:`(*\text{dims})`.

        Raises
        ------
        ValueError
            If a required ``aux`` entry is missing.
        """
        raise NotImplementedError

    def interval(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        r"""Turn a calibrated ``threshold`` into an interval ``(lo, hi)``.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output of shape :math:`(*\text{dims})`.
        threshold : torch.Tensor
            Fitted threshold, broadcastable against ``prediction``: one value
            per element of shape :math:`(*\text{dims})` for the cellwise
            calibrator, or a scalar (already multiplied by the
            ``AuxDifficulty`` scale, if any) for the functional band
            calibrator.
        aux : Mapping[str, torch.Tensor], optional
            The same ``aux`` entries that :meth:`score` needs.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Lower and upper bounds ``(lo, hi)``, each of shape
            :math:`(*\text{dims})` and in the dtype of ``prediction``. Bounds
            are rounded outward, so every target the threshold admits lies
            inside them, even in low precision.

        Raises
        ------
        ValueError
            If a required ``aux`` entry is missing.
        """
        raise NotImplementedError


class AbsoluteErrorScore(_NonconformityScore):
    r"""Absolute error :math:`|y - \hat y|`, for models that output a point prediction.

    The simplest choice: it needs no ``aux`` inputs and no change to the
    model. A threshold :math:`t` gives the interval
    :math:`[\hat y - t, \hat y + t]`. This is the split conformal score of
    `Distribution-Free Predictive Inference for Regression
    <https://arxiv.org/abs/1604.04173>`_ (Lei et al., 2018).

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import AbsoluteErrorScore
    >>> _ = torch.manual_seed(0)
    >>> score = AbsoluteErrorScore()
    >>> prediction, target = torch.randn(50, 2), torch.randn(50, 2)
    >>> score.score(prediction, target).shape
    torch.Size([50, 2])
    >>> threshold = torch.tensor(0.5, dtype=torch.float64)
    >>> lo, hi = score.interval(prediction, threshold)
    >>> lo.shape, bool((hi > lo).all())
    (torch.Size([50, 2]), True)
    """

    _kind = "absolute_error"

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        return (target - prediction).abs()

    def interval(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        p = prediction.to(torch.float64)
        t = _slack_threshold(threshold, prediction)
        return _outward_interval(prediction, p - t, p + t)


class NormalizedErrorScore(_NonconformityScore):
    r"""Error scaled by predicted spread, :math:`|y - \mu| / \max(\sigma, \epsilon)`.

    Use it when the model outputs a mean :math:`\mu` (passed as
    ``prediction``) and a standard deviation :math:`\sigma` (passed as
    ``aux["sigma"]``), for example from an NLL head, MC dropout, or an
    ensemble. A threshold :math:`t` gives
    :math:`[\mu - t\sigma, \mu + t\sigma]`, so intervals are wider where the
    model is less certain.

    Parameters
    ----------
    eps : float, optional, default=1e-8
        Smallest :math:`\sigma` used, in the units of ``sigma``, so zero or
        near-zero spread does not blow up the score. Pick it for your output
        scale, for example a small fraction of a typical ``sigma``.

    Notes
    -----
    Rules for ``aux["sigma"]``:

    - Pass the raw predictive standard deviation, for example
      ``variance.sqrt()`` from an ensemble; do not clamp it yourself.
    - Compute it the same way at calibration and at prediction.
    - Values below ``eps`` are raised to ``eps`` (or to the dtype's smallest
      positive normal value, if larger). An ``eps`` too large for the dtype
      raises ``ValueError``.
    - It must be finite. A missing ``aux["sigma"]`` raises ``ValueError``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import NormalizedErrorScore
    >>> _ = torch.manual_seed(0)
    >>> score = NormalizedErrorScore()
    >>> prediction, target = torch.randn(50, 2), torch.randn(50, 2)
    >>> aux = {"sigma": torch.rand(50, 2) + 0.1}
    >>> score.score(prediction, target, aux=aux).shape
    torch.Size([50, 2])
    >>> threshold = torch.tensor(2.0, dtype=torch.float64)
    >>> lo, hi = score.interval(prediction, threshold, aux=aux)
    >>> hi.shape
    torch.Size([50, 2])
    """

    aux_keys = ("sigma",)
    _scale_aux_keys = ("sigma",)  # the residual is divided by sigma
    _kind = "normalized_error"
    _saved_kwargs = {"eps": float}

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = positive_finite_float(eps, "eps")

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        _require_aux(aux, self.aux_keys, type(self).__name__)
        sigma = clamp_min_floor(aux["sigma"], self.eps)
        # Round the float64 quotient up so thresholds can only grow.
        s64 = (target.to(torch.float64) - prediction.to(torch.float64)).abs()
        s64 = s64 / sigma.to(torch.float64)
        out_dtype = torch.result_type(prediction, target)
        return cast_directed(s64, out_dtype, up=True)

    def interval(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        _require_aux(aux, self.aux_keys, type(self).__name__)
        # Same clamp as score(), so the interval inverts the score.
        sigma = clamp_min_floor(aux["sigma"], self.eps).to(torch.float64)
        p = prediction.to(torch.float64)
        # Sigma converts to float64 without rounding, so it needs no slack.
        half = _slack_threshold(threshold, prediction) * sigma
        return _outward_interval(prediction, p - half, p + half)


class QuantileRegressionScore(_NonconformityScore):
    r"""Conformalized quantile regression (CQR) score, for models with quantile heads.

    Use it when the model outputs lower and upper quantile estimates
    :math:`q_{lo}` and :math:`q_{hi}`, passed as ``aux["lo"]`` and
    ``aux["hi"]``. The score :math:`\max(q_{lo} - y,\; y - q_{hi})` is the
    signed distance to the nearest violated bound, negative inside the band.
    A threshold :math:`t` gives :math:`[q_{lo} - t, q_{hi} + t]`. When the
    heads already cover more than needed, :math:`t` is negative and
    calibration shrinks the band. Introduced in `Conformalized Quantile
    Regression <https://arxiv.org/abs/1905.03222>`_ (Romano, Patterson and
    Candes, 2019).

    Notes
    -----
    The score ignores the values of ``prediction``; pass the point prediction
    anyway, because the returned interval takes its dtype. Calling
    :meth:`score` or :meth:`interval` without both ``aux["lo"]`` and
    ``aux["hi"]`` raises ``ValueError``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import QuantileRegressionScore
    >>> _ = torch.manual_seed(0)
    >>> score = QuantileRegressionScore()
    >>> prediction, target = torch.randn(50, 2), torch.randn(50, 2)
    >>> aux = {"lo": prediction - 1.0, "hi": prediction + 1.0}
    >>> score.score(prediction, target, aux=aux).shape
    torch.Size([50, 2])
    >>> threshold = torch.tensor(0.25, dtype=torch.float64)
    >>> lo, hi = score.interval(prediction, threshold, aux=aux)
    >>> lo.shape
    torch.Size([50, 2])
    """

    aux_keys = ("lo", "hi")
    _kind = "quantile_regression"
    _signed_threshold = True

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        _require_aux(aux, self.aux_keys, type(self).__name__)
        return torch.maximum(aux["lo"] - target, target - aux["hi"])

    def interval(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        _require_aux(aux, self.aux_keys, type(self).__name__)
        t = _slack_threshold(threshold, prediction, aux["lo"], aux["hi"])
        lo64 = aux["lo"].to(torch.float64) - t
        hi64 = aux["hi"].to(torch.float64) + t
        return _outward_interval(prediction, lo64, hi64)


_Score = AbsoluteErrorScore | NormalizedErrorScore | QuantileRegressionScore
"""Built-in score types accepted by calibrators and predictors."""

_SCORE_REGISTRY: dict[str, type[_NonconformityScore]] = {
    cls._kind: cls
    for cls in (AbsoluteErrorScore, NormalizedErrorScore, QuantileRegressionScore)
}
"""Serialization names of the built-in score types."""


class AuxDifficulty:
    r"""Per-point scale read from ``aux`` that widens the band where it is large.

    Use it with
    :class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator`
    when the model outputs a per-point uncertainty estimate (a predicted sigma,
    MC-dropout or ensemble spread). The fitted threshold is multiplied by the
    scale :math:`s(x)`, which is read at each point, so point sets may differ
    between samples. Pass the same ``aux`` entry at calibration and at
    prediction. Without an ``AuxDifficulty`` (:math:`s = 1`), coverage still
    holds, but bands can be wider than needed.

    For an entry of shape :math:`(n_{\text{points}}, C)` or with more
    trailing dimensions, :math:`s` is the maximum over the trailing
    dimensions, so each point gets one scale large enough for every channel.

    Parameters
    ----------
    key : str, optional, default="sigma"
        Name of the ``aux`` entry to read.
    eps : float, optional, default=1e-8
        Smallest allowed :math:`s`; smaller values, including zero and
        negatives, are raised to it.

    Notes
    -----
    Do not pair ``AuxDifficulty("sigma")`` with
    :class:`~physicsnemo.experimental.uq.conformal.NormalizedErrorScore`: the
    score already divides by sigma, so intervals would scale by sigma twice.
    Calibrators and predictors raise ``ValueError`` on that pair; use
    ``AbsoluteErrorScore`` with ``AuxDifficulty``, or ``NormalizedErrorScore``
    without it.

    Whatever produces the scale must not see the calibration samples: fit it
    on training data or a separate split, or the coverage guarantee no
    longer holds.

    The entry must be present on every calibration and prediction call and
    must be a finite floating-point tensor; NaN or infinity raises an error.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import AuxDifficulty
    >>> _ = torch.manual_seed(0)
    >>> difficulty = AuxDifficulty("sigma")
    >>> sigma = torch.rand(50, 3) + 0.1
    >>> difficulty(aux={"sigma": sigma}).shape
    torch.Size([50])
    """

    _kind = "aux"
    _saved_kwargs = {"key": str, "eps": float}

    def __init__(self, key: str = "sigma", eps: float = 1e-8) -> None:
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}.")
        if not key:
            raise ValueError("key must be a non-empty string.")
        self.key = key
        self.eps = positive_finite_float(eps, "eps")

    def __call__(self, aux: Mapping[str, Tensor]) -> Float[Tensor, " n_points"]:
        r"""Return one positive scale per point from ``aux[key]``.

        Calibrators and predictors call this for you; call it directly to
        inspect the scales.

        Parameters
        ----------
        aux : Mapping[str, torch.Tensor]
            Must contain ``key``, a finite real floating-point tensor of
            shape :math:`(n_{\text{points}}, *\text{dims})`.

        Returns
        -------
        torch.Tensor
            Positive scales of shape :math:`(n_{\text{points}},)`: the
            maximum over trailing dimensions, clamped below by the larger of
            ``eps`` and the dtype's smallest positive normal value.

        Raises
        ------
        ValueError
            If ``aux`` is ``None`` or lacks ``key``, or the entry holds NaN
            or infinity.
        TypeError
            If the entry is not a floating-point tensor (for example
            ``None``).
        """
        return self._scales(aux)

    def _scales(
        self, aux: Mapping[str, Tensor], pending: list[Tensor] | None = None
    ) -> Tensor:
        """``__call__`` that appends the finiteness flag to ``pending`` if given."""
        if not isinstance(aux, Mapping) or self.key not in aux:
            raise ValueError(
                f"AuxDifficulty requires aux entry '{self.key}' at every call; "
                f"pass aux={{'{self.key}': tensor}}."
            )
        s = aux[self.key]
        if not isinstance(s, Tensor):
            raise TypeError(
                f"AuxDifficulty aux '{self.key}' must be a torch.Tensor, got "
                f"{type(s).__name__}."
            )
        check_real(self.key, "AuxDifficulty input", s, pending)
        if s.ndim >= 2:
            # One scale per point, large enough for every channel.
            s = s.amax(dim=tuple(range(1, s.ndim)))
        return clamp_min_floor(s, self.eps)


def _check_no_double_scale(
    score: _NonconformityScore, difficulty: AuxDifficulty | None
) -> None:
    """Raise ``ValueError`` if ``difficulty`` reads a key the score divides by."""
    if difficulty is not None and difficulty.key in score._scale_aux_keys:
        raise ValueError(
            f"{type(score).__name__} already divides by aux "
            f"'{difficulty.key}', so AuxDifficulty('{difficulty.key}') would "
            "scale by it twice. Use AbsoluteErrorScore with AuxDifficulty, or "
            "drop difficulty."
        )


_DIFFICULTY_REGISTRY: dict[str, type[AuxDifficulty]] = {
    AuxDifficulty._kind: AuxDifficulty
}
"""Serialization names of the built-in difficulty types."""


def _strategy_kind(strategy: object, registry: Mapping[str, type]) -> str | None:
    """Serialization name of a built-in strategy, or ``None`` for any other type."""
    kind = getattr(type(strategy), "_kind", None)
    return kind if registry.get(kind) is type(strategy) else None


def _snapshot_strategy(strategy: object, registry: Mapping[str, type], what: str):
    """Deep-copy a built-in strategy; raise ``TypeError`` for any other type."""
    if _strategy_kind(strategy, registry) is None:
        names = ", ".join(sorted(cls.__name__ for cls in registry.values()))
        raise TypeError(
            f"{what} must be one of: {names}; got {type(strategy).__name__}. "
            "Subclasses are not supported."
        )
    return copy.deepcopy(strategy)

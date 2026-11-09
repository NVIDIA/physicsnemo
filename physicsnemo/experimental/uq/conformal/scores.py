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

:class:`AuxScale` is an optional per-point scale for the functional
band and risk-control calibrators.
"""

import copy
from collections.abc import Callable, Mapping

import torch
from jaxtyping import Float
from torch import Tensor

from ._utils import cast_directed, check_real, clamp_min_floor, positive_finite_float

__all__ = [
    "AbsoluteErrorScore",
    "AuxScale",
    "NonconformityScore",
    "NormalizedErrorScore",
    "QuantileRegressionScore",
    "check_score",
    "register_score",
]


def _coarsest_finfo(*tensors: Tensor) -> torch.finfo:
    """``finfo`` of the least precise floating dtype among the inputs."""
    return max((torch.finfo(t.dtype) for t in tensors), key=lambda fi: fi.eps)


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


_CONFIG_TYPES = (bool, int, float, str)
"""Types a score's ``config_fields`` may hold, so saved files load with ``weights_only``."""

_SCORE_REGISTRY: dict[str, type["NonconformityScore"]] = {}
"""Registered score classes by saved name."""


def _check_config_value(owner: str, name: str, value: object) -> None:
    """Require a ``config_fields`` value to be a ``bool``, ``int``, ``float``, or ``str``."""
    if type(value) not in _CONFIG_TYPES:
        raise TypeError(
            f"{owner}.{name} must be a bool, int, float, or str so a saved "
            f"predictor can store it; got {type(value).__name__}."
        )


class NonconformityScore:
    r"""Base class for nonconformity scores, built-in and custom.

    A score measures how far each target falls from the prediction, and
    turns a calibrated threshold back into an interval. To define your own,
    subclass it, implement :meth:`score` and :meth:`bounds`, and register the
    class with :func:`register_score` so calibrators accept it and saved
    predictors can restore it.

    The two methods must agree element by element: the targets with
    ``score(prediction, target) <= t`` are exactly the values in
    ``bounds(prediction, t)``, and that interval grows with ``t``. Do not
    override :meth:`interval`; it calls :meth:`bounds` in float64 and rounds
    the result outward so the guarantee holds in low precision.
    :func:`check_score` tests the agreement on sample data, and calibrators
    run the same test on their first sample.

    Attributes
    ----------
    aux_keys : tuple[str, ...]
        ``aux`` entries the score reads. Calibrators and predictors require
        them and check that each is finite and has the prediction's shape.
    config_fields : tuple[str, ...]
        Attributes that configure the score, each also a constructor
        argument of the same name. A saved predictor stores their values,
        which must be ``bool``, ``int``, ``float``, or ``str``, and passes
        them back to the constructor on load.
    signed_threshold : bool
        Whether scores can be negative, as for
        :class:`~physicsnemo.experimental.uq.conformal.QuantileRegressionScore`.
        Otherwise predictors reject negative thresholds.
    scale_aux_keys : tuple[str, ...]
        ``aux`` entries the score divides by, as ``NormalizedErrorScore``
        divides by ``"sigma"``. Calibrators reject an
        :class:`~physicsnemo.experimental.uq.conformal.AuxScale` that
        reads one of them, since the interval would scale by it twice.

    Examples
    --------
    A band proportional to the prediction, :math:`|y - \hat y| / |\hat y|`:

    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import (
    ...     NonconformityScore, check_score, register_score,
    ... )
    >>> @register_score("relative_error")
    ... class RelativeErrorScore(NonconformityScore):
    ...     config_fields = ("floor",)
    ...
    ...     def __init__(self, floor=1e-6):
    ...         self.floor = floor
    ...
    ...     def score(self, prediction, target, *, aux=None):
    ...         scale = prediction.abs().clamp_min(self.floor)
    ...         return (target - prediction).abs() / scale
    ...
    ...     def bounds(self, prediction, threshold, *, aux=None):
    ...         half = threshold * prediction.abs().clamp_min(self.floor)
    ...         return prediction - half, prediction + half
    >>> _ = torch.manual_seed(0)
    >>> prediction = torch.rand(50, 2) + 1.0
    >>> target = prediction * (1.0 + 0.1 * torch.randn(50, 2))
    >>> check_score(RelativeErrorScore(), prediction, target)
    """

    aux_keys: tuple[str, ...] = ()
    config_fields: tuple[str, ...] = ()
    signed_threshold: bool = False
    scale_aux_keys: tuple[str, ...] = ()
    _kind: str

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
            The entries named in ``aux_keys``, each a finite real
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
        raise NotImplementedError(f"{type(self).__name__} must implement score().")

    def bounds(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        r"""Return the targets whose score is at most ``threshold``, as ``(lo, hi)``.

        Implement this in a subclass; call :meth:`interval` to use it.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output of shape :math:`(*\text{dims})`, in float64.
        threshold : torch.Tensor
            Threshold in float64, broadcastable against ``prediction``.
        aux : Mapping[str, torch.Tensor], optional
            The ``aux`` entries, in their original dtype. Repeat any
            dtype-dependent step of :meth:`score`, such as a clamp, before
            converting them to float64.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Lower and upper bounds, broadcastable to the shape of
            ``prediction``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement bounds().")

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
            per element of shape :math:`(*\text{dims})` for the pointwise
            calibrator, or a scalar (already multiplied by the
            per-point ``AuxScale`` value, if any) for the functional band and
            risk-control calibrators.
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
        _require_aux(aux, self.aux_keys, type(self).__name__)
        fi = _coarsest_finfo(*self._precision_inputs(prediction, aux))
        # Grow the threshold so a score rounded down in a low-precision dtype
        # still admits its own target.
        t = threshold.to(torch.float64)
        t = t + t.abs() * (4.0 * fi.eps)
        lo, hi = self.bounds(prediction.to(torch.float64), t, aux=aux)
        # The absolute floor is in output units, so bounds() must not scale it.
        floor = 4.0 * fi.tiny
        lo = (lo.to(torch.float64) - floor).expand(prediction.shape)
        hi = (hi.to(torch.float64) + floor).expand(prediction.shape)
        return _outward_interval(prediction, lo, hi)

    def _precision_inputs(
        self, prediction: Tensor, aux: Mapping[str, Tensor] | None
    ) -> tuple[Tensor, ...]:
        """Tensors whose dtypes limit how precisely :meth:`score` is computed."""
        return (prediction, *(aux[key] for key in self.aux_keys))


def register_score(name: str) -> Callable[[type], type]:
    r"""Register a :class:`NonconformityScore` subclass under ``name``.

    Calibrators and predictors accept only registered score classes, and a
    saved predictor records the score by ``name``. To load that file in
    another process, import the module that registers the class first.

    Parameters
    ----------
    name : str
        Name stored in saved predictors. It must be unique.

    Returns
    -------
    Callable[[type], type]
        A class decorator that registers the class and returns it unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered to a different class.
    TypeError
        If ``name`` is not a non-empty string, the decorated object is not a
        :class:`NonconformityScore` subclass, or its ``config_fields`` is not
        a tuple of strings.

    Examples
    --------
    See :class:`NonconformityScore`.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(f"Score name must be a non-empty string, got {name!r}.")

    def decorator(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, NonconformityScore)):
            raise TypeError(
                f"@register_score({name!r}) needs a NonconformityScore subclass, "
                f"got {cls!r}."
            )
        fields = cls.config_fields
        if not isinstance(fields, tuple) or not all(isinstance(f, str) for f in fields):
            raise TypeError(
                f"{cls.__name__}.config_fields must be a tuple of attribute names."
            )
        existing = _SCORE_REGISTRY.get(name)
        # Re-running a module (for example in a notebook) re-registers the
        # same class under the same name.
        if existing is not None and (existing.__module__, existing.__qualname__) != (
            cls.__module__,
            cls.__qualname__,
        ):
            raise ValueError(
                f"Score name {name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}."
            )
        cls._kind = name
        _SCORE_REGISTRY[name] = cls
        return cls

    return decorator


def _uncovered_count(
    score: NonconformityScore,
    prediction: Tensor,
    target: Tensor,
    aux: Mapping[str, Tensor] | None,
) -> Tensor:
    """Number of targets outside the interval built from their own scores."""
    with torch.no_grad():
        own = score.score(prediction, target, aux=aux)
        lo, hi = score.interval(prediction, own, aux=aux)
        return (~((lo <= target) & (target <= hi))).sum()


def check_score(
    score: NonconformityScore,
    prediction: Float[Tensor, "*dims"],
    target: Float[Tensor, "*dims"],
    *,
    aux: Mapping[str, Tensor] | None = None,
) -> None:
    r"""Check that a score's interval contains each target at its own score.

    Uses each element's own score as its threshold, so every target must lie
    inside the resulting interval. A failure means :meth:`~NonconformityScore.bounds`
    does not invert :meth:`~NonconformityScore.score`, and intervals from
    that score would not keep the coverage guarantee. Calibrators run this
    check on their first sample; call it in your tests on representative
    data, in each dtype you use.

    Parameters
    ----------
    score : NonconformityScore
        The score to check.
    prediction : torch.Tensor
        Model output of shape :math:`(*\text{dims})`.
    target : torch.Tensor
        Observed values, same shape as ``prediction``.
    aux : Mapping[str, torch.Tensor], optional
        The ``aux`` entries the score reads.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any target lies outside the interval built from its own score.

    Examples
    --------
    See :class:`NonconformityScore`.
    """
    uncovered = int(_uncovered_count(score, prediction, target, aux))
    if uncovered:
        raise ValueError(_inversion_message(score, uncovered, target.numel()))


def _inversion_message(score: NonconformityScore, uncovered: int, total: int) -> str:
    return (
        f"{type(score).__name__}.bounds() does not invert score(): {uncovered} "
        f"of {total} targets fall outside the interval built from their own "
        "score. bounds() must return exactly the targets whose score is at most "
        "the threshold."
    )


@register_score("absolute_error")
class AbsoluteErrorScore(NonconformityScore):
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

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        return (target - prediction).abs()

    def bounds(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        return prediction - threshold, prediction + threshold


@register_score("normalized_error")
class NormalizedErrorScore(NonconformityScore):
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
    scale_aux_keys = ("sigma",)  # the residual is divided by sigma
    config_fields = ("eps",)

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

    def bounds(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        # Same clamp as score(), so the interval inverts the score.
        sigma = clamp_min_floor(aux["sigma"], self.eps).to(torch.float64)
        half = threshold * sigma
        return prediction - half, prediction + half

    def _precision_inputs(
        self, prediction: Tensor, aux: Mapping[str, Tensor] | None
    ) -> tuple[Tensor, ...]:
        # score() divides in float64 and rounds up, and sigma converts to
        # float64 exactly, so only the prediction dtype limits precision.
        return (prediction,)


@register_score("quantile_regression")
class QuantileRegressionScore(NonconformityScore):
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
    signed_threshold = True

    def score(
        self,
        prediction: Float[Tensor, "*dims"],
        target: Float[Tensor, "*dims"],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> Float[Tensor, "*dims"]:
        _require_aux(aux, self.aux_keys, type(self).__name__)
        return torch.maximum(aux["lo"] - target, target - aux["hi"])

    def bounds(
        self,
        prediction: Float[Tensor, "*dims"],
        threshold: Float[Tensor, "*dims"] | Float[Tensor, ""],
        *,
        aux: Mapping[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "*dims"], Float[Tensor, "*dims"]]:
        lo = aux["lo"].to(torch.float64) - threshold
        hi = aux["hi"].to(torch.float64) + threshold
        return lo, hi


class AuxScale:
    r"""Per-point scale read from ``aux`` that widens the band where it is large.

    Use it with
    :class:`~physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator` or
    :class:`~physicsnemo.experimental.uq.conformal.RiskControlCalibrator` when
    the model outputs a per-point uncertainty estimate (a predicted sigma,
    MC-dropout or ensemble spread). The fitted threshold is multiplied by the
    scale :math:`s(x)`, which is read at each point, so point sets may differ
    between samples. Pass the same ``aux`` entry at calibration and at
    prediction. Without an ``AuxScale`` (:math:`s = 1`), coverage still
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
    Do not pair ``AuxScale("sigma")`` with
    :class:`~physicsnemo.experimental.uq.conformal.NormalizedErrorScore`: the
    score already divides by sigma, so intervals would scale by sigma twice.
    Calibrators and predictors raise ``ValueError`` on that pair; use
    ``AbsoluteErrorScore`` with ``AuxScale``, or ``NormalizedErrorScore``
    without it.

    Whatever produces the scale must not see the calibration samples: fit it
    on training data or a separate split, or the coverage guarantee no
    longer holds.

    The entry must be present on every calibration and prediction call and
    must be a finite floating-point tensor; NaN or infinity raises an error.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.uq.conformal import AuxScale
    >>> _ = torch.manual_seed(0)
    >>> scale = AuxScale("sigma")
    >>> sigma = torch.rand(50, 3) + 0.1
    >>> scale(aux={"sigma": sigma}).shape
    torch.Size([50])
    """

    _kind = "aux"
    config_fields = ("key", "eps")

    def __init__(self, key: str = "sigma", eps: float = 1e-8) -> None:
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}.")
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
                f"AuxScale requires aux entry '{self.key}' at every call; "
                f"pass aux={{'{self.key}': tensor}}."
            )
        s = aux[self.key]
        if not isinstance(s, Tensor):
            raise TypeError(
                f"AuxScale aux '{self.key}' must be a torch.Tensor, got "
                f"{type(s).__name__}."
            )
        check_real(self.key, "AuxScale input", s, pending)
        if s.ndim >= 2:
            # One scale per point, the unit in which risk control counts misses.
            s = s.amax(dim=tuple(range(1, s.ndim)))
        return clamp_min_floor(s, self.eps)


def _check_no_double_scale(
    score: NonconformityScore, scale: AuxScale | None
) -> None:
    """Raise ``ValueError`` if ``scale`` reads a key the score divides by."""
    if scale is not None and scale.key in score.scale_aux_keys:
        raise ValueError(
            f"{type(score).__name__} already divides by aux "
            f"'{scale.key}', so AuxScale('{scale.key}') would "
            "scale by it twice. Use AbsoluteErrorScore with AuxScale, or "
            "drop scale=."
        )


_SCALE_REGISTRY: dict[str, type[AuxScale]] = {
    AuxScale._kind: AuxScale
}
"""Serialization names of the built-in scale types."""


def _strategy_kind(strategy: object, registry: Mapping[str, type]) -> str | None:
    """Registered name of a strategy, or ``None`` if its exact type is not registered."""
    kind = getattr(type(strategy), "_kind", None)
    return kind if registry.get(kind) is type(strategy) else None


def _snapshot_strategy(strategy: object, registry: Mapping[str, type], what: str):
    """Deep-copy a registered strategy; raise ``TypeError`` for any other type."""
    if _strategy_kind(strategy, registry) is None:
        names = ", ".join(sorted(cls.__name__ for cls in registry.values()))
        hint = (
            "Register a custom class with @register_score(name); a subclass "
            "needs its own registration."
            if registry is _SCORE_REGISTRY
            else "Subclasses are not supported."
        )
        raise TypeError(
            f"{what} must be an instance of a registered class ({names}); got "
            f"{type(strategy).__name__}. {hint}"
        )
    for name in type(strategy).config_fields:
        _check_config_value(type(strategy).__name__, name, getattr(strategy, name))
    return copy.deepcopy(strategy)

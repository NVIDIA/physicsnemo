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

r"""Held-out coverage and width checks for fitted conformal predictors."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from ._utils import (
    TENSOR_KEY,
    TIERS,
    Tier,
    _field_label,
    alpha_as_fraction,
    check_exact_shape,
    check_real,
    fetch_ints,
    field_items,
    normalize_keys,
    pack_fields,
    require_container_kind,
)

__all__ = ["CoverageAccumulator"]


@dataclass
class _FieldCounters:
    """Running totals for one field; the sums are float64 device tensors."""

    width_sum: Tensor | None = None
    width_count: int = 0
    n_samples: int = 0


def _add(total: Tensor | None, value: Tensor) -> Tensor:
    """``total + value`` on the device of ``total``; ``value`` alone if no total."""
    return value if total is None else total + value.to(device=total.device)


def _require(ok: Tensor, pending: list[Tensor] | None, problem: str) -> None:
    """Defer ``ok`` to ``pending``, or raise the width overflow error if false."""
    if pending is not None:
        pending.append(ok)
    elif not ok:
        raise ValueError(
            f"{problem} Rescale lo, hi, and target by the same factor and start "
            "a new accumulator with predictor.coverage_accumulator()."
        )


def _minimum_hits_at_target(n_samples: int, alpha: float) -> int:
    """Smallest hit count that reaches coverage ``1 - alpha`` over ``n_samples``."""
    return math.ceil(n_samples * (1 - alpha_as_fraction(alpha)))


class CoverageAccumulator:
    r"""Measure coverage and interval width of a fitted predictor on held-out data.

    Use it to check that a
    :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor` reaches
    its target on data not used for calibration, and to compare interval
    widths across scores. Create one with
    ``predictor.coverage_accumulator()``, which sets the parameters below
    from the predictor, so the report measures what the predictor's
    calibrator guarantees: how often each element lies inside its interval.

    Call :meth:`update` once per held-out sample, then :meth:`finalize` for
    the report.

    Parameters
    ----------
    tier : {"cellwise"}
        The predictor's ``tier``, naming the calibrator that produced it.
    alpha : float
        The predictor's target miscoverage level. Reported in the
        metadata and used to count cellwise elements at or above target
        coverage.
    n_cal : int
        Number of calibration samples of the predictor; reported only.
    keys : Sequence[str], optional
        Calibrated field names for ``TensorDict`` inputs, or ``None`` for
        plain tensors.

    Notes
    -----
    The accumulator does not communicate across ranks: its report covers
    only the samples passed to :meth:`update` in this process. To report on
    a held-out set split across ranks, send every sample to one process.

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
    >>> accumulator = predictor.coverage_accumulator()
    >>> for _ in range(5):
    ...     prediction = torch.randn(50, 2)
    ...     lo, hi = predictor.predict_interval(prediction, points=points)
    ...     accumulator.update(lo, hi, prediction + 0.1 * torch.randn(50, 2))
    >>> report = accumulator.finalize()
    >>> sorted(report)
    ['fields', 'meta']
    >>> report["fields"]["tensor"]["n_samples"]
    5
    """

    def __init__(
        self,
        *,
        tier: Tier,
        alpha: float,
        n_cal: int,
        keys: Sequence[str] | None = None,
    ) -> None:
        self._tier = tier
        self._alpha = alpha
        self._n_cal = n_cal
        self._keys = normalize_keys(keys)
        field_keys = (TENSOR_KEY,) if self._keys is None else self._keys
        self._counters = {key: _FieldCounters() for key in field_keys}
        self._element_hits: dict[str, Tensor] = {}

    def update(
        self,
        lo: Float[Tensor, "*dims"] | TensorDict,
        hi: Float[Tensor, "*dims"] | TensorDict,
        target: Float[Tensor, "*dims"] | TensorDict,
    ) -> None:
        r"""Add one held-out sample to the statistics.

        ``TensorDict`` inputs may contain fields beyond the calibrated ones (for
        example the model's full output as ``target``); only the calibrated
        fields are read, as in ``predict_interval`` on
        :class:`~physicsnemo.experimental.uq.conformal.ConformalPredictor`.

        Parameters
        ----------
        lo : torch.Tensor | TensorDict
            Lower bounds from ``predict_interval``, of shape
            :math:`(*\text{dims})`, or a ``TensorDict`` of such tensors.
        hi : torch.Tensor | TensorDict
            Upper bounds, same shape and container type as ``lo``.
        target : torch.Tensor | TensorDict
            Observed values, same shape and container type as ``lo``.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the inputs are plain tensors for a predictor calibrated on
            ``TensorDict`` fields (or the reverse), or use a non-floating
            dtype.
        KeyError
            If a ``TensorDict`` lacks a calibrated field.
        ValueError
            If ``target`` is empty, shapes differ, a value is NaN or
            infinite, a width or the running width sum overflows float64, or
            the sample shape differs from earlier samples.

        Notes
        -----
        A rejected sample is not counted, even in fields that passed.

        Running totals stay on the device of the first sample, so each call
        synchronizes with the GPU only once. Samples on another device are
        copied there.

        Infinite bounds are rejected: recompute them from an upcast prediction
        (casting the bounds afterward does not fix them), and if widths
        overflow float64, rescale ``lo``, ``hi``, and ``target`` by the same
        factor and start a new accumulator.
        """
        # field_items with self._keys returns the same keys as self._counters.
        inputs = (("lo", lo), ("hi", hi), ("target", target))
        for name, value in inputs:
            require_container_kind(
                value, self._keys, "This accumulator's predictor", name
            )
        containers = {
            name: dict(field_items(value, self._keys)) for name, value in inputs
        }
        pending: list[Tensor] = []
        staged = self._stage(containers, pending)
        if not all(fetch_ints(pending)):
            # Repeat with one sync per check to raise the first error.
            staged = self._stage(containers, None)

        for key, coverage, width_total, width_count in staged:
            counters = self._counters[key]
            counters.width_sum = width_total
            counters.width_count += width_count
            counters.n_samples += 1
            if key in self._element_hits:
                previous = self._element_hits[key]
                previous += coverage.to(device=previous.device)
            else:
                self._element_hits[key] = coverage

    def _stage(
        self, containers: dict[str, dict[str, Tensor]], pending: list[Tensor] | None
    ) -> list[tuple[str, Tensor, Tensor, int]]:
        """Check one sample and compute its updates without committing them.

        ``pending`` works as in ``check_finite``.
        """
        staged = []
        for key in self._counters:
            lo_field = containers["lo"][key]
            hi_field = containers["hi"][key]
            target_field = containers["target"][key]
            if target_field.numel() == 0:
                raise ValueError(f"{_field_label(key)}: empty target tensor.")
            check_exact_shape(key, "lo", lo_field, "target", target_field)
            check_exact_shape(key, "hi", hi_field, "target", target_field)
            check_real(key, "lo", lo_field, pending)
            check_real(key, "hi", hi_field, pending)
            check_real(key, "target", target_field, pending)

            element_covered = (target_field >= lo_field) & (target_field <= hi_field)
            widths = hi_field.to(torch.float64) - lo_field.to(torch.float64)
            _require(
                torch.isfinite(widths).all(),
                pending,
                f"{_field_label(key)}: interval width overflows float64.",
            )
            # A negative quantile-regression threshold can give hi < lo: width 0.
            widths = widths.clamp_min(0.0)
            # Nonnegative widths make this catch both sample and total overflow.
            width_total = _add(self._counters[key].width_sum, widths.sum())
            _require(
                torch.isfinite(width_total),
                pending,
                f"{_field_label(key)}: interval width sum overflows float64.",
            )

            match self._tier:
                case "cellwise":
                    # Hit counts stay on device until finalize().
                    coverage = element_covered.to(torch.int64)
                    previous = self._element_hits.get(key)
                    if previous is not None and previous.shape != coverage.shape:
                        raise ValueError(
                            f"{_field_label(key)}: a cellwise accumulator needs a "
                            f"fixed sample shape; got {tuple(coverage.shape)} after "
                            f"{tuple(previous.shape)}."
                        )
                case _:
                    raise ValueError(
                        f"tier must be one of {TIERS}, got {self._tier!r}."
                    )

            staged.append((key, coverage, width_total, widths.numel()))
        return staged

    def empirical_coverage_map(self) -> Float[Tensor, "*dims"] | TensorDict:
        r"""Fraction of samples in which each element was covered.

        Use it to locate the regions of the mesh that the predictor under- or
        over-covers.

        Returns
        -------
        torch.Tensor | TensorDict
            Float64 coverage per element, of the calibrated shape
            :math:`(*\text{dims})`, one tensor per field for ``TensorDict``
            inputs.

        Raises
        ------
        RuntimeError
            Before the first :meth:`update`.
        """
        if not self._element_hits:
            raise RuntimeError("No samples collected; call update() first.")
        return pack_fields(
            {
                key: hits.to(torch.float64) / self._counters[key].n_samples
                for key, hits in self._element_hits.items()
            }
        )

    def finalize(self) -> dict:
        r"""Return the coverage report as a JSON-serializable ``dict``.

        Returns
        -------
        dict
            ``{"meta": {...}, "fields": {<field>: {...}}}``. ``meta`` holds
            ``tier``, ``alpha``, ``n_cal``, and ``target_coverage``
            (``1 - alpha``).
            ``fields`` has one entry per calibrated field (key ``"tensor"``
            for plain tensors) with ``n_samples``, ``mean_interval_width``
            (mean width over every element of every sample),
            ``mean_element_coverage``, ``minimum_element_coverage``, and
            ``fraction_at_target`` (the fraction of elements whose coverage is
            at least the target).

            Statistics are ``None`` before the first :meth:`update`.
        """
        metadata = {
            "tier": self._tier,
            "alpha": self._alpha,
            "n_cal": self._n_cal,
            "target_coverage": 1.0 - self._alpha,
        }
        fields: dict = {}
        for key, counters in self._counters.items():
            n = counters.n_samples
            entry: dict = {
                "n_samples": n,
                "mean_interval_width": (
                    float(counters.width_sum) / counters.width_count
                    if counters.width_count
                    else None
                ),
            }
            match self._tier:
                case "cellwise" if n:
                    hits = self._element_hits[key]
                    at_target = hits >= _minimum_hits_at_target(n, self._alpha)
                    entry.update(
                        mean_element_coverage=float(hits.to(torch.float64).mean()) / n,
                        minimum_element_coverage=float(hits.min()) / n,
                        fraction_at_target=float(at_target.to(torch.float64).mean()),
                    )
                case "cellwise":
                    entry.update(
                        mean_element_coverage=None,
                        minimum_element_coverage=None,
                        fraction_at_target=None,
                    )
            fields["tensor" if key == TENSOR_KEY else key] = entry
        return {"meta": metadata, "fields": fields}

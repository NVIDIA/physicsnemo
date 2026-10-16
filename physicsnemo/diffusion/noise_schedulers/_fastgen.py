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

r"""Private translation of Gaussian training schedules to FastGen."""

import importlib
import math
from collections.abc import Callable
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from physicsnemo.core import Module
from physicsnemo.core.version_check import check_version_spec

from .domain_parallel import DomainParallelNoiseScheduler
from .edm import EDMNoiseScheduler
from .iddpm import IDDPMNoiseScheduler
from .linear_gaussian import LinearGaussianNoiseScheduler
from .rectified_flow import RectifiedFlowNoiseScheduler
from .student_t_edm import StudentTEDMNoiseScheduler
from .trigflow import TrigFlowNoiseScheduler
from .ve import VENoiseScheduler
from .vp import VPNoiseScheduler

_AVAILABLE = check_version_spec("fastgen", "0.1.0", hard_fail=False)
if _AVAILABLE:
    _backend = importlib.import_module("fastgen.networks.noise_schedule")
    _BaseNoiseSchedule = _backend.BaseNoiseSchedule
else:
    _BaseNoiseSchedule = Module


class _FastGenNoiseScheduler(_BaseNoiseSchedule):
    r"""Expose PhysicsNeMo noise coefficients and training times to FastGen.

    Default sampling uses the source scheduler's distribution. Select a
    bounded distribution to intersect requested limits with the source's
    training bounds. Coefficient evaluation uses the current source settings
    and remains available outside sampling bounds, including at zero.
    """

    def __init__(
        self, source: LinearGaussianNoiseScheduler | DomainParallelNoiseScheduler
    ) -> None:
        if not _AVAILABLE:
            raise ImportError(
                "PhysicsNeMo distillation requires the pinned FastGen backend. "
                "Install nvidia-physicsnemo[diffusion-fastgen]."
            )
        torch.nn.Module.__init__(self)
        self._sampling_source = source
        self._domain = isinstance(source, DomainParallelNoiseScheduler)
        self._scheduler = source.inner_scheduler if self._domain else source
        if not isinstance(self._scheduler, LinearGaussianNoiseScheduler):
            raise TypeError("Expected a LinearGaussianNoiseScheduler.")
        if isinstance(self._scheduler, StudentTEDMNoiseScheduler):
            raise TypeError("FastGen training requires Gaussian, not Student-t, noise.")
        self.t_precision = torch.float64
        self.clamp_min = 1e-6
        self._time_bounds = None
        self._sample_bounds = None
        self.register_buffer("_training_times", None, persistent=False)
        self._supported_time_dist_types = (
            "physicsnemo",
            "uniform",
            "lognormal",
            "logitnormal",
            "log_t",
        )

    def _configure_training(
        self,
        *,
        time_bounds: tuple[float, float]
        | Callable[[], tuple[float, float]]
        | None = None,
        sample_bounds: tuple[float, float]
        | Callable[[], tuple[float, float]]
        | None = None,
        training_times: Float[Tensor, " steps"] | None = None,
        time_precision: torch.dtype = torch.float64,
    ) -> None:
        r"""Supply training metadata that a custom source need not define."""
        if time_precision not in (torch.float32, torch.float64):
            raise ValueError("Training time precision must be float32 or float64.")
        if training_times is not None:
            if (
                training_times.ndim != 1
                or training_times.numel() < 2
                or not torch.isfinite(training_times).all()
                or training_times[-1] != 0
                or not torch.all(training_times[:-1] > training_times[1:])
            ):
                raise ValueError(
                    "training_times must decrease strictly and end at zero."
                )
            training_times = training_times.detach().clone().to(time_precision)
        self._time_bounds = time_bounds
        self._sample_bounds = sample_bounds
        self._training_times = training_times
        self.t_precision = time_precision

    def _bounds(self) -> tuple[float, float]:
        r"""Read live native-time bounds, requiring explicit custom metadata."""
        if self._time_bounds is not None:
            bounds = (
                self._time_bounds()
                if callable(self._time_bounds)
                else self._time_bounds
            )
        else:
            bounds = self._source_bounds()
        if bounds is None:
            raise ValueError("Custom scheduler requires explicit training time_bounds.")
        return self._check_bounds(bounds)

    def _source_bounds(self) -> tuple[float, float] | None:
        r"""Translate known source limits to native time, without inventing limits."""
        source = self._scheduler
        if isinstance(source, (EDMNoiseScheduler, IDDPMNoiseScheduler)):
            return source.sigma_min, source.sigma_max
        if isinstance(source, VENoiseScheduler):
            return source.sigma_min**2, source.sigma_max**2
        if isinstance(source, VPNoiseScheduler):
            return source.epsilon_s, source.t_max
        if isinstance(source, (RectifiedFlowNoiseScheduler, TrigFlowNoiseScheduler)):
            return source.t_min, source.t_max
        return None

    @staticmethod
    def _check_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
        r"""Require a finite ordered training interval."""
        lo, hi = bounds
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError(
                f"Training bounds must be finite and ordered, got {bounds}."
            )
        return lo, hi

    @property
    def min_t(self) -> float:
        r"""Read the current lower operational time."""
        return self._bounds()[0]

    @property
    def max_t(self) -> float:
        r"""Read the current upper operational time."""
        return self._bounds()[1]

    @property
    def max_sigma(self) -> float:
        r"""Read the noise scale at the upper operational time."""
        if self._time_bounds is None and isinstance(
            self._scheduler, (EDMNoiseScheduler, VENoiseScheduler, IDDPMNoiseScheduler)
        ):
            return self._scheduler.sigma_max
        return float(self.sigma(torch.tensor(self.max_t, dtype=self.t_precision)))

    @property
    def num_steps(self) -> int:
        r"""Return the number of explicitly configured training intervals."""
        if self._training_times is None:
            raise ValueError("num_steps requires explicit training_times.")
        return self._training_times.numel() - 1

    def alpha(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Delegate signal coefficients to the source."""
        return self._scheduler.alpha(t)

    _alpha = alpha

    def sigma(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Delegate noise coefficients to the source."""
        return self._scheduler.sigma(t)

    _sigma = sigma

    def alpha_prime(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Delegate signal derivatives to the source."""
        return self._scheduler.alpha_dot(t)

    _alpha_prime = alpha_prime

    def sigma_prime(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Delegate noise derivatives to the source."""
        return self._scheduler.sigma_dot(t)

    _sigma_prime = sigma_prime

    def rescale_t(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Preserve native time; model preconditioning belongs to PhysicsNeMo."""
        return t

    _rescale_t = rescale_t

    def sqrt_snr(self, t: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Use the source amplitude ratio without endpoint regularization."""
        return self._scheduler.snr(t)

    def sqrt_snr_to_t(self, q: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        r"""Delegate differentiable amplitude inversion to the source."""
        return self._scheduler.snr_inv(q)

    def is_t_valid(self, t: Float[Tensor, " *shape"]) -> Bool[Tensor, ""]:
        r"""Check the configured operational interval."""
        return torch.all((t >= self.min_t) & (t <= self.max_t))

    def forward_process(
        self,
        x: Float[Tensor, " B *dims"],
        eps: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""Reuse the supplied Gaussian draw: :math:`x_t=\alpha x+\sigma\epsilon`."""
        t_bc = t.to(torch.float64).reshape(-1, *([1] * (x.ndim - 1)))  # (B, 1, ...)
        return (self.alpha(t_bc) * x.double() + self.sigma(t_bc) * eps.double()).to(
            x.dtype
        )

    def cond_velocity(
        self,
        x: Float[Tensor, " B *dims"],
        eps: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""Evaluate :math:`\dot\alpha x+\dot\sigma\epsilon`, including zero time."""
        t_bc = t.to(torch.float64).reshape(-1, *([1] * (x.ndim - 1)))  # (B, 1, ...)
        return (
            self.alpha_prime(t_bc) * x.double() + self.sigma_prime(t_bc) * eps.double()
        ).to(x.dtype)

    def latents(
        self,
        noise: Float[Tensor, " B *dims"],
        t_init: Float[Tensor, " *shape"] | None = None,
    ) -> Float[Tensor, " B *dims"]:
        r"""Scale supplied initial noise without consulting an inference grid."""
        if t_init is None:
            t_init = torch.tensor(
                self.max_t, dtype=self.t_precision, device=noise.device
            )
        t_bc = t_init.double().reshape(
            -1, *([1] * (noise.ndim - 1))
        )  # (B or 1, 1, ...)
        return (noise.double() * self.sigma(t_bc)).to(noise.dtype)

    def x0_to_eps(
        self,
        xt: Float[Tensor, " B *dims"],
        x0: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""Reorder arguments for the source epsilon conversion."""
        return self._scheduler.x0_to_epsilon(x0, xt, t).to(xt.dtype)

    def eps_to_x0(
        self,
        xt: Float[Tensor, " B *dims"],
        eps: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""Reorder arguments for the source clean-data conversion."""
        return self._scheduler.epsilon_to_x0(eps, xt, t).to(xt.dtype)

    def x0_to_flow(
        self,
        xt: Float[Tensor, " B *dims"],
        x0: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""Convert clean data to instantaneous native-time velocity."""
        return self._scheduler.x0_to_flow(x0, xt, t).to(xt.dtype)

    def flow_to_x0(
        self,
        xt: Float[Tensor, " B *dims"],
        v: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
    ) -> Float[Tensor, " B *dims"]:
        r"""Convert instantaneous velocity to clean data."""
        return self._scheduler.flow_to_x0(v, xt, t).to(xt.dtype)

    def convert_model_output(
        self,
        xt: Float[Tensor, " B *dims"],
        model_output: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        src_pred_type: str = "x0",
        target_pred_type: str = "eps",
    ) -> Float[Tensor, " B *dims"]:
        r"""Reuse backend dispatch without assuming diffusion-v normalization."""
        if "v" in (src_pred_type, target_pred_type):
            raise NotImplementedError(
                "Diffusion-v conversion is not supported; use x0, eps, or flow."
            )
        return super().convert_model_output(
            xt, model_output, t, src_pred_type, target_pred_type
        )

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "physicsnemo",
        min_t: float | None = None,
        max_t: float | None = None,
        device: torch.device | None = None,
        **sample_kwargs: Any,
    ) -> Float[Tensor, " N"]:
        r"""Delegate source sampling or explicitly request a bounded backend policy."""
        for name in (
            "flow_matching_ratio",
            "consistency_ratio",
            "deterministic_buckets",
            "enabled",
        ):
            sample_kwargs.pop(name, None)
        if time_dist_type == "physicsnemo":
            if min_t is not None or max_t is not None or sample_kwargs:
                raise ValueError(
                    "Source sampling does not accept backend bounds or distribution options."
                )
            return self._sampling_source.sample_time(
                n, device=device, dtype=self.t_precision
            )
        bounds = self._sample_bounds
        lo, hi = (
            self._bounds()
            if bounds is None
            else bounds()
            if callable(bounds)
            else bounds
        )
        lo, hi = self._check_bounds((lo, hi))
        source_bounds = self._source_bounds()
        if source_bounds is not None:
            source_lo, source_hi = self._check_bounds(source_bounds)
            lo, hi = max(lo, source_lo), min(hi, source_hi)
        lo = max(lo, min_t) if min_t is not None else lo
        hi = min(hi, max_t) if max_t is not None else hi
        if lo > hi:
            raise ValueError(
                f"Requested sampling bounds have no intersection: [{lo}, {hi}]."
            )
        if time_dist_type not in self._supported_time_dist_types:
            raise ValueError(
                f"Unsupported distribution {time_dist_type!r}; expected {self._supported_time_dist_types}."
            )
        if lo == hi:
            t = torch.full((n,), lo, device=device, dtype=self.t_precision)
        elif time_dist_type == "uniform":
            if sample_kwargs:
                raise ValueError(
                    f"Uniform sampling has no distribution options, got {tuple(sample_kwargs)}."
                )
            t = lo + (hi - lo) * torch.rand(n, device=device, dtype=self.t_precision)
        else:
            mean = sample_kwargs.pop(
                "train_p_mean", 0.0 if time_dist_type == "logitnormal" else -1.2
            )
            std = sample_kwargs.pop(
                "train_p_std", 1.0 if time_dist_type == "logitnormal" else 1.2
            )
            df = sample_kwargs.pop("log_t_df", 2.0)
            if sample_kwargs:
                raise ValueError(f"Unknown sampling options: {tuple(sample_kwargs)}.")
            if std <= 0:
                raise ValueError(
                    f"Sampling standard deviation must be positive, got {std}."
                )
            if time_dist_type == "logitnormal":
                t = (
                    lo
                    + (hi - lo)
                    * (
                        torch.randn(n, device=device, dtype=self.t_precision) * std
                        + mean
                    ).sigmoid()
                )
            elif time_dist_type == "lognormal":
                t = self._truncated_lognormal_sample(
                    n, mean, std, lo, hi, device=device
                )
            else:
                t = self._truncated_log_t_sample(
                    n, mean, std, lo, hi, df=df, device=device
                )
        if self._domain:
            t = self._sampling_source._broadcast_time(t)
        return t

    def _truncated_lognormal_sample(
        self, *args: Any, **kwargs: Any
    ) -> Float[Tensor, " N"]:
        r"""Reuse FastGen's truncated draw, without clipping unbounded samples."""
        return _backend.EDMNoiseSchedule._truncated_lognormal_sample(
            self, *args, **kwargs
        )

    def _truncated_log_t_sample(self, *args: Any, **kwargs: Any) -> Float[Tensor, " N"]:
        r"""Reuse FastGen's bounded log-Student-t time distribution."""
        return _backend.EDMNoiseSchedule._truncated_log_t_sample(self, *args, **kwargs)

    def get_t_list(
        self, sample_steps: int, device: torch.device | None = None
    ) -> Float[Tensor, " steps"]:
        r"""Read an explicit training rollout, or the single interval to clean data."""
        if self._training_times is None:
            if sample_steps != 1:
                raise ValueError("Multistep training requires explicit training_times.")
            return torch.tensor([self.max_t, 0], device=device, dtype=self.t_precision)
        if sample_steps != self.num_steps:
            raise ValueError(
                f"Expected {self.num_steps} training intervals, got {sample_steps}."
            )
        return self._training_times.to(device=device, dtype=self.t_precision)

    def sample_from_t_list(
        self,
        n: int,
        sample_steps: int,
        t_list: list[float] | Float[Tensor, " steps"] | None = None,
        return_ids: bool = False,
        device: torch.device | None = None,
    ) -> Float[Tensor, " N"] | tuple[Float[Tensor, " N"], Int[Tensor, " N"]]:
        r"""Choose training intervals without requiring quantization tables."""
        times = (
            self.get_t_list(sample_steps, device)
            if t_list is None
            else torch.as_tensor(t_list, device=device, dtype=self.t_precision)
        )
        ids = torch.randint(times.numel() - 1, (n,), device=device)
        if self._domain:
            ids = self._sampling_source._broadcast_time(ids)
        return (times[ids], ids) if return_ids else times[ids]

    @property
    def sigmas(self) -> Float[Tensor, " steps"]:
        r"""Reserve the quantization interface without building a discrete table."""
        raise NotImplementedError(
            "Quantization is not supported by the PhysicsNeMo FastGen adapter."
        )

    def closest_sigma_idx(
        self, sigma_t: Float[Tensor, " *shape"]
    ) -> Int[Tensor, " *shape"]:
        r"""Reserve nearest-noise lookup for future quantization support."""
        raise NotImplementedError(
            "Quantization is not supported by the PhysicsNeMo FastGen adapter."
        )

    def sigma_idx_to_t(
        self, sigma_idx: Int[Tensor, " *shape"]
    ) -> Float[Tensor, " *shape"]:
        r"""Reserve discrete-index conversion for future quantization support."""
        raise NotImplementedError(
            "Quantization is not supported by the PhysicsNeMo FastGen adapter."
        )

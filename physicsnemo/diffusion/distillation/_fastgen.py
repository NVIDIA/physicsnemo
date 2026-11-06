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

r"""Connect PhysicsNeMo models to FastGen distillation objectives."""

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

import torch
from jaxtyping import Float, Shaped
from tensordict import TensorDict
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from physicsnemo.diffusion._fastgen import _FastGenNetwork
from physicsnemo.diffusion.base import (
    DiffusionExtraTimes,
)
from physicsnemo.diffusion.noise_schedulers import (
    DomainParallelNoiseScheduler,
    LinearGaussianNoiseScheduler,
)
from physicsnemo.diffusion.noise_schedulers._fastgen import _FastGenNoiseScheduler


class _ObjectiveReduction:
    r"""Compute masked statistics and weighted squared residuals.

    Statistics use :math:`m=1_{w>0}` and count :math:`\sum m`. Residuals
    use :math:`w` with the full element count for means. Positive weights
    scale residuals linearly and leave the adaptive target unchanged.
    """

    def __init__(
        self,
        reference: Float[Tensor, " B *dims"],
        weight: Float[Tensor, " #B *#dims"] | None,
    ) -> None:
        self.weight = (
            torch.ones_like(reference)
            if weight is None
            else torch.broadcast_to(weight, reference.shape)
        )
        self.valid = self.weight > 0
        self.dims = tuple(range(1, reference.ndim))
        self.count = self.valid.sum(self.dims)
        self.active = self.count > 0

    def clean(self, value: Float[Tensor, " B *dims"]) -> Float[Tensor, " B *dims"]:
        r"""Remove invalid entries before nonlinear operations."""
        return torch.where(self.valid, value, torch.zeros_like(value))

    def norm(self, value: Float[Tensor, " B *dims"]) -> Float[Tensor, " B"]:
        r"""Compute the Euclidean norm on valid entries only."""
        return torch.linalg.vector_norm(self.clean(value).flatten(1), dim=1)

    def residual(
        self, value: Float[Tensor, " B *dims"], *, mean: bool
    ) -> Float[Tensor, " B"]:
        r"""Weight squared residuals linearly, retaining the backend denominator."""
        weighted = self.clean(value).square() * self.weight
        return weighted.mean(self.dims) if mean else weighted.sum(self.dims)


class _FastGenBinding:
    r"""Connect student and teacher models and collect batch loss outputs."""

    def _bind_execution_models(self, models: Mapping[str, torch.nn.Module]) -> None:
        r"""Rebind wrappers only when they keep the configured role parameters."""
        for role, attribute in (
            ("student", "net"),
            ("teacher", "teacher"),
            ("fake_score", "fake_score"),
            ("discriminator", "discriminator"),
        ):
            adapter = getattr(self, attribute, None)
            if adapter is None:
                continue
            if role not in models:
                raise ValueError(f"Missing configured loss role {role!r}.")
            replacement = models[role]
            if {id(value) for value in adapter.model.parameters()} != {
                id(value) for value in replacement.parameters()
            }:
                raise ValueError(
                    f"Execution wrapper for {role!r} must preserve parameter identities; wrap before constructing the loss otherwise."
                )
            adapter.model = replacement

    def _jvp(self, *args: Any, **kwargs: Any) -> Float[Tensor, " B *dims"]:
        r"""Use differentiable attention kernels for backend tangent calculations."""
        with sdpa_kernel(SDPBackend.MATH):
            return super()._jvp(*args, **kwargs)

    def _bind_models(
        self,
        student: torch.nn.Module,
        noise_scheduler: LinearGaussianNoiseScheduler | DomainParallelNoiseScheduler,
        teacher: torch.nn.Module | None,
        model_options: Mapping[str, Mapping[str, Any]],
        *,
        average_flow: bool = False,
    ) -> None:
        r"""Attach supplied role modules, preserving compilation and freeze masks."""
        unknown = set(model_options) - {"student", "teacher"}
        if unknown:
            raise ValueError(f"Unknown model roles: {sorted(unknown)}.")
        for role, model in (("student", student), ("teacher", teacher)):
            if model is None:
                self.teacher = None
                continue
            options = dict(model_options.get(role, {}))
            source = options.pop("noise_scheduler", noise_scheduler)
            schedule = _FastGenNoiseScheduler(source)
            schedule._configure_training(**options.pop("scheduler_options", {}))
            if average_flow and role == "student":
                options.setdefault("prediction_type", "flow")
                options.setdefault("prediction_semantics", "average")
            else:
                options.setdefault("prediction_type", "x0")
            network = _FastGenNetwork(model, schedule, **options)
            if (
                average_flow
                and role == "teacher"
                and schedule._scheduler is not self.net.noise_scheduler._scheduler
            ):
                network._input_noise_scheduler = self.net.noise_scheduler
            setattr(self, "net" if role == "student" else "teacher", network)
        self.precision_amp = None
        self.precision_amp_jvp = None
        self._call_weight = None

    def evaluate(
        self,
        x0: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"]
        | tuple[Float[Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]]
        | None,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None,
        *,
        phase: str | None,
        step: int,
        return_outputs: bool,
        update_state: bool,
        weight: Float[Tensor, " #B *#dims"] | None = None,
        neg_condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        sample_t_fn: Callable[
            ...,
            Float[Tensor, " B"]
            | tuple[Float[Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]],
        ]
        | None = None,
    ) -> (
        tuple[dict[str, Float[Tensor, " *shape"]], dict[str, Float[Tensor, " *shape"]]]
        | dict[str, Float[Tensor, " *shape"]]
    ):
        r"""Validate one explicit phase and release all graph-bearing call state."""
        if phase not in (None, "student"):
            raise ValueError(f"This method has only the student phase, got {phase!r}.")
        if step < 0:
            raise ValueError(f"step must be nonnegative, got {step}.")
        if t is not None and sample_t_fn is not None:
            raise ValueError("Supply explicit times or sample_t_fn, not both.")
        if not torch.compiler.is_compiling():
            if x0.ndim < 2 or x0.shape[0] == 0:
                raise ValueError("Expected nonempty states with shape [B, ...].")
            if weight is not None and (
                not torch.isfinite(weight).all() or torch.any(weight < 0)
            ):
                raise ValueError("Weights must be finite and nonnegative.")
        if weight is not None:
            weight = torch.broadcast_to(weight.to(device=x0.device), x0.shape)
        self.device = x0.device
        if sample_t_fn is not None:
            t = sample_t_fn(
                x0.shape[0],
                device=x0.device,
                dtype=self.net.noise_scheduler.t_precision,
            )
        previous_weight = self._call_weight
        self._call_weight = weight
        try:
            losses, outputs = self.single_train_step(
                {
                    "real": x0,
                    "t": t,
                    "condition": condition,
                    "neg_condition": neg_condition,
                    "return_outputs": return_outputs,
                },
                step,
            )
        finally:
            self._call_weight = previous_weight
        losses = {
            key: value if key == "total_loss" else value.detach()
            for key, value in losses.items()
        }
        return (
            (losses, {key: value.detach() for key, value in outputs.items()})
            if return_outputs
            else losses
        )

    def loss_state_dict(self) -> dict[str, Any]:
        r"""These objectives have no running loss statistics."""
        return {}

    def load_loss_state_dict(self, state: Mapping[str, Any]) -> None:
        r"""Reject unrelated model or optimizer state at the loss boundary."""
        if state:
            raise ValueError(f"Expected empty loss state, got keys {tuple(state)}.")


def _configure_options(
    defaults: Mapping[str, Any], options: Mapping[str, Any]
) -> SimpleNamespace:
    r"""Reject misspelled controls before invoking backend computation."""
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown method options {sorted(unknown)}; expected {tuple(defaults)}."
        )
    return SimpleNamespace(**(dict(defaults) | dict(options)))

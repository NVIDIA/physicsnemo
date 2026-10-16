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

r"""Adapt PhysicsNeMo model calls and predictions for FastGen losses."""

import importlib
from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from physicsnemo.core import Module
from physicsnemo.core.version_check import check_version_spec

from .base import (
    DiffusionExtraOutputs,
    DiffusionExtraTimes,
)
from .noise_schedulers._fastgen import _FastGenNoiseScheduler

_AVAILABLE = check_version_spec("fastgen", "0.1.0", hard_fail=False)
if _AVAILABLE:
    _FastGenNetworkBase = importlib.import_module(
        "fastgen.networks.network"
    ).FastGenNetwork
else:
    _FastGenNetworkBase = Module


class _FastGenNetwork(_FastGenNetworkBase):
    r"""Adapt model time inputs and output fields to FastGen conventions.

    Wrap an initialized PhysicsNeMo model, preserving its parameters and
    distributed configuration. Save the original model in checkpoints and
    reconstruct this adapter when resuming training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        noise_scheduler: _FastGenNoiseScheduler,
        *,
        prediction_type: Literal["x0", "epsilon", "score", "flow"],
        prediction_semantics: Literal["instantaneous", "average"] = "instantaneous",
        model_kwargs: Mapping[str, Any] | None = None,
        feature_names: tuple[str, ...] = (),
        feature_fn: Callable[
            [Float[Tensor, " B *dims"], DiffusionExtraOutputs | None], TensorDict
        ]
        | None = None,
    ) -> None:
        if not _AVAILABLE:
            raise ImportError(
                "PhysicsNeMo distillation requires the pinned FastGen backend. "
                "Install nvidia-physicsnemo[diffusion-fastgen]."
            )
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "Training requires a parameter-owning module, not a bare callable."
            )
        if prediction_type not in ("x0", "epsilon", "score", "flow"):
            raise ValueError(
                f"Expected x0, epsilon, score, or flow, got {prediction_type!r}."
            )
        if prediction_semantics not in ("instantaneous", "average"):
            raise ValueError(
                f"Expected instantaneous or average semantics, got {prediction_semantics!r}."
            )
        if prediction_semantics == "average" and prediction_type != "flow":
            raise ValueError("Average flow requires flow prediction.")
        torch.nn.Module.__init__(self)
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.net_pred_type = (
            "eps" if prediction_type in ("epsilon", "score") else prediction_type
        )
        self.schedule_type = "physicsnemo"
        self._prediction_type = prediction_type
        self._prediction_semantics = prediction_semantics
        self._model_kwargs = dict(model_kwargs or {})
        self._input_noise_scheduler = None
        self._feature_names = tuple(feature_names)
        self._feature_fn = feature_fn

    def set_noise_schedule(
        self, schedule_type: str | None = None, **noise_schedule_kwargs: Any
    ) -> None:
        r"""Preserve the configured source across FastGen's reset lifecycle."""
        if schedule_type is not None or noise_schedule_kwargs:
            raise ValueError(
                "Configure the PhysicsNeMo noise scheduler before creating this bridge."
            )

    def fully_shard(self, **kwargs: Any) -> None:
        r"""Keep distributed wrapping under the caller's PhysicsNeMo runtime."""
        raise NotImplementedError(
            "Shard the PhysicsNeMo model before constructing the distillation loss."
        )

    def forward(
        self,
        x_t: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        r: Float[Tensor, " B"] | None = None,
        return_features_early: bool = False,
        feature_indices: set[int] | None = None,
        return_logvar: bool = False,
        fwd_pred_type: str | None = None,
        **fwd_kwargs: Any,
    ) -> (
        Float[Tensor, " B *dims"]
        | list[Float[Tensor, " B *feature_dims"]]
        | tuple[Float[Tensor, " B *dims"], Float[Tensor, "B 1"]]
        | tuple[Float[Tensor, " B *dims"], list[Float[Tensor, " B *feature_dims"]]]
    ):
        r"""Evaluate once, extract requested fields, and convert only predictions."""
        target = self.net_pred_type if fwd_pred_type is None else fwd_pred_type
        if target not in ("x0", "eps", "flow"):
            raise ValueError(f"Expected x0, eps, or flow prediction, got {target!r}.")
        if self._prediction_semantics == "average" and target != "flow":
            raise ValueError(
                "An average flow cannot use instantaneous prediction conversions."
            )
        input_x, input_t = x_t, t
        source = self._input_noise_scheduler
        if source is not None:
            if r is not None:
                raise ValueError(
                    "Two-time predictions cannot use instantaneous coordinate mapping."
                )
            native_t = self.noise_scheduler.sqrt_snr_to_t(source.sqrt_snr(t.double()))
            source_scale = torch.hypot(
                source.alpha(t.double()), source.sigma(t.double())
            )
            target_scale = torch.hypot(
                self.noise_scheduler.alpha(native_t),
                self.noise_scheduler.sigma(native_t),
            )
            scale = target_scale / source_scale
            if not torch.compiler.is_compiling() and not (
                torch.isfinite(native_t).all() and torch.isfinite(scale).all()
            ):
                raise ValueError(
                    "Teacher coordinate conversion needs finite native times and state scales."
                )
            scale = scale.reshape(-1, *([1] * (x_t.ndim - 1)))  # (B, 1, ...)
            x_t, t = (x_t.double() * scale).to(x_t.dtype), native_t.to(t.dtype)
        time = (
            t if r is None else (t, DiffusionExtraTimes(r=r, batch_size=[x_t.shape[0]]))
        )
        output = self.model(
            x_t, time, condition=condition, **(self._model_kwargs | fwd_kwargs)
        )
        prediction, extras = output if isinstance(output, tuple) else (output, None)
        if not isinstance(prediction, Tensor):
            raise TypeError("A diffusion model must return a prediction tensor.")
        features = None
        if return_features_early or feature_indices:
            if self._feature_fn is None:
                raise ValueError(
                    "Adversarial feature requests require an explicit feature_fn."
                )
            named_features = self._feature_fn(prediction, extras)
            indices = (
                sorted(feature_indices)
                if feature_indices
                else range(len(self._feature_names))
            )
            features = [named_features[self._feature_names[index]] for index in indices]
            if return_features_early:
                return features
        if self._prediction_type == "score":
            prediction = self.noise_scheduler._scheduler.score_to_epsilon(prediction, t)
        prediction = self.noise_scheduler.convert_model_output(
            x_t,
            prediction,
            t,
            self.net_pred_type,
            "x0" if source is not None else target,
        )
        if source is not None:
            prediction = source.convert_model_output(
                input_x, prediction, input_t, "x0", target
            )
        if return_logvar:
            if not isinstance(extras, DiffusionExtraOutputs):
                raise ValueError(
                    "This method requires (prediction, DiffusionExtraOutputs(logvar=...))."
                )
            logvar = extras.logvar
            if logvar is None:
                raise ValueError("This method requires a logvar tensor; got None.")
            if logvar.shape != (x_t.shape[0],):
                raise ValueError(
                    f"Expected logvar shape {(x_t.shape[0],)}, got {logvar.shape}."
                )
            return prediction, logvar[:, None]  # (B, 1)
        return (prediction, features) if features is not None else prediction

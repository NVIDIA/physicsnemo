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

r"""Compute sCM objectives with optional masks and element weights."""

import importlib
import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.core import Module
from physicsnemo.core.version_check import check_version_spec
from physicsnemo.diffusion.noise_schedulers import (
    DomainParallelNoiseScheduler,
    LinearGaussianNoiseScheduler,
    TrigFlowNoiseScheduler,
)
from physicsnemo.diffusion.noise_schedulers._fastgen import _FastGenNoiseScheduler

from ..._fastgen import _configure_options, _FastGenBinding

_AVAILABLE = check_version_spec("fastgen", "0.1.0", hard_fail=False)
if _AVAILABLE:
    _backend = importlib.import_module("fastgen.methods.consistency_model.sCM")
    _SCMBase = _backend.SCMModel
else:
    _SCMBase = Module


class _SCMLoss(_FastGenBinding, _SCMBase):
    r"""Compute sCM targets and adaptive losses for initialized models."""

    def __init__(
        self,
        student: torch.nn.Module,
        noise_scheduler: LinearGaussianNoiseScheduler | DomainParallelNoiseScheduler,
        *,
        teacher: torch.nn.Module | None,
        method_options: Mapping[str, Any],
        model_options: Mapping[str, Mapping[str, Any]],
    ) -> None:
        torch.nn.Module.__init__(self)
        self._bind_models(student, noise_scheduler, teacher, model_options)
        options = _configure_options(
            {
                "sigma_data": 0.5,
                "tangent_warmup_steps": 10000,
                "tangent_warmup_const": 0.1,
                "prior_weighting_enabled": True,
                "g_norm_spatial_invariance": True,
                "divide_x_0_spatial_dim": True,
                "use_jvp_finite_diff": False,
                "jvp_finite_diff_eps": 1e-3,
                "guidance_scale": None,
                "sampling": {},
            },
            method_options,
        )
        if (
            options.sigma_data <= 0
            or options.tangent_warmup_steps <= 0
            or options.tangent_warmup_const <= 0
        ):
            raise ValueError(
                "sCM requires positive sigma_data, tangent_warmup_steps, and tangent_warmup_const."
            )
        self.sigma_data = options.sigma_data
        self.loss_config = options
        self.loss_config.use_cd = teacher is not None
        self.config = SimpleNamespace(
            guidance_scale=options.guidance_scale, skip_layers=None
        )
        self.sample_t_cfg = SimpleNamespace(quantize=False)
        self._sampling_options = dict(options.sampling)
        self.net_trigflow = _backend.TrigFlowPrecond(
            self.net, sigma_data=self.sigma_data
        )
        self.net_trigflow.noise_scheduler = _FastGenNoiseScheduler(
            TrigFlowNoiseScheduler(sigma_data=1, t_min=0, t_max=math.pi / 2 - 1e-5)
        )
        self.teacher_trigflow = (
            None
            if self.teacher is None
            else _backend.TrigFlowPrecond(self.teacher, sigma_data=self.sigma_data)
        )
        if self.teacher_trigflow is not None:
            self.teacher_trigflow.noise_scheduler = self.net_trigflow.noise_scheduler

    def _scm_pred_to_loss(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Float[Tensor, " B"], Float[Tensor, " B"], float]:
        r"""Compute the weighted consistency loss with tangent norms over valid entries."""
        return super()._scm_pred_to_loss(*args, loss_weight=self._call_weight, **kwargs)

    def single_train_step(
        self, data: Mapping[str, Any], iteration: int
    ) -> tuple[
        dict[str, Float[Tensor, " *shape"]], dict[str, Float[Tensor, " *shape"]]
    ]:
        r"""Honor explicit native times and avoid output-only inference closures."""
        real = data["real"]
        t = data["t"]
        if isinstance(t, tuple):
            raise ValueError("sCM requires a single diffusion time tensor.")
        if t is None:
            t = self.net.noise_scheduler.sample_t(
                real.shape[0], device=real.device, **self._sampling_options
            )
        loss, residual, logvar, prediction, warmup = self._compute_scm_loss(
            real, t, data["condition"], data["neg_condition"], iteration
        )
        losses = {
            "total_loss": loss.mean(),
            "scm_loss": loss.mean(),
            "unweighted_scm_loss": residual.mean(),
            "logvar_loss": logvar.mean(),
            "warmup_weight": real.new_tensor(warmup),
        }
        outputs = {"prediction": prediction, "t": t} if data["return_outputs"] else {}
        return losses, outputs

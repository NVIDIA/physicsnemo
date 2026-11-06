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

r"""Train MeanFlow students to predict velocity averaged over a time interval."""

import importlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from jaxtyping import Float
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from physicsnemo.core import Module
from physicsnemo.core.version_check import check_version_spec
from physicsnemo.diffusion.noise_schedulers import (
    DomainParallelNoiseScheduler,
    LinearGaussianNoiseScheduler,
)

from ..._fastgen import _configure_options, _FastGenBinding, _ObjectiveReduction

_AVAILABLE = check_version_spec("fastgen", "0.1.0", hard_fail=False)
if _AVAILABLE:
    _backend = importlib.import_module("fastgen.methods.consistency_model.mean_flow")
    _MeanFlowBase = _backend.MeanFlowModel
else:
    _MeanFlowBase = Module


class _MeanFlowLoss(_FastGenBinding, _MeanFlowBase):
    r"""Compute MeanFlow targets and adaptive losses for initialized models."""

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
        self._bind_models(
            student, noise_scheduler, teacher, model_options, average_flow=True
        )
        options = _configure_options(
            {
                "loss_type": "opt_grad",
                "norm_method": "poly_1.0",
                "norm_const": 0.1,
                "tangent_warmup_steps": 0,
                "tangent_spatial_invariance": False,
                "use_jvp_finite_diff": False,
                "jvp_finite_diff_eps": 1e-4,
                "flow_matching_ratio": 0.25,
                "consistency_ratio": 0.0,
                "guidance_scale": None,
                "guidance_time_bounds": None,
                "sampling": {},
            },
            method_options,
        )
        if options.loss_type not in ("l2", "opt_grad") or options.norm_const <= 0:
            raise ValueError(
                "MeanFlow requires loss_type='l2' or 'opt_grad' and norm_const > 0."
            )
        if (
            not 0 <= options.flow_matching_ratio <= 1
            or not 0 <= options.consistency_ratio <= 1 - options.flow_matching_ratio
        ):
            raise ValueError(
                "MeanFlow bucket fractions must be nonnegative and sum to at most one."
            )
        options.use_cd = teacher is not None
        options.weight_type = None
        options.rebalance_to_flow_matching = False
        guidance_bounds = options.guidance_time_bounds or (float("-inf"), float("inf"))
        if len(guidance_bounds) != 2 or not guidance_bounds[0] <= guidance_bounds[1]:
            raise ValueError(
                "guidance_time_bounds must be an ordered native-time pair."
            )
        self.config = SimpleNamespace(
            loss_config=options,
            precision_amp_jvp=None,
            guidance_scale=options.guidance_scale,
            guidance_mixture_ratio=None,
            guidance_fuse_scale=None,
            cond_dropout_prob=None,
            guidance_t_start=guidance_bounds[0],
            guidance_t_end=guidance_bounds[1],
        )
        sampling = {
            "time_dist_type": "physicsnemo",
            **dict(options.sampling),
            "flow_matching_ratio": options.flow_matching_ratio,
            "consistency_ratio": options.consistency_ratio,
            "deterministic_buckets": False,
        }
        self._init_flow_map_loss(
            OmegaConf.create(sampling), OmegaConf.create({"enabled": False})
        )

    def _get_velocity(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Any, Float[Tensor, " B *dims"], Tensor]:
        r"""Keep caller-selected child modes across backend guidance queries."""
        modes = tuple((module, module.training) for module in self.net.modules())
        try:
            return super()._get_velocity(*args, **kwargs)
        finally:
            for module, mode in modes:
                module.training = mode

    def _mf_pred_to_loss(
        self,
        u_theta: Float[Tensor, " B *dims"],
        u_theta_jvp: Float[Tensor, " B *dims"],
        x_t: Float[Tensor, " B *dims"],
        dxt_dt: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"],
        r: Float[Tensor, " B"],
        iteration: int,
    ) -> tuple[
        Float[Tensor, " B"], Float[Tensor, " B *dims"], Float[Tensor, " *shape"], float
    ]:
        r"""Exclude masked entries from adaptive norms before weighting residuals.

        .. math::

            h=v-\lambda(t-r)\partial_t u,\qquad
            L_{\mathrm{L2}}=\operatorname{reduce}(w(u-\operatorname{sg}(h))^2).

        The explicit-gradient variant also subtracts the stopped prediction
        from its tangent, retaining FastGen's adaptive weighting function.
        """
        u = u_theta.double()
        reduction = _ObjectiveReduction(u, self._call_weight)
        shape = (-1,) + (1,) * (u.ndim - 1)
        delta = (t.double() - r.double()).reshape(shape)  # (B, 1, ...)
        steps = self.loss_config.tangent_warmup_steps
        warmup = min(1.0, iteration / steps) if steps > 0 else 1.0
        tangent = dxt_dt.double() - warmup * delta * u_theta_jvp.double()
        if self.loss_config.loss_type == "l2":
            tangent = reduction.clean(tangent)
            residual = reduction.clean(u - tangent)
            statistic = residual.square().sum(reduction.dims)
            if self.loss_config.norm_method is None:
                statistic = statistic / reduction.count.clamp_min(1)
            adaptive = self._compute_weight(statistic, t)
            loss = (
                reduction.residual(residual, mean=self.loss_config.norm_method is None)
                * adaptive
            )
        else:
            tangent = reduction.clean(tangent - u.detach())
            if self.loss_config.tangent_spatial_invariance:
                tangent = tangent / reduction.count.clamp_min(1).sqrt().reshape(
                    shape
                )  # (B, 1, ...)
            adaptive = self._compute_weight(reduction.norm(tangent), t)
            target = (u + tangent * adaptive.reshape(shape)).detach()  # (B, 1, ...)
            loss = reduction.residual(u - target, mean=False)
        loss = torch.where(reduction.active, loss, torch.zeros_like(loss))
        return loss, tangent, adaptive, warmup

    def single_train_step(
        self, data: Mapping[str, Any], iteration: int
    ) -> tuple[
        dict[str, Float[Tensor, " *shape"]], dict[str, Float[Tensor, " *shape"]]
    ]:
        r"""Compute a flow-map loss without an instantaneous x0 conversion."""
        real = data["real"]
        time = data["t"]
        if time is None:
            t, r, buckets = self._sample_t_r_buckets(real.shape[0])
        elif isinstance(time, tuple):
            t, extra_times = time
            r = extra_times.r
            if r is None:
                raise ValueError(
                    "Explicit MeanFlow times require an r tensor; got None."
                )
            buckets = t == r
        else:
            raise ValueError(
                "Explicit MeanFlow times require (t, DiffusionExtraTimes(r=r))."
            )
        if not torch.compiler.is_compiling() and torch.any(r > t):
            raise ValueError("MeanFlow requires r <= t.")
        z = torch.randn_like(real)
        x_t = self.net.noise_scheduler.forward_process(real, z, t)
        condition, velocity, _ = self._get_velocity(
            real, z, t, condition=data["condition"], neg_condition=data["neg_condition"]
        )
        torch.clear_autocast_cache()
        if self.loss_config.use_jvp_finite_diff:
            jvp = self._jvp(x_t, t, r, velocity, condition=condition)
            prediction = self.net(
                x_t, t, r=r, condition=condition, fwd_pred_type="flow"
            )
        else:
            # Reuse the primal so stochastic layers share one draw with the JVP.
            # Only the primal contributes parameter gradients to the loss.
            with sdpa_kernel(SDPBackend.MATH):
                prediction, jvp = torch.func.jvp(
                    lambda x, time, start: self.net(
                        x, time, r=start, condition=condition, fwd_pred_type="flow"
                    ),
                    (x_t, t, r),
                    (velocity, torch.ones_like(t), torch.zeros_like(r)),
                )
            jvp = jvp.detach()
        loss, tangent, adaptive, warmup = self._mf_pred_to_loss(
            prediction, jvp, x_t, velocity, t, r, iteration
        )
        total = self._reduce_mf_loss(loss, buckets)
        losses = {
            "total_loss": total,
            "mf_loss": total,
            "loss_weight": adaptive.mean(),
            "tangent_warmup_weight": real.new_tensor(warmup),
        }
        outputs = (
            {"prediction": prediction, "x_t": x_t, "t": t, "r": r}
            if data["return_outputs"]
            else {}
        )
        return losses, outputs

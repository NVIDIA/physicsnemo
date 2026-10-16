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

r"""f-distill distribution matching with running density-ratio statistics."""

import importlib
from collections.abc import Mapping
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from physicsnemo.diffusion.noise_schedulers import DomainParallelNoiseScheduler

from ..._fastgen import _configure_options, _ObjectiveReduction
from ..dmd2._fastgen import _DMD2Loss


class _FDistillLoss(_DMD2Loss):
    r"""Weight student score matching by a selected f-divergence.

    Use a discriminator to estimate density ratios and running time-bin
    statistics to normalize them. Save these statistics with the training
    checkpoint, and select a data-replica group for domain-parallel training.
    """

    def __init__(
        self, *args: Any, method_options: Mapping[str, Any], **kwargs: Any
    ) -> None:
        options = dict(method_options)
        self._ratio_options = _configure_options(
            {
                "f_div": "js",
                "ratio_lower": 0.1,
                "ratio_upper": 20.0,
                "ratio_normalization": True,
                "bin_num": 10,
                "ratio_ema_rate": 0.0,
                "statistics_group": None,
            },
            options.pop("f_distill", {}),
        )
        super().__init__(*args, method_options=options, **kwargs)
        if self.config.gan_loss_weight_gen <= 0:
            raise ValueError(
                "f-distill requires a discriminator and positive gan_loss_weight_gen."
            )
        backend = importlib.import_module(
            "fastgen.methods.distribution_matching.f_distill"
        )
        ratio = self._ratio_options
        if ratio.f_div not in backend.all_f_div_weighting_function:
            raise ValueError(f"Unsupported f-divergence {ratio.f_div!r}.")
        if (
            not 0 < ratio.ratio_lower <= ratio.ratio_upper
            or ratio.bin_num < 1
            or not 0 <= ratio.ratio_ema_rate <= 1
        ):
            raise ValueError(
                "Require positive ratio bounds/bin count and an EMA rate in [0, 1]."
            )
        self.f_div_weighting_function = backend.all_f_div_weighting_function[
            ratio.f_div
        ]
        self._bins = torch.ones(ratio.bin_num)
        if (
            isinstance(
                self.net.noise_scheduler._sampling_source, DomainParallelNoiseScheduler
            )
            and ratio.statistics_group is None
        ):
            raise ValueError(
                "Domain-parallel f-distill requires an explicit data-parallel statistics_group."
            )

    def _sum_statistics(self, value: Tensor) -> Tensor:
        r"""Reduce over data replicas without mixing spatial partitions into counts."""
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                value, group=self._ratio_options.statistics_group
            )
        return value

    def _ratio_weight(
        self,
        logits: Float[Tensor, " B logits"],
        t: Float[Tensor, " B"],
        weight: Tensor | None,
    ) -> Float[Tensor, " B"]:
        r"""Normalize ratios and divergence weights over valid samples only."""
        with torch.no_grad():
            if weight is None:
                mean = logits.float().mean(1)
                active = torch.ones_like(t, dtype=torch.bool)
            else:
                valid = weight > 0
                count = valid.sum(1)
                mean = torch.where(valid, logits.float(), 0).sum(1) / count.clamp_min(1)
                active = count > 0
            if self._call_weight is not None:
                active = (
                    active
                    & _ObjectiveReduction(self._call_weight, self._call_weight).active
                )
            options = self._ratio_options
            ratio = (
                mean.clamp(-10, 10)
                .exp()
                .clamp(options.ratio_lower, options.ratio_upper)
            )
            if options.ratio_normalization:
                source = self.net.noise_scheduler
                lo, hi = source.min_t, source.max_t
                width = (hi - lo) / options.bin_num
                idx = (
                    ((t.double() - lo) / width)
                    .floor()
                    .long()
                    .clamp(0, options.bin_num - 1)
                )
                count = torch.bincount(
                    idx, weights=active.float(), minlength=options.bin_num
                )
                total = torch.bincount(
                    idx,
                    weights=torch.where(active, ratio, 0),
                    minlength=options.bin_num,
                )
                statistics = self._sum_statistics(torch.stack((count, total)))
                count, total = statistics.unbind(0)
                bins = self._bins.to(device=t.device, dtype=torch.float32)
                if self._update_state:
                    new_bins = bins * options.ratio_ema_rate + (
                        1 - options.ratio_ema_rate
                    ) * total / (count + 1e-6)
                    bins = torch.where(count > 0, new_bins, bins)
                    self._bins = bins.detach()
                ratio = ratio / (bins[idx] + 1e-6)
            h = self.f_div_weighting_function(ratio).expand_as(ratio)
            totals = self._sum_statistics(
                torch.stack((torch.where(active, h, 0).sum(), active.float().sum()))
            )
            h_mean = totals[0] / totals[1].clamp_min(1)
            return torch.where(active, h / (h_mean + 1e-6), 0).to(logits.dtype)

    def loss_state_dict(self) -> dict[str, Any]:
        r"""Return a copy of the running density-ratio bins for checkpointing."""
        return {"schema_version": 1, "bins": self._bins.detach().clone()}

    def load_loss_state_dict(self, state: Mapping[str, Any]) -> None:
        r"""Restore density-ratio bins using the configured bin count."""
        if set(state) != {"schema_version", "bins"} or state["schema_version"] != 1:
            raise ValueError("Expected f-distill state schema version 1 with bins.")
        bins = state["bins"]
        if (
            not isinstance(bins, Tensor)
            or bins.shape != self._bins.shape
            or not torch.isfinite(bins).all()
            or torch.any(bins <= 0)
        ):
            raise ValueError(
                "Expected finite positive ratio bins with the configured shape."
            )
        self._bins = bins.detach().clone()

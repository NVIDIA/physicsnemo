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
        self._loss_support = importlib.import_module("fastgen.methods.loss_support")
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

    def _ratio_weight(
        self,
        logits: Float[Tensor, " B logits"],
        t: Float[Tensor, " B"],
        weight: Tensor | None,
    ) -> Float[Tensor, " B"]:
        r"""Normalize divergence weights over valid samples, optionally updating running bins."""
        options = self._ratio_options
        sample_mask = (
            None
            if self._call_weight is None
            else _ObjectiveReduction(self._call_weight, self._call_weight).active
        )
        h, bins = self._loss_support.f_divergence_weights(
            logits,
            t,
            self.f_div_weighting_function,
            ratio_lower=options.ratio_lower,
            ratio_upper=options.ratio_upper,
            bins=self._bins if options.ratio_normalization else None,
            min_t=self.net.noise_scheduler.min_t,
            max_t=self.net.noise_scheduler.max_t,
            ema_rate=options.ratio_ema_rate,
            logit_weight=weight,
            sample_mask=sample_mask,
            statistics_group=options.statistics_group,
            update_state=self._update_state,
        )
        if options.ratio_normalization and self._update_state:
            self._bins = bins.detach()
        return h

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

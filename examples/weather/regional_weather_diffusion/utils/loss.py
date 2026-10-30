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


"""Loss functions for StormCast training.

Diffusion training uses :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss`
with an :class:`~physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler` or
:class:`~physicsnemo.diffusion.noise_schedulers.EDMLogUniformNoiseScheduler`
(``training.loss.type: edm``), or
:class:`~physicsnemo.diffusion.metrics.losses.WeightedFlowMatchingLoss` with a
:class:`~physicsnemo.diffusion.noise_schedulers.RectifiedFlowNoiseScheduler`
(``training.loss.type: flow_matching``).
When domain parallelism is active, the caller wraps the scheduler with
:class:`~physicsnemo.diffusion.DomainParallelNoiseScheduler` (via
:meth:`~utils.parallel.ParallelHelper.make_domain_parallel_scheduler`) before
passing it to the loss so that sampled sigmas are broadcast across spatial
shards.
"""

from collections.abc import Sequence

import numpy as np
import torch
from utils.nn import flow_conversions

from physicsnemo.diffusion.metrics.losses import (
    WeightedFlowMatchingLoss,
    WeightedMSEDSMLoss,
)
from physicsnemo.diffusion.noise_schedulers import (
    EDMLogUniformNoiseScheduler,
    EDMNoiseScheduler,
    NoiseScheduler,
    RectifiedFlowNoiseScheduler,
)


def build_noise_scheduler(
    loss_cfg,
    logger=None,
) -> NoiseScheduler:
    """Create the training noise scheduler from the Hydra loss config.

    Parameters
    ----------
    loss_cfg : object
        Loss config. With ``type: "flow_matching"`` it must provide ``t_min``
        and ``t_max``; otherwise ``sigma_distribution``, ``sigma_data``, and
        distribution-specific params (``P_mean``/``P_std`` or
        ``sigma_min``/``sigma_max``).
    logger : optional
        Logger for informational messages.

    Returns
    -------
    NoiseScheduler
        A :class:`~physicsnemo.diffusion.noise_schedulers.RectifiedFlowNoiseScheduler`
        for flow matching, otherwise an EDM scheduler.
    """
    if loss_cfg.type == "flow_matching":
        return RectifiedFlowNoiseScheduler(t_min=loss_cfg.t_min, t_max=loss_cfg.t_max)

    sigma_data = loss_cfg.sigma_data
    if isinstance(sigma_data, Sequence):
        sigma_data = list(sigma_data)
        if logger:
            logger.info(
                f"Per-channel sigma_data detected ({len(sigma_data)} channels)."
            )
    else:
        sigma_data = float(sigma_data)

    sigma_dist = loss_cfg.sigma_distribution
    if sigma_dist == "lognormal":
        return EDMNoiseScheduler(
            sigma_data=sigma_data,
            P_mean=loss_cfg.P_mean,
            P_std=loss_cfg.P_std,
        )
    elif sigma_dist == "loguniform":
        return EDMLogUniformNoiseScheduler(
            sigma_min=loss_cfg.sigma_min,
            sigma_max=loss_cfg.sigma_max,
            sigma_data=sigma_data,
        )
    else:
        raise ValueError(
            "training.loss.sigma_distribution must be 'lognormal' or 'loguniform'"
        )


def build_diffusion_loss(
    model: torch.nn.Module, noise_scheduler: NoiseScheduler, loss_cfg
) -> WeightedMSEDSMLoss | WeightedFlowMatchingLoss:
    """Create the diffusion training loss for the configured loss type.

    Both losses share the call signature
    ``loss_fn(x0, weight, t=..., condition=..., **model_kwargs)`` and return
    the per-element weighted loss. ``loss_cfg.prediction_type`` tells the loss
    what the network outputs (``x0``, ``epsilon``, ``score``, or ``flow``);
    the network wrapper (``model.preconditioner``) must produce that quantity.

    Parameters
    ----------
    model : torch.nn.Module
        The wrapped network, see :func:`~utils.nn.wrap_diffusion_network`.
    noise_scheduler : NoiseScheduler
        Training scheduler from :func:`build_noise_scheduler`, possibly
        wrapped for domain parallelism.
    loss_cfg : object
        Loss config with ``type`` and ``prediction_type``.

    Returns
    -------
    WeightedMSEDSMLoss | WeightedFlowMatchingLoss
    """
    prediction_type = loss_cfg.prediction_type
    if loss_cfg.type == "edm":
        # Predictions other than x0 reach the loss through the scheduler's
        # conversion to x0 (score_to_x0, epsilon_to_x0, flow_to_x0). The
        # domain-parallel wrapper only implements the sampling and loss
        # methods; the element-wise conversions come from the wrapped scheduler
        to_x0 = {}
        if prediction_type != "x0":
            converter = getattr(noise_scheduler, "inner_scheduler", noise_scheduler)
            to_x0[f"{prediction_type}_to_x0_fn"] = getattr(
                converter, f"{prediction_type}_to_x0"
            )
        return WeightedMSEDSMLoss(
            model,
            noise_scheduler,
            prediction_type=prediction_type,
            reduction="none",
            **to_x0,
        )
    if loss_cfg.type != "flow_matching":
        raise ValueError("training.loss.type must be 'edm' or 'flow_matching'")

    # The loss converts the prediction and the target to the flow with the
    # scheduler, except for x-prediction, which compares the predicted and true
    # clean states directly: ||x0_hat - x0||^2 (the scheduler's x0_to_flow would
    # weight the error by 1 / t^2)
    to_flow = flow_conversions(noise_scheduler)
    conversions = {"x0_to_flow_fn": to_flow["x0"]}
    if prediction_type == "x0":
        conversions["x0_to_flow_fn"] = lambda x0, x_t, t: x0
    elif prediction_type in ("score", "epsilon"):
        conversions[f"{prediction_type}_to_flow_fn"] = to_flow[prediction_type]
    return WeightedFlowMatchingLoss(
        model,
        noise_scheduler,
        prediction_type=prediction_type,
        reduction="none",
        **conversions,
    )


class RegressionLoss:
    """MSE loss for a regression model.

    Holds a reference to the model so that the call signature mirrors
    :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss`:
    ``loss_fn(x0, weight, condition=..., **kwargs)``.

    Parameters
    ----------
    model : torch.nn.Module
        Regression network (e.g. ``StormCastUNet``).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def __call__(
        self,
        x0: torch.Tensor,
        weight: torch.Tensor,
        *,
        condition: torch.Tensor,
        lead_time_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-pixel squared error, optionally weighted.

        Parameters
        ----------
        x0 : torch.Tensor
            Target data, shape ``[B, C, H, W]``.
        weight : torch.Tensor
            Per-element weight broadcastable to ``x0`` shape.
        condition : torch.Tensor
            Model input, shape ``[B, C_cond, H, W]``.
        lead_time_label : torch.Tensor | None
            Optional lead-time label, shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Weighted per-pixel squared error ``[B, C, H, W]``.
        """
        labels = {} if lead_time_label is None else {"lead_time_label": lead_time_label}
        prediction = self.model(x=condition, **labels)
        return weight * (prediction - x0) ** 2


class SigmaBinTracker:
    """Track per-sigma-bin loss for diffusion training diagnostics.

    Accumulates sample-level L2 loss into equal-probability sigma bins,
    then logs per-bin means via an experiment logger.

    Parameters
    ----------
    loss_cfg : object
        Loss config with attributes: ``track_sigma_bin_loss``, ``sigma_bin_count``,
        ``sigma_bin_edges``, ``sigma_distribution``, ``sigma_min``, ``sigma_max``,
        ``P_mean``, ``P_std``.
    device : torch.device
        Device for accumulator tensors.
    loss_type : str
        ``"regression"``, ``"edm"``, or ``"flow_matching"``.  Tracking is
        disabled for regression. For flow matching the bins divide the
        uniform time distribution on ``[t_min, t_max]`` instead of the sigma
        distribution.
    """

    def __init__(self, loss_cfg, device: torch.device, loss_type: str = "edm"):
        self.enabled = loss_type != "regression" and bool(loss_cfg.track_sigma_bin_loss)
        self.device = device
        self._edges: torch.Tensor | None = None
        self._loss_sum: torch.Tensor | None = None
        self._count: torch.Tensor | None = None
        if not self.enabled:
            return

        if len(loss_cfg.sigma_bin_edges) >= 2:
            edges = np.asarray(loss_cfg.sigma_bin_edges, dtype=np.float64)
        elif loss_type == "flow_matching":
            edges = np.linspace(
                loss_cfg.t_min, loss_cfg.t_max, int(loss_cfg.sigma_bin_count) + 1
            )
        else:
            n_edges = int(loss_cfg.sigma_bin_count) + 1
            if loss_cfg.sigma_distribution == "loguniform":
                q = np.linspace(0.0, 1.0, n_edges, dtype=np.float64)
                log_lo = float(np.log(loss_cfg.sigma_min))
                log_hi = float(np.log(loss_cfg.sigma_max))
                edges = np.exp(log_lo + q * (log_hi - log_lo))
            else:
                q = torch.linspace(0.0, 1.0, n_edges, dtype=torch.float64)
                q = q.clamp(1e-6, 1.0 - 1e-6)
                z = torch.distributions.Normal(0.0, 1.0).icdf(q)
                log_edges = float(loss_cfg.P_mean) + float(loss_cfg.P_std) * z
                edges = torch.exp(log_edges).cpu().numpy()
        self._edges = torch.as_tensor(edges, dtype=torch.float32, device=device)

    @property
    def edges(self) -> list[float] | None:
        """Bin edges as a Python list, or None if disabled."""
        if self._edges is None:
            return None
        return self._edges.detach().cpu().tolist()

    def reset(self) -> None:
        """Zero accumulators at the start of each training step."""
        if not self.enabled:
            return
        n = int(self._edges.numel() - 1)
        self._loss_sum = torch.zeros(n, device=self.device, dtype=torch.float32)
        self._count = torch.zeros(n, device=self.device, dtype=torch.float32)

    def update(
        self,
        loss: torch.Tensor,
        sigma: torch.Tensor | None,
    ) -> None:
        """Accumulate one micro-batch of per-sample loss into bins.

        Parameters
        ----------
        loss : torch.Tensor
            Per-pixel loss, shape ``[B, C, H, W]``.
        sigma : torch.Tensor | None
            Sampled sigma values, shape ``[B, 1, 1, 1]`` or ``[B]``.
        """
        if not self.enabled or sigma is None:
            return
        sample_loss = loss.detach().mean(dim=(1, 2, 3))
        if hasattr(sample_loss, "to_local"):
            # Cast to regular tensor for bucketize after reducing across sharded dims
            sample_loss = sample_loss.to_local()
        sample_sigma = sigma.detach().reshape(-1).to(torch.float32)
        bin_idx = torch.bucketize(sample_sigma, self._edges) - 1
        n_bins = int(self._edges.numel() - 1)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        if not torch.any(valid):
            return
        idx = bin_idx[valid]
        self._loss_sum.index_add_(0, idx, sample_loss[valid])
        self._count.index_add_(
            0, idx, torch.ones_like(sample_loss[valid], dtype=torch.float32)
        )

    def log(self, logger, world_size: int = 1) -> None:
        """All-reduce across ranks and log per-bin means.

        Parameters
        ----------
        logger : ExperimentLogger
            Must have a ``log_value(tag, value)`` method.
        world_size : int
            Number of distributed ranks (1 = single GPU).
        """
        if not self.enabled or self._count is None:
            return
        if world_size > 1:
            for t in (self._loss_sum, self._count):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        edges = self._edges.detach().cpu().tolist()
        for b in range(int(self._edges.numel() - 1)):
            count = float(self._count[b].item())
            if count <= 0:
                continue
            tag = f"[{edges[b]:.3e},{edges[b + 1]:.3e})"
            logger.log_value(
                f"loss/train_sigma_bin/{tag}",
                float((self._loss_sum[b] / self._count[b]).item()),
            )

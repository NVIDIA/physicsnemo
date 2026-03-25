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


"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from typing import Callable

import numpy as np
import torch
from torch.distributed.tensor.placement_types import Replicate


from physicsnemo.domain_parallel.shard_tensor import scatter_tensor


class EDMLoss:
    """
    Loss function proposed in the EDM paper.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float | torch.Tensor, optional
        Standard deviation for data, by default 0.5. Can also be a tensor; to use
        per-channel sigma_data, pass a tensor of shape (1, number_of_channels, 1, 1).

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float | torch.Tensor = 0.5,
        sigma_source_rank: int | None = None,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_source_rank = sigma_source_rank

    def get_noise_level(self, y: torch.Tensor) -> torch.Tensor:
        """Sample the sigma noise parameter for each sample."""
        shape = (y.shape[0], 1, 1, 1)
        rnd_normal = torch.randn(shape, device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        return sigma

    def get_loss_weight(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute loss weight for each sample."""
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        return weight

    def sample_noise(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Sample the noise."""
        return torch.randn_like(y) * sigma

    def replicate_in_mesh(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if (hasattr(y, "_local_tensor") or hasattr(y, "placements")) and hasattr(
            y, "device_mesh"
        ):
            # y is sharded - need to replicate sigma to be compatible with DTensor operations
            # Get the source rank for replication
            # Replicate sigma across the domain mesh
            # This ensures all spatial shards see the same sigma values
            x = scatter_tensor(
                x,
                self.sigma_source_rank,
                y.device_mesh,
                placements=(Replicate(),),
                global_shape=x.shape,
                dtype=x.dtype,
            )
        return x

    def __call__(
        self,
        net: torch.nn.Module,
        images: torch.Tensor,
        condition: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        augment_pipe: Callable | None = None,
        lead_time_label: torch.Tensor | None = None,
        return_sigma: bool = False,
    ) -> torch.Tensor:
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is drawn from the `get_noise_level`
        function. The calculated loss is weighted as a function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        condition: torch.Tensor
            Condition to be passed to the `condition` argument of `net.forward`.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        lead_time_label: torch.Tensor, optional
            Lead-time labels to pass to the model, shape ``(batch_size,)``.
            If not provided, the model is called without a lead-time label input.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        sigma = self.get_noise_level(y)
        sigma = self.replicate_in_mesh(sigma, y)
        weight = self.get_loss_weight(y, sigma)

        n = self.sample_noise(y, sigma)

        optional_args = {
            "augment_labels": augment_labels,
            "lead_time_label": lead_time_label,
            "class_labels": labels,
        }
        # drop None items to support models that don't have these arguments in `forward`
        optional_args = {k: v for (k, v) in optional_args.items() if v is not None}
        if condition is not None:
            D_yn = net(
                y + n,
                sigma.flatten(),
                condition=condition,
                # class_labels=labels,
                **optional_args,
            )
        else:
            D_yn = net(y + n, sigma.flatten(), labels, **optional_args)
        loss = weight * ((D_yn - y) ** 2)
        if return_sigma:
            return (loss, sigma)
        return loss


class EDMLossLogUniform(EDMLoss):
    """
    EDM Loss with log-uniform sampling for `sigma`.

    Parameters
    ----------
    sigma_min: float, optional
        Minimum value for `sigma` computation, by default 0.02.
    sigma_max: float, optional:
        Minimum value for `sigma` computation, by default 1000.
    sigma_data: float | torch.Tensor, optional
        Standard deviation for data, by default 0.5. Can also be a tensor; to use
        per-channel sigma_data, pass a tensor of shape (1, number_of_channels, 1, 1).
    """

    def __init__(
        self,
        sigma_min: float = 0.02,
        sigma_max: float = 1000,
        sigma_data: float | torch.Tensor = 0.5,
        sigma_source_rank: int | None = None,
    ):
        self.sigma_data = sigma_data
        self.sigma_source_rank = sigma_source_rank
        self.log_sigma_min = float(np.log(sigma_min))
        self.log_sigma_diff = float(np.log(sigma_max)) - self.log_sigma_min

    def get_noise_level(self, y: torch.Tensor) -> torch.Tensor:
        """Sample the sigma noise parameter for each sample."""
        shape = (y.shape[0], 1, 1, 1)
        rnd_uniform = torch.rand(shape, device=y.device)
        sigma = (self.log_sigma_min + rnd_uniform * self.log_sigma_diff).exp()
        return sigma


def regression_loss_fn(
    net,
    images: torch.Tensor,
    condition: torch.Tensor,
    class_labels=None,
    lead_time_label: torch.Tensor | None = None,
    augment_pipe=None,
    return_model_outputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """MSE loss for the StormCast regression model.

    Shares call signature with EDMLoss so the same training loop works for both.

    Args:
        net: regression network (e.g. StormCastUNet).
        images: target data, shape ``[B, C, H, W]``.
        condition: model input, shape ``[B, C_cond, H, W]``.
        class_labels: unused (present for EDMLoss call-signature parity).
        lead_time_label: optional lead-time label, shape ``(B,)``.
        augment_pipe: optional data augmentation callable.
        return_model_outputs: if True, return ``(loss, prediction)``.

    Returns:
        Per-pixel squared error ``[B, C, H, W]``, or ``(loss, prediction)``
        when *return_model_outputs* is True.
    """
    y, augment_labels = (
        augment_pipe(images) if augment_pipe is not None else (images, None)
    )
    labels = {} if lead_time_label is None else {"lead_time_label": lead_time_label}
    D_yn = net(x=condition, **labels)
    loss = (D_yn - y) ** 2
    if return_model_outputs:
        return loss, D_yn
    return loss


class SigmaBinTracker:
    """Track per-sigma-bin loss and bias for diffusion training diagnostics.

    Accumulates sample-level L2 loss and signed denoising bias into
    equal-probability sigma bins, then logs per-bin means via an
    experiment logger.

    Parameters
    ----------
    loss_cfg : object
        Loss config with attributes: ``track_sigma_bin_loss``, ``sigma_bin_count``,
        ``sigma_bin_edges``, ``sigma_distribution``, ``sigma_min``, ``sigma_max``,
        ``P_mean``, ``P_std``.
    device : torch.device
        Device for accumulator tensors.
    loss_type : str
        ``"regression"`` or ``"edm"``.  Tracking is disabled for regression.
    """

    def __init__(self, loss_cfg, device: torch.device, loss_type: str = "edm"):
        self.enabled = loss_type != "regression" and bool(loss_cfg.track_sigma_bin_loss)
        self.device = device
        self._edges: torch.Tensor | None = None
        self._loss_sum: torch.Tensor | None = None
        self._bias_sum: torch.Tensor | None = None
        self._count: torch.Tensor | None = None
        if not self.enabled:
            return

        if len(loss_cfg.sigma_bin_edges) >= 2:
            edges = np.asarray(loss_cfg.sigma_bin_edges, dtype=np.float64)
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
        self._bias_sum = torch.zeros(n, device=self.device, dtype=torch.float32)
        self._count = torch.zeros(n, device=self.device, dtype=torch.float32)

    def update(
        self,
        loss: torch.Tensor,
        sigma: torch.Tensor | None,
        bias: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one micro-batch of per-sample loss/bias into bins.

        Parameters
        ----------
        loss : torch.Tensor
            Per-pixel loss, shape ``[B, C, H, W]``.
        sigma : torch.Tensor | None
            Sampled sigma values, shape ``[B, 1, 1, 1]`` or ``[B]``.
        bias : torch.Tensor | None
            Per-sample mean signed error ``[B]`` (from EDMLoss ``return_sigma``).
        """
        if not self.enabled or sigma is None:
            return
        sample_loss = loss.detach().mean(dim=(1, 2, 3))
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
        if bias is not None:
            self._bias_sum.index_add_(0, idx, bias.detach().to(torch.float32)[valid])

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
            for t in (self._loss_sum, self._bias_sum, self._count):
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
            logger.log_value(
                f"bias/train_sigma_bin/{tag}",
                float((self._bias_sum[b] / self._count[b]).item()),
            )

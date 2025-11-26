# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""
Unified loss functions for CO2 sequestration modeling.

Available base losses:
- mse: Mean Squared Error (L2 loss)
- l1: Mean Absolute Error (L1 loss)
- relative_l2: Relative L2 loss (scale-invariant)

Optional features:
- Masking: Apply loss only on active reservoir regions (irregular domains)
- Derivative: Add physics-informed spatial derivative constraints
"""

import torch
import torch.nn as nn


class SimpleRelativeL2Loss(nn.Module):
    """
    Simple Relative L2 Loss - No masking, no derivatives, no complications.

    Formula:
        loss = ||pred - target||_2 / ||target||_2

    Where ||.||_2 is the L2 norm (Frobenius norm for tensors).
    This is computed per sample and then averaged across the batch.
    """

    def __init__(self):
        super(SimpleRelativeL2Loss, self).__init__()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute simple relative L2 loss.

        Args:
            predictions: Model predictions (B, H, W, T)
            targets: Ground truth (B, H, W, T)
            inputs: Unused (kept for compatibility with loss function interface)

        Returns:
            Scalar loss value
        """
        batch_size = predictions.shape[0]

        # Flatten spatial dimensions for each sample
        pred_flat = predictions.reshape(batch_size, -1)  # (B, H*W*T)
        target_flat = targets.reshape(batch_size, -1)  # (B, H*W*T)

        # Compute L2 norm of difference and target for each sample
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)  # (B,)
        target_norm = torch.norm(target_flat, p=2, dim=1)  # (B,)

        # Relative L2 loss per sample
        relative_loss = diff_norm / target_norm  # (B,)

        # Average across batch
        return relative_loss.mean()


class UnifiedLoss(nn.Module):
    """Unified loss function with configurable base loss, masking, and derivatives.

    This flexible loss function supports:
    1. Multiple base losses (MSE, L1, Relative L2)
    2. Optional masking for irregular domains
    3. Optional physics-informed derivative constraints

    Formula:
        If use_derivative=False:
            loss = base_loss(pred, target)

        If use_derivative=True:
            loss = base_loss(pred, target) + derivative_weight * base_loss(∂pred, ∂target)

    where derivatives are computed using central finite differences with adaptive grid spacing.

    Parameters
    ----------
    base_loss_type : str
        Base loss function: 'mse', 'l1', or 'relative_l2'
    use_mask : bool, optional
        Whether to apply masking for irregular domains, by default False
    use_derivative : bool, optional
        Whether to add derivative (physics-informed) term, by default False
    derivative_weight : float, optional
        Weight for derivative term (only used if use_derivative=True), by default 0.5
    derivative_dim : str or list of str, optional
        Dimension(s) for derivative: 'dz' (height/z-direction), 'dx' (width/x-direction),
        or ['dx', 'dz'] for both, by default 'dx'
    eps : float, optional
        Epsilon for numerical stability (relative_l2 only), by default 1e-6
    reduction : str, optional
        Reduction method: 'mean', 'sum', or 'none', by default 'mean'

    Example
    -------
    >>> # Baseline: Pure Relative L2
    >>> loss_fn = UnifiedLoss(base_loss_type='relative_l2', use_mask=False, use_derivative=False)
    >>> loss = loss_fn(pred, target)

    >>> # With masking: Relative L2 on active regions only
    >>> loss_fn = UnifiedLoss(base_loss_type='relative_l2', use_mask=True, use_derivative=False)
    >>> loss = loss_fn(pred, target, inputs)

    >>> # Physics-informed: Relative L2 + derivatives in x-direction
    >>> loss_fn = UnifiedLoss(base_loss_type='relative_l2', use_mask=True, use_derivative=True,
    ...                       derivative_weight=0.5, derivative_dim='dx')
    >>> loss = loss_fn(pred, target, inputs)

    >>> # Physics-informed with both x and z derivatives
    >>> loss_fn = UnifiedLoss(base_loss_type='relative_l2', use_derivative=True,
    ...                       derivative_weight=0.5, derivative_dim=['dx', 'dz'])
    >>> loss = loss_fn(pred, target, inputs)

    Note
    ----
    - If use_mask=True or use_derivative=True, inputs must be provided in forward()
    - Masking uses first channel at first time step: (inputs[:, 0, :, :, 0] != 0)
    - Derivative grid spacing is extracted from inputs[:, 0, :, 0, -3]

    Reference
    ---------
    Based on U-FNO paper: Wen, G., et al. "U-FNO" (2022)
    https://www.sciencedirect.com/science/article/pii/S0309170822000562
    """

    def __init__(
        self,
        base_loss_type: str = "relative_l2",
        use_mask: bool = False,
        use_derivative: bool = False,
        derivative_weight: float = 0.5,
        derivative_dim="dx",
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()

        # Validate parameters
        if base_loss_type not in ["mse", "l1", "relative_l2"]:
            raise ValueError(
                f"base_loss_type must be 'mse', 'l1', or 'relative_l2', got {base_loss_type}"
            )
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

        # Normalize derivative_dim to list format
        if isinstance(derivative_dim, str):
            derivative_dims = [derivative_dim]
        elif isinstance(derivative_dim, list):
            derivative_dims = derivative_dim
        elif isinstance(derivative_dim, int):
            # Support legacy format: 2 -> 'dz', 3 -> 'dx'
            derivative_dims = ["dz" if derivative_dim == 2 else "dx"]
        else:
            raise ValueError(
                f"derivative_dim must be str, list, or int, got {type(derivative_dim)}"
            )

        # Validate derivative dimensions
        for dim in derivative_dims:
            if dim not in ["dx", "dz"]:
                raise ValueError(
                    f"derivative_dim values must be 'dx' or 'dz', got {dim}"
                )

        self.base_loss_type = base_loss_type
        self.use_mask = use_mask
        self.use_derivative = use_derivative
        self.derivative_weight = derivative_weight
        self.derivative_dims = derivative_dims
        self.eps = eps
        self.reduction = reduction

        # Grid spacings (extracted from data)
        self.grid_spacings = {}

    def _compute_base_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute base loss (MSE, L1, or Relative L2).

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values
        target : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Loss value
        """
        if self.base_loss_type == "mse":
            # Mean Squared Error
            loss = (pred - target) ** 2

        elif self.base_loss_type == "l1":
            # Mean Absolute Error
            loss = torch.abs(pred - target)

        elif self.base_loss_type == "relative_l2":
            # Relative L2: ||pred - target||_2 / ||target||_2
            batch_size = pred.shape[0]
            pred_flat = pred.reshape(batch_size, -1)
            target_flat = target.reshape(batch_size, -1)

            diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
            target_norm = torch.norm(target_flat, p=2, dim=1)

            loss = diff_norm / target_norm

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def _extract_grid_spacing(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract grid spacing from input data.

        The 12 input channels are: [kr, kz, porosity, inj_loc, inj_rate, pressure,
        temperature, Swi, Lam, grid_x, grid_y, grid_t]
        Channel -3 = grid_x (spatial coordinates)

        Parameters
        ----------
        inputs : torch.Tensor
            Input data of shape (batch, H, W, T, C) where C=12
            H = height dimension (96), W = width dimension (200)

        Returns
        -------
        torch.Tensor
            Grid spacing of shape (1, 1, W-2, 1) for broadcasting with (B, H, W-2, T)
        """
        # Extract grid_x values along WIDTH dimension: [batch 0, height 0, all widths, time 0, channel -3]
        grid_x = inputs[0, 0, :, 0, -3]  # (W=200,) - extract grid_x channel values
        grid_dx = grid_x[1:-1] + grid_x[:-2] / 2 + grid_x[2:] / 2  # (W-2=198,)
        grid_dx = grid_dx[
            None, None, :, None
        ]  # (1, 1, 198, 1) for broadcasting with (B, H, W-2, T)

        return grid_dx

    def _compute_derivative(
        self, field: torch.Tensor, grid_dx: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial derivative using central finite differences.


        Parameters
        ----------
        field : torch.Tensor
            Input field of shape (batch, H=96, W=200, T=24)
        grid_dx : torch.Tensor
            Grid spacing of shape (1, 1, W-2=198, 1)

        Returns
        -------
        torch.Tensor
            Derivative field of shape (batch, H=96, W-2=198, T=24)
        """
        # Compute derivative in WIDTH direction (dimension 2)
        derivative = (field[:, :, 2:, :] - field[:, :, :-2, :]) / grid_dx

        return derivative

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute unified loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted values of shape (batch, H, W, T)
        target : torch.Tensor
            Target values of shape (batch, H, W, T)
        inputs : torch.Tensor, optional
            Input data of shape (batch, H, W, T, C) - required if use_mask=True or use_derivative=True

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Validate inputs
        if (self.use_mask or self.use_derivative) and inputs is None:
            raise ValueError(
                "inputs must be provided when use_mask=True or use_derivative=True"
            )

        batch_size = pred.shape[0]

        # --- Case 1: Masking enabled (vectorized implementation) ---
        if self.use_mask:
            # Create mask from inputs: non-zero locations at first channel and first timestep
            # inputs shape: (B, H, W, T, C)
            mask = (inputs[:, :, :, 0:1, 0] != 0).repeat(
                1, 1, 1, pred.shape[3]
            )  # (B, H, W, T)

            # Vectorized masked loss computation
            if self.base_loss_type == "relative_l2":
                # Compute relative L2 loss per sample with masking (optimized)
                # pred, target, mask: (B, H, W, T)
                # Use list comprehension for faster iteration
                ori_loss_per_sample = [
                    torch.norm(pred[i][mask[i]] - target[i][mask[i]], p=2)
                    / torch.norm(target[i][mask[i]], p=2)
                    for i in range(batch_size)
                ]
                ori_loss = torch.stack(ori_loss_per_sample).mean()  # Mean across batch

            elif self.base_loss_type == "mse":
                # MSE with masking - can be fully vectorized
                diff = (pred - target) ** 2
                ori_loss = (diff * mask.float()).sum() / mask.float().sum()

            elif self.base_loss_type == "l1":
                # L1 with masking - can be fully vectorized
                diff = torch.abs(pred - target)
                ori_loss = (diff * mask.float()).sum() / mask.float().sum()

            # Add derivative loss if enabled
            if self.use_derivative:
                # Extract grid spacing
                grid_dx = self._extract_grid_spacing(inputs).to(pred.device)

                # Compute derivatives: dy = (y[:,:,2:,:] - y[:,:,:-2,:])/grid_dx
                dy_pred = self._compute_derivative(pred, grid_dx)  # (B, H, W-2, T)
                dy_target = self._compute_derivative(target, grid_dx)  # (B, H, W-2, T)

                # Adjust mask for derivative (width dimension reduced by 2)
                mask_dy = mask[:, :, : dy_pred.shape[2], :]  # (B, H, W-2, T)

                # Vectorized derivative loss computation (optimized)
                if self.base_loss_type == "relative_l2":
                    der_loss_per_sample = [
                        torch.norm(
                            dy_pred[i][mask_dy[i]] - dy_target[i][mask_dy[i]], p=2
                        )
                        / torch.norm(dy_target[i][mask_dy[i]], p=2)
                        for i in range(batch_size)
                    ]
                    der_loss = torch.stack(der_loss_per_sample).mean()

                elif self.base_loss_type == "mse":
                    diff = (dy_pred - dy_target) ** 2
                    der_loss = (diff * mask_dy.float()).sum() / mask_dy.float().sum()

                elif self.base_loss_type == "l1":
                    diff = torch.abs(dy_pred - dy_target)
                    der_loss = (diff * mask_dy.float()).sum() / mask_dy.float().sum()

                total_loss = ori_loss + self.derivative_weight * der_loss
            else:
                total_loss = ori_loss

        # --- Case 2: No masking ---
        else:
            # Compute data loss on full fields
            data_loss = self._compute_base_loss(pred, target)

            # Add derivative loss if enabled
            if self.use_derivative:
                # Extract grid spacing
                grid_dx = self._extract_grid_spacing(inputs).to(pred.device)

                # Compute derivatives
                dy_pred = self._compute_derivative(pred, grid_dx)
                dy_target = self._compute_derivative(target, grid_dx)

                # Compute derivative loss
                der_loss = self._compute_base_loss(dy_pred, dy_target)

                total_loss = data_loss + self.derivative_weight * der_loss
            else:
                total_loss = data_loss

        return total_loss


def get_loss_function(loss_config):
    """Factory function to create loss function from config.

    Parameters
    ----------
    loss_config : DictConfig or dict
        Loss configuration with fields:
        - base_loss_type: str, base loss ('mse', 'l1', 'relative_l2', 'simple_relative_l2')
        - use_mask: bool, whether to use masking
        - use_derivative: bool, whether to add derivative term
        - derivative_weight: float, weight for derivative
        - derivative_dim: str or list, dimension(s) for derivatives ('dx', 'dz', or ['dx', 'dz'])
        - eps: float, epsilon for relative_l2
        - reduction: str, reduction method

    Returns
    -------
    UnifiedLoss or SimpleRelativeL2Loss
        Configured loss function

    Example
    -------
    >>> from omegaconf import DictConfig
    >>> cfg = DictConfig({
    ...     'base_loss_type': 'simple_relative_l2',
    ... })
    >>> loss_fn = get_loss_function(cfg)
    """
    loss_type = loss_config.get("base_loss_type", "relative_l2")

    # Use simple loss if requested
    if loss_type == "simple_relative_l2":
        return SimpleRelativeL2Loss()

    # Otherwise use UnifiedLoss
    return UnifiedLoss(
        base_loss_type=loss_type,
        use_mask=loss_config.get("use_mask", False),
        use_derivative=loss_config.get("use_derivative", False),
        derivative_weight=loss_config.get("derivative_weight", 0.5),
        derivative_dim=loss_config.get("derivative_dim", "dx"),
        eps=loss_config.get("eps", 1e-6),
        reduction=loss_config.get("reduction", "mean"),
    )


# For convenience, export main class and factory
__all__ = [
    "SimpleRelativeL2Loss",
    "UnifiedLoss",
    "get_loss_function",
]

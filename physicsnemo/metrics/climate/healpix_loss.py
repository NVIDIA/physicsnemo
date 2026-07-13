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

from typing import Sequence

import numpy as np
import torch as th

from physicsnemo.core.version_check import OptionalImport

xr = OptionalImport("xarray")
earth2grid = OptionalImport(
    "earth2grid", "Install earth2grid from https://github.com/earth2grid/earth2grid"
)
cuhpx = OptionalImport("cuhpx", "Install cuhpx from https://github.com/NVIDIA/cuhpx")

"""
Custom dlwp compatible loss classes that allow for more sophisticated training optimization.

Each custom loss should inherit all methods of th.nn._Loss base class or subclasses thereof. 
Additionally, custom loss classes should define a setup function which receives the trainer object. 
The setup function should be used to move tensors to appropriate gpus and finalize configuration
of the loss calculation using information about the model (trainer.model) and trainer. Custom
losses should also redefine the forward function to contain a flag indicating whether or not to 
average output channels. This is used in the varible wise logging of validation loss by the trainer. 

"""


class BaseMSE(th.nn.MSELoss):
    """
    Base MSE class offers impementaion for basic MSE loss compatable with dlwp custom loss training
    """

    def __init__(
        self,
    ):
        """Constructer for BaseMSE"""
        super().__init__()
        self.device = None

    def setup(self, trainer):
        """
        Nothing to implement here
        """
        pass

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the base MSE class
        Tensors are expected to be in the shape [N, B, F, C, H, W]

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        if not (prediction.ndim == 6 and target.ndim == 6):
            raise AssertionError("Expected predictions to have 6 dimensions")

        d = ((target - prediction) ** 2).mean(dim=(0, 1, 2, 4, 5))
        if average_channels:
            return th.mean(d)
        else:
            return d


class WeightedMSE(th.nn.MSELoss):
    """
    Loss object that allows for user defined weighting of variables when calculating MSE
    """

    def __init__(
        self,
        weights: Sequence = [],
    ):
        """
        Parameters
        ----------
        weights: Sequence
            list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        self.device = None

    def setup(self, trainer):
        """
        pushes weights to cuda device
        """

        if len(trainer.output_variables) != len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the WeightedMSE pass
        Tensors are expected to be in the shape [N, B, F, C, H, W]

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        if not (prediction.ndim == 6 and target.ndim == 6):
            raise AssertionError("Expected predictions to have 6 dimensions")

        d = ((target - prediction) ** 2).mean(dim=(0, 1, 2, 4, 5)) * self.loss_weights
        if average_channels:
            return th.mean(d)
        else:
            return d


class ConditionalWeightLoss(th.nn.MSELoss):
    """
    Conditional loss for precipitation diagnostic model.
    (Total 6hr precipitation is the only output field.)
    """

    def __init__(
        self,
        weight=(0.01, 1.0),
        b=None,
        w=1,
    ):
        """
        Parameters
        -----------
        weight: tuple of floats
            weight[0] is used when the target precipitation value is zero
            weight[1] is used for all non-zero precipitation
        b: float
            Exponential scaling factor used to define weighting curve for non-zero precip. weights
        w: float
            Final scaling factor applied to loss.
        """

        super().__init__()
        self.weight_zero = weight[0]
        self.weight_nonzero = weight[1]
        self.b = b
        self.device = None
        self.w = w

    def setup(self, trainer):
        """Setup function that moves tensors to the same device as the model"""
        self.b = th.tensor(self.b, device=trainer.device)
        self.w = th.tensor(self.w, device=trainer.device)

    def forward(self, prediction, target, average_channels=True):
        """
        Computes the MSE of model prediction and applies weights for zero and non-zero precipitation cases.

        Parameters
        -----------
        prediction: torch.tensor
            The prediction tensor
        target: torch.Tensor
            The target tensor
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """
        weights_for_zero = th.ones_like(target) * self.weight_zero
        weights_for_nonzero = (th.ones_like(target) * self.weight_nonzero) * th.exp(
            self.b * target
        )
        weights = th.where(target > 0, weights_for_nonzero, weights_for_zero)
        if average_channels:
            loss = (th.mean(weights * (prediction - target) ** 2)) * self.w
        else:
            loss = (weights * (prediction - target) ** 2) * self.w
        return loss


class OceanMSE(th.nn.MSELoss):
    """
    Ocean MSE class offers impementaion for MSE loss weighted by a land-sea-mask field.
    """

    def __init__(
        self,
        lsm_file: str,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "lsm"},
    ):
        """
        Parameters
        ----------
        lsm_file: str
            land-sea-mask file
        open_dict: dict, optional
            dictionary that store land-sea-mask file information
        selection_dict: dict, optional
            dictionary that store channel selection information
        """
        super().__init__()
        self.device = None
        self.lsm_file = lsm_file
        self.lsm_ds = None
        self.open_dict = open_dict
        self.selection_dict = selection_dict
        self.lsm_tensor = None
        self.lsm_sum_calculated = False
        self.lsm_sum = None
        self.lsm_var_sum = None

    def setup(self, trainer):
        """
        reshape lsm and put on device
        """
        self.lsm_ds = xr.open_dataset(self.lsm_file, **self.open_dict).constants.sel(
            self.selection_dict
        )
        # 1-lsm gives the percentage of pixel that has ocean
        self.lsm_tensor = 1 - th.tensor(
            np.expand_dims(self.lsm_ds.values, (0, 2, 3))
        ).to(trainer.device)

    def forward(self, prediction, target, average_channels=True):
        if not self.lsm_sum_calculated:
            self.lsm_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum()
            self.lsm_var_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum(
                dim=(0, 1, 2, 4, 5)
            )
            self.lsm_sum_calculated = True
        # average weighted
        ocean_err = ((target - prediction) ** 2) * self.lsm_tensor
        ocean_mean_err = ocean_err.sum(dim=(0, 1, 2, 4, 5))
        if average_channels:
            return th.sum(ocean_mean_err) / self.lsm_sum
        else:
            return ocean_mean_err / self.lsm_var_sum


class WeightedOceanMSE(th.nn.MSELoss):
    """
    Ocean MSE class offers impementaion for MSE loss with:
    1) weighted by a land-sea-mask field.
    2) weighted by channel (e.g. sic more than sst)
    """

    def __init__(
        self,
        lsm_file: str,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "lsm"},
        weights: Sequence = [],
    ):
        """ """
        super().__init__()
        self.device = None
        self.lsm_file = lsm_file
        self.lsm_ds = None
        self.open_dict = open_dict
        self.selection_dict = selection_dict
        self.lsm_tensor = None
        self.lsm_sum_calculated = False
        self.lsm_sum = None
        self.lsm_var_sum = None
        self.loss_weights = th.tensor(weights)

    def setup(self, trainer):
        """
        reshape lsm and put on device
        pushes weights to cuda device
        """
        ### 1. OCEAN PREP ###
        self.lsm_ds = xr.open_dataset(self.lsm_file, **self.open_dict).constants.sel(
            self.selection_dict
        )
        # 1-lsm gives the percentage of pixel that has ocean
        self.lsm_tensor = 1 - th.tensor(
            np.expand_dims(self.lsm_ds.values, (0, 2, 3))
        ).to(trainer.device)

        ### 2. WEIGHTS PREP ###

        if not len(trainer.output_variables) == len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)

    def forward(self, prediction, target, average_channels=True):
        if not self.lsm_sum_calculated:
            self.lsm_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum()
            self.lsm_var_sum = th.broadcast_to(self.lsm_tensor, target.shape).sum(
                dim=(0, 1, 2, 4, 5)
            )
            self.lsm_sum_calculated = True
        # average weighted
        ocean_err = ((target - prediction) ** 2) * self.lsm_tensor
        ocean_mean_err = ocean_err.sum(dim=(0, 1, 2, 4, 5))
        ocean_mean_err = ocean_mean_err * self.loss_weights

        if average_channels:
            return th.sum(ocean_mean_err) / self.lsm_sum
        else:
            return ocean_mean_err / self.lsm_var_sum


class WeightedCRPSLoss(th.nn.MSELoss):
    """
    Probabilistic loss function that allows for user defined weighting of variables when calculating CRPS.
    """

    def __init__(
        self,
        weights: Sequence = [],
        n_members: int = 2,
        alpha: float = 0.95,
        mean_penalty: float = 0.0,
        lsm_file: str = None,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "land_sea_mask"},
        multiscale: float = 0.0,
        masked_processing: bool = False,
    ):
        """
        Parameters
        ----------
        weights: Sequence
            list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        n_members: int
            number of ensemble members in the model output
        alpha: float
            hyperparamter for approximating fair CRPS loss. between 0 and 1, 1 corresponds to a fair CRPS loss.
        mean_penalty: float
            weight for the penalty constraining the global mean of the ensemble to be close to the target mean
            if 0, no penalty is applied
        lsm_file: str
            land-sea-mask file
        open_dict: dict, optional
            dictionary that store land-sea-mask file information
        selection_dict: dict, optional
            dictionary that store channel selection information
        multiscale: float, optional
            weight for the multiscale CRPS loss. Default is 0, no multiscale loss is applied.
        masked_processing: bool, optional
            whether masked pixels should be excluded from the processing. Default is False, no masked processing is applied.
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        if n_members < 2:
            raise ValueError("n_members must be at least 2 for CRPS loss to be defined")
        else:
            self.n_members = n_members
        self.device = None
        self.mean_penalty = mean_penalty
        self.multiscale = multiscale
        self.scales = [4, 16, 32]
        self.masked_processing = masked_processing

        if lsm_file is not None:
            self.lsm_ds = xr.open_dataset(lsm_file, **open_dict).constants.sel(
                selection_dict
            )
            self.lsm_tensor = 1 - th.tensor(
                np.expand_dims(self.lsm_ds.values, (0, 2, 3))
            )
        else:
            self.lsm_tensor = th.ones(
                1, 1, 1, 1, 1, 1
            )  # Spoof the tensor dimensions for broadcasting

        # Parameters for "almost fair CRPS" loss. See https://arxiv.org/html/2412.15832v1
        self.coeff_eps = 1 - ((1 - alpha) / (n_members))
        self.averaging_coeff = 1 / (2 * n_members * (n_members - 1))

        # For n>2, will use pairwise distance to copmute [NxN] distance matrix
        # Diagonal elements of (prediciton - target) matrix are zeroed out to avoid double counting
        self.pdist = th.nn.PairwiseDistance(p=1)
        self.diag_mask = th.ones(self.n_members, self.n_members) - th.eye(
            self.n_members
        )  # Mask to zero out diagonal elements

    def setup(self, trainer):
        """
        pushes constants to cuda device
        """

        if len(trainer.output_variables) != len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)
        self.averaging_coeff = th.tensor(self.averaging_coeff, device=trainer.device)
        self.coeff_eps = th.tensor(self.coeff_eps, device=trainer.device)
        self.pdist = self.pdist.to(device=trainer.device)
        self.diag_mask = self.diag_mask.to(device=trainer.device)
        self.lsm_tensor = self.lsm_tensor.to(device=trainer.device)

    def _2member_crps(self, prediction, target, lsm_tensor):
        diff_target = th.abs(prediction - target.unsqueeze(0)).sum(
            dim=0
        )  # [B, F, T, C, H, W]
        diff_ensemble = th.abs(prediction[0] - prediction[1])  # [B, F, T, C, H, W]
        crps = self.averaging_coeff * (
            diff_target - self.coeff_eps * diff_ensemble
        )  # [B, F, T, C, H, W]
        crps *= lsm_tensor
        return crps

    def _pool(self, tensor, scale):
        shape = tensor.shape
        h, w = shape[-2:]
        pooled = th.nn.functional.avg_pool2d(
            tensor.reshape(shape[0], -1, h, w), scale, scale
        )
        return pooled.reshape(*shape[:-2], h // scale, w // scale)

    def _masked_pool(self, tensor, scale, mask):
        """
        Pools a tensor while excluding masked pixels

        Parameters
        ----------
        tensor: torch.Tensor
            The tensor to pool
        scale: int
            The scale factor for the pooling
        mask: torch.Tensor
            The mask to apply to the tensor, masked should only contain 1s and 0s

        Returns
        -------
        torch.Tensor
            The pooled tensor
        torch.Tensor
            The pooled mask
        """
        # Apply the mask to the tensor
        masked_values = tensor * mask

        # Compute the non-masked values for the pooled tensor
        pooled_values = self._pool(masked_values, scale)

        # pool the mask to use as a weight for the pooled tensor
        pooled_mask = self._pool(mask, scale)

        pooled_tensor = pooled_values / (pooled_mask + 1e-8)  # Avoid division by zero
        return pooled_tensor, pooled_mask

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the WeightedCRPSLoss
        Computes the CRPS loss for the model prediction and target.

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor shape [Cond*B, F, T, C, H, W] where Cond is the number of ensemble members
        target: torch.Tensor
            The target tensor shape [B, F, T, C, H, W]
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """

        # Unfold ensemble dimension from batch dimension to have shape [Cond, B, F, T, C, H, W]
        b, f, t, c, h, w = target.shape
        prediction = prediction.view(self.n_members, b, f, t, c, h, w)

        # checks for dimensions
        if not prediction.shape[1:] == target.shape:
            raise ValueError(
                f"Shape of prediction should match shape of target along non-ensemble dimensions, got {prediction.shape} and {target.shape}"
            )

        if not prediction.shape[0] == self.n_members:
            raise ValueError(
                f"Shape of prediction should have ensemble dimension of size {self.n_members}, got {prediction.shape[0]}"
            )

        n = self.n_members

        # Manual Cast
        prediction = prediction.to(th.float32)
        target = target.to(th.float32)

        # Apply channel weights across channel dims
        prediction *= self.loss_weights[None, None, None, None, :, None, None]
        target *= self.loss_weights[None, None, None, :, None, None]

        if n == 2:
            # Use faster explicit implementation
            crps = self._2member_crps(prediction, target, self.lsm_tensor)

            if average_channels:
                loss = crps.mean()
            else:
                loss = crps.mean(dim=(0, 1, 2, 4, 5))

            if self.mean_penalty > 0:
                # the fraction of valid pixels in land sea mask
                if self.masked_processing and self.lsm_tensor.numel() > 1:
                    valid_pixels = self.lsm_tensor.numel()
                else:
                    valid_pixels = f * h * w

                prediction_count = n * b * t * valid_pixels
                target_count = b * t * valid_pixels
                ens_global_means = (prediction * self.lsm_tensor).sum(
                    dim=(0, 1, 2, 3, 5, 6)
                ) / prediction_count
                target_global_means = (target * self.lsm_tensor).sum(
                    dim=(0, 1, 2, 4, 5)
                ) / target_count

                bias_penalty = self.mean_penalty * th.abs(
                    ens_global_means - target_global_means
                )
                if average_channels:
                    loss += bias_penalty.mean()
                else:
                    loss += bias_penalty

            # spatial multiscale loss
            if self.multiscale > 0.0:
                crps_scales = 0
                for scale in self.scales:
                    if self.masked_processing and self.lsm_tensor.numel() > 1:
                        masked_pred, masked_lsm = self._masked_pool(
                            prediction, scale, self.lsm_tensor
                        )
                        masked_tar, _ = self._masked_pool(
                            target, scale, self.lsm_tensor
                        )
                        crps_scale = self._2member_crps(
                            masked_pred, masked_tar, masked_lsm
                        )
                    else:
                        pred, tar, lsm = (
                            self._pool(prediction, scale),
                            self._pool(target, scale),
                            self._pool(self.lsm_tensor, scale),
                        )
                        crps_scale = self._2member_crps(pred, tar, lsm)

                    if average_channels:
                        crps_scale = crps_scale.mean()
                    else:
                        crps_scale = crps_scale.mean(dim=(0, 1, 2, 4, 5))
                    crps_scales += crps_scale

                crps_scales = crps_scales / len(self.scales)
                loss += self.multiscale * crps_scales

            return loss
        else:
            # Do mean penalty
            bias_penalty = 0
            if self.mean_penalty > 0:
                # the fraction of valid pixels in land sea mask
                if self.masked_processing and self.lsm_tensor.numel() > 1:
                    valid_pixels = self.lsm_tensor.numel()
                else:
                    valid_pixels = f * h * w

                prediction_count = n * b * t * valid_pixels
                target_count = b * t * valid_pixels
                ens_global_means = (prediction * self.lsm_tensor).sum(
                    dim=(0, 1, 2, 3, 5, 6)
                ) / prediction_count
                target_global_means = (target * self.lsm_tensor).sum(
                    dim=(0, 1, 2, 4, 5)
                ) / target_count

                bias_penalty = self.mean_penalty * th.abs(
                    ens_global_means - target_global_means
                )
                if average_channels:
                    bias_penalty = bias_penalty.mean()

            # zero out land and determine number of valid pixels
            if self.masked_processing and self.lsm_tensor.numel() > 1:
                prediction = prediction * self.lsm_tensor
                target = target * self.lsm_tensor
                valid_pixels = b * t * self.lsm_tensor.sum()
            else:
                valid_pixels = b * f * t * h * w

            # Use pairwise distance method
            if not average_channels:
                # Move channels to first dimension and exclude that dimension from the reductions
                prediction = prediction.permute(
                    4, 0, 1, 2, 3, 5, 6
                )  # [C, Cond, B, F, T, H, W]
                target = target.permute(3, 0, 1, 2, 4, 5)  # [C, B, F, T, H, W]

                prediction = prediction.reshape(c, n, -1)  # [C, Cond, ...]
                target = target.unsqueeze(1).reshape(
                    c, 1, -1
                )  # [C, 1, ...] (second dim will broadcast across ensemble)

                diff = self.pdist(prediction, target)  # [C, Cond]
                dist_matrix = self.pdist(
                    prediction.unsqueeze(1), prediction.unsqueeze(2)
                )  # [C, Cond, Cond]

                diff_terms = self.diag_mask[None, ...] * (
                    diff.unsqueeze(1) + diff.unsqueeze(2)
                )  # [C, Cond, Cond], diagonal elements zeroed out
                crps = (
                    self.averaging_coeff
                    * (diff_terms - self.coeff_eps * dist_matrix).sum(dim=(1, 2))
                    / valid_pixels
                )
            else:
                prediction = prediction.reshape(n, -1)
                target = target.unsqueeze(0).reshape(
                    1, -1
                )  # [1, ...] (first dim will broadcast across ensemble)
                diff = self.pdist(prediction, target)  # [Cond]
                dist_matrix = self.pdist(
                    prediction.unsqueeze(1), prediction.unsqueeze(0)
                )  # [Cond, Cond]

                diff_terms = self.diag_mask * (
                    diff.unsqueeze(0) + diff.unsqueeze(1)
                )  # [Cond, Cond], diagonal elements zeroed out
                crps = (
                    self.averaging_coeff
                    * (diff_terms - self.coeff_eps * dist_matrix).sum()
                    / valid_pixels
                )

            return crps + bias_penalty


class WeightedCRPSLossSpectral(th.nn.MSELoss):
    """
    Probabilistic loss function that allows for user defined weighting of variables when calculating CRPS.
    """

    def __init__(
        self,
        weights: Sequence = [],
        n_members: int = 2,
        alpha: float = 0.95,
        lambda_spec: float = 0.1,
        nside: int = 64,
        lmax: int = 3 * 64 - 1,
        mmax: int = 3 * 64 - 1,
        multiscale: float = 0.0,
        lsm_file: str = None,
        open_dict: dict = {"engine": "zarr"},
        selection_dict: dict = {"channel_c": "land_sea_mask"},
    ):
        """
        Parameters
        ----------
        weights: Sequence
            list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        n_members: int
            number of ensemble members in the model output
        alpha: float
            hyperparamter for approximating fair CRPS loss. between 0 and 1, 1 corresponds to a fair CRPS loss.
        lambda_spec: float
            weight for the spectral loss. Default is 0, no spectral loss is applied.
        nside: int
            nside for the HEALPix grid. Default is 64.
        lmax: int
            lmax for the SHT. Default is 3*nside - 1.
        mmax: int
            mmax for the SHT. Default is 3*nside - 1.
        multiscale: float, optional
            weight for the multiscale loss. Default is 0, no multiscale loss is applied.
        lsm_file: str
            path to the lsm file. Default is None, no lsm is applied.
        open_dict: dict
            dictionary of keyword arguments for xarray.open_dataset. Default is {"engine": "zarr"}.
        selection_dict: dict
            dictionary of keyword arguments for xarray.open_dataset. Default is {"channel_c": "land_sea_mask"}.
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        self.n_members = n_members
        self.device = None
        self.lambda_spec = lambda_spec
        self.multiscale = multiscale

        # Parameters for "almost fair CRPS" loss. See https://arxiv.org/html/2412.15832v1
        self.coeff_eps = 1 - ((1 - alpha) / (n_members))
        if n_members < 2:
            raise ValueError("n_members must be at least 2 for CRPS loss to be defined")
        else:
            self.n_members = n_members
        self.averaging_coeff = 1 / (2 * n_members * (n_members - 1))

        # SHT utils: transform, grid reordering, output indexing
        self.lmax = lmax
        self.mmax = mmax
        self.nside = nside
        self.sht = cuhpx.SHTCUDA(nside=nside, lmax=lmax, mmax=mmax, quad_weights="ring")
        src_grid = earth2grid.healpix.Grid(
            level=int(np.log2(nside)), pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
        )
        tar_grid = earth2grid.healpix.Grid(
            level=int(np.log2(nside)), pixel_order=earth2grid.healpix.PixelOrder.RING
        )
        self.reorder_to_ring = earth2grid.get_regridder(src_grid, tar_grid).to(
            th.float32
        )
        if self.multiscale > 0:
            self.scales = [200, 400, 800, 1600]  # in units of km
            self.isht = cuhpx.iSHTCUDA(
                nside=nside, lmax=lmax, mmax=mmax, quad_weights="ring"
            )
            self.reorder_from_ring = earth2grid.get_regridder(tar_grid, src_grid).to(
                th.float32
            )

        self.lsm_file = lsm_file
        if lsm_file is not None:
            self.lsm_ds = xr.open_dataset(lsm_file, **open_dict).constants.sel(
                selection_dict
            )
            self.lsm_tensor = 1 - th.tensor(
                np.expand_dims(self.lsm_ds.values, (0, 2, 3))
            )
        else:
            self.lsm_tensor = th.ones(
                1, 1, 1, 1, 1, 1
            )  # Spoof the tensor dimensions for broadcasting

        # For n>2, will use pairwise distance to copmute [NxN] distance matrix
        # Diagonal elements of (prediciton - target) matrix are zeroed out to avoid double counting
        self.pdist = th.nn.PairwiseDistance(p=1)
        self.diag_mask = th.ones(self.n_members, self.n_members) - th.eye(
            self.n_members
        )  # Mask to zero out diagonal elements

    def setup(self, trainer):
        """
        pushes constants to cuda device
        """

        if len(trainer.output_variables) != len(self.loss_weights):
            raise ValueError("Length of outputs and loss_weights is not the same!")

        self.loss_weights = self.loss_weights.to(device=trainer.device)
        self.averaging_coeff = th.tensor(self.averaging_coeff, device=trainer.device)
        self.coeff_eps = th.tensor(self.coeff_eps, device=trainer.device)
        self.reorder_to_ring = self.reorder_to_ring.to(device=trainer.device)
        self.sht = self.sht.to(device=trainer.device)
        self.pdist = self.pdist.to(device=trainer.device)
        self.diag_mask = self.diag_mask.to(device=trainer.device)

        if self.multiscale > 0:
            self.isht = self.isht.to(device=trainer.device)
            self.reorder_from_ring = self.reorder_from_ring.to(device=trainer.device)
        self.lsm_tensor = self.lsm_tensor.to(device=trainer.device)

    def _apply_sht(self, x, face_dim, return_abs=True):
        """Apply SHT to a tensor
        Reshape to [..., F*H*W], reorder to ring, apply SHT
        If return_abs is True, return the absolute value of the SHT (real**2 + imag**2)

        Parameters
        ----------
        x: torch.Tensor
            The tensor to apply SHT to
        face_dim: int
            The dimension of the tensor corrsponding to HEALPix faces
        return_abs: bool, optional
            Whether to return the absolute value of the SHT (real**2 + imag**2)
        """
        x = th.movedim(x, face_dim, -3)
        if x.shape[-3:] != (12, self.nside, self.nside):
            face, height, width = x.shape[-3:]
            raise ValueError(
                f"Shape of input tensor should be [..., 12, ..., {self.nside}, {self.nside}] with F (12) in position {face_dim},"
                f"got {x.shape}"
            )

        x = x.reshape(*x.shape[:-3], -1)
        x = self.reorder_to_ring(
            x.contiguous()
        )  # contiguous needed for channels first format in validation loop
        x = self.sht(x)
        if return_abs:
            x = x.real**2 + x.imag**2
        return x

    def _apply_isht(self, x, face_dim):
        """Apply inverse SHT to a tensor shape [..., l, m]
        Inverse transform, reorder from ring, Reshape to [..., F, H, W], move face dim appropriately
        """

        x = self.isht(x)  # [..., l, m] -> [..., F*H*W]
        x = self.reorder_from_ring(x)
        x = x.reshape(
            *x.shape[:-1], 12, self.nside, self.nside
        )  # [..., F*H*W] -> [..., F, H, W]
        x = th.movedim(x, -3, face_dim)  # [..., F, H, W] -> [..., F, ..., H, W]
        return x

    def _l_filter(self, scale, device="cuda"):
        """Return a spherical gaussian filter of scale `scale` (in units of km)"""
        scale_radians = scale / 6371.0
        ell = th.arange(self.lmax, device=device, dtype=th.float32)
        return th.exp(-0.5 * ell * (ell + 1) * (scale_radians**2))

    def forward(self, prediction, target, average_channels=True):
        """
        Forward pass of the WeightedCRPSLoss
        Computes the CRPS loss for the model prediction and target.

        Parameters
        ----------
        prediction: torch.Tensor
            The prediction tensor shape [Cond*B, F, T, C, H, W] where Cond is the number of ensemble members
        target: torch.Tensor
            The target tensor shape [B, F, T, C, H, W]
        average_channels: bool, optional
            whether the mean of the channels should be taken
        """

        # Unfold ensemble dimension from batch dimension to have shape [Cond, B, F, T, C, H, W]
        b, f, t, c, h, w = target.shape
        prediction = prediction.view(self.n_members, b, f, t, c, h, w)

        # checks for dimensions
        if not prediction.shape[1:] == target.shape:
            raise ValueError(
                f"Shape of prediction should match shape of target along non-ensemble dimensions, got {prediction.shape} and {target.shape}"
            )

        if not prediction.shape[0] == self.n_members:
            raise ValueError(
                f"Shape of prediction should have ensemble dimension of size {self.n_members}, got {prediction.shape[0]}"
            )

        n = self.n_members

        # Manual cast
        prediction = prediction.to(th.float32)
        target = target.to(th.float32)

        # Apply channel weights across channel dims
        prediction *= self.loss_weights[None, None, None, None, :, None, None]
        target *= self.loss_weights[None, None, None, :, None, None]

        if n == 2:
            # Use faster explicit implementation
            diff_target = th.abs(prediction - target.unsqueeze(0)).sum(
                dim=0
            )  # [B, F, T, C, H, W]
            diff_ensemble = th.abs(prediction[0] - prediction[1])  # [B, F, T, C, H, W]
            crps = self.averaging_coeff * (
                diff_target - self.coeff_eps * diff_ensemble
            )  # [B, F, T, C, H, W]

            if average_channels:
                loss = crps.mean()
            else:
                loss = crps.mean(dim=(0, 1, 2, 4, 5))

            if self.lambda_spec > 0:
                with th.amp.autocast("cuda", enabled=False):
                    # # Reorder predictions: [N, B, F, T, C, H, W] -> [N, B, T, C, F*H*W]
                    # pred_ring = self.reorder_to_ring(prediction.permute(0, 1, 3, 4, 2, 5, 6).reshape(n, b, t, c, f*h*w))

                    # # Reorder targets: [B, F, T, C, H, W] -> [B, T, C, F*H*W]
                    # tar_ring = self.reorder_to_ring(target.permute(0, 2, 3, 1, 4, 5).reshape(b, t, c, f*h*w))

                    # # Compute SHT of predictions and targets
                    # sht_pred = self.sht(pred_ring) # [N, B, T, C, l, m]
                    # sht_tar = self.sht(tar_ring) # [B, T, C, l, m]
                    # sht_pred = sht_pred.real ** 2 + sht_pred.imag ** 2
                    # sht_tar = sht_tar.real ** 2 + sht_tar.imag ** 2

                    sht_pred = self._apply_sht(prediction, face_dim=2, return_abs=True)
                    sht_tar = self._apply_sht(target, face_dim=1, return_abs=True)

                    diff_sht_target = th.abs(sht_pred - sht_tar.unsqueeze(0)).sum(
                        dim=(0, 4, 5)
                    )  # [B, T, C]
                    diff_sht_ensemble = th.abs(sht_pred[0] - sht_pred[1]).sum(
                        dim=(-1, -2)
                    )  # [B, T, C]
                    crps_sht = self.averaging_coeff * (
                        diff_sht_target - self.coeff_eps * diff_sht_ensemble
                    )  # [B, T, C]

                    # Compute spectral afCRPS
                    if average_channels:
                        spec_loss = crps_sht.mean()
                    else:
                        spec_loss = crps_sht.mean(dim=(0, 1))

                loss += self.lambda_spec * spec_loss

            if self.multiscale > 0:
                for scale in self.scales:
                    l_filter = self._l_filter(scale, device=prediction.device)
                    with th.amp.autocast("cuda", enabled=False):
                        sht_pred = self._apply_sht(
                            prediction, face_dim=2, return_abs=False
                        )
                        sht_tar = self._apply_sht(target, face_dim=1, return_abs=False)

                        l_filter_pred = l_filter[
                            None, None, None, None, :, None
                        ]  # [1, 1, 1, 1, lmax, 1]
                        l_filter_tar = l_filter[
                            None, None, None, :, None
                        ]  # [1, 1, 1, lmax, 1]

                        pred_smooth = self._apply_isht(
                            l_filter_pred * sht_pred, face_dim=2
                        )
                        tar_smooth = self._apply_isht(
                            l_filter_tar * sht_tar, face_dim=1
                        )

                        diff_target = th.abs(pred_smooth - tar_smooth.unsqueeze(0)).sum(
                            dim=0
                        )  # [B, F, T, C, H, W]
                        diff_ensemble = th.abs(
                            pred_smooth[0] - pred_smooth[1]
                        )  # [B, F, T, C, H, W]
                        crps = self.averaging_coeff * (
                            diff_target - self.coeff_eps * diff_ensemble
                        )  # [B, F, T, C, H, W]

                        crps *= self.lsm_tensor

                        if average_channels:
                            loss += self.multiscale * crps.mean()
                        else:
                            loss += self.multiscale * crps.mean(dim=(0, 1, 2, 4, 5))

            return loss

        else:
            # Use pairwise distance method
            if not average_channels:
                # Move channels to first dimension and exclude that dimension from the reductions
                prediction = prediction.permute(
                    4, 0, 1, 2, 3, 5, 6
                )  # [C, Cond, B, F, T, H, W]
                target = target.permute(3, 0, 1, 2, 4, 5)  # [C, B, F, T, H, W]

                pred = prediction.reshape(c, n, -1)  # [C, Cond, ...]
                tar = target.unsqueeze(1).reshape(
                    c, 1, -1
                )  # [C, 1, ...] (second dim will broadcast across ensemble)

                diff = self.pdist(pred, tar)  # [C, Cond]
                dist_matrix = self.pdist(
                    pred.unsqueeze(1), pred.unsqueeze(2)
                )  # [C, Cond, Cond]

                diff_terms = self.diag_mask[None, ...] * (
                    diff.unsqueeze(1) + diff.unsqueeze(2)
                )  # [C, Cond, Cond], diagonal elements zeroed out
                loss = (
                    self.averaging_coeff
                    * (diff_terms - self.coeff_eps * dist_matrix).sum(dim=(1, 2))
                    / (b * f * t * h * w)
                )

                if self.lambda_spec > 0:
                    with th.amp.autocast("cuda", enabled=False):
                        # # Reorder predictions: [C, Cond, B, F, T, H, W] -> [C, Cond, B, T, F*H*W]
                        # pred_ring = self.reorder_to_ring(prediction.permute(0, 1, 2, 4, 3, 5, 6).reshape(c, n, b, t, f*h*w))

                        # # Reorder targets: [C, B, F, T, H, W] -> [C, B, T, F*H*W]
                        # tar_ring = self.reorder_to_ring(target.permute(0, 1, 3, 2, 4, 5).reshape(c, b, t, f*h*w))

                        # # Compute SHT of predictions and targets
                        # sht_pred = self.sht(pred_ring).reshape(c, n, -1) # [C, Cond, B, T, l, m] -> [C, Cond, ...]
                        # sht_tar = self.sht(tar_ring).unsqueeze(1).reshape(c, 1, -1) # [C, B, T, l, m] -> [C, 1, ...] (second dim will broadcast across ensemble)
                        # sht_pred = sht_pred.real ** 2 + sht_pred.imag ** 2
                        # sht_tar = sht_tar.real ** 2 + sht_tar.imag ** 2

                        sht_pred = self._apply_sht(
                            prediction, face_dim=3, return_abs=True
                        ).reshape(c, n, -1)
                        sht_tar = (
                            self._apply_sht(target, face_dim=2, return_abs=True)
                            .unsqueeze(1)
                            .reshape(c, 1, -1)
                        )

                        diff = self.pdist(sht_pred, sht_tar)  # [C, Cond]
                        dist_matrix = self.pdist(
                            sht_pred.unsqueeze(1), sht_pred.unsqueeze(2)
                        )  # [C, Cond, Cond]

                        diff_terms = self.diag_mask[None, ...] * (
                            diff.unsqueeze(1) + diff.unsqueeze(2)
                        )  # [C, Cond, Cond], diagonal elements zeroed out
                        loss += (
                            self.lambda_spec
                            * self.averaging_coeff
                            * (diff_terms - self.coeff_eps * dist_matrix).sum(
                                dim=(1, 2)
                            )
                            / (b * t * self.lmax * self.mmax)
                        )

            else:
                pred = prediction.reshape(n, -1)
                tar = target.unsqueeze(0).reshape(
                    1, -1
                )  # [1, ...] (first dim will broadcast across ensemble)
                diff = self.pdist(pred, tar)  # [Cond]
                dist_matrix = self.pdist(
                    pred.unsqueeze(1), pred.unsqueeze(0)
                )  # [Cond, Cond]

                diff_terms = self.diag_mask * (
                    diff.unsqueeze(0) + diff.unsqueeze(1)
                )  # [Cond, Cond], diagonal elements zeroed out
                loss = (
                    self.averaging_coeff
                    * (diff_terms - self.coeff_eps * dist_matrix).sum()
                    / (b * f * c * t * h * w)
                )

                if self.lambda_spec > 0:
                    with th.amp.autocast("cuda", enabled=False):
                        # # Reorder predictions: [Cond, B, F, T, C, H, W] -> [Cond, B, T, C, F*H*W]
                        # pred_ring = self.reorder_to_ring(prediction.permute(0, 1, 3, 4, 2, 5, 6).reshape(n, b, t, c, f*h*w))

                        # # Reorder targets: [B, F, T, C, H, W] -> [B, T, C, F*H*W]
                        # tar_ring = self.reorder_to_ring(target.permute(0, 2, 3, 1, 4, 5).reshape(b, t, c, f*h*w))

                        # # Compute SHT of predictions and targets
                        # sht_pred = self.sht(pred_ring).reshape(n, -1) # [Cond, B, T, C, l, m] -> [Cond, ...]
                        # sht_tar = self.sht(tar_ring).unsqueeze(0).reshape(1, -1) # [B, T, C, l, m] -> [1, ...] (first dim will broadcast across ensemble)
                        # sht_pred = sht_pred.real ** 2 + sht_pred.imag ** 2
                        # sht_tar = sht_tar.real ** 2 + sht_tar.imag ** 2
                        sht_pred = self._apply_sht(
                            prediction, face_dim=2, return_abs=True
                        ).reshape(n, -1)
                        sht_tar = (
                            self._apply_sht(target, face_dim=1, return_abs=True)
                            .unsqueeze(0)
                            .reshape(1, -1)
                        )

                        diff = self.pdist(sht_pred, sht_tar)  # [Cond]
                        dist_matrix = self.pdist(
                            sht_pred.unsqueeze(1), sht_pred.unsqueeze(0)
                        )  # [Cond, Cond]

                        diff_terms = self.diag_mask * (
                            diff.unsqueeze(0) + diff.unsqueeze(1)
                        )  # [Cond, Cond], diagonal elements zeroed out
                        loss += (
                            self.lambda_spec
                            * self.averaging_coeff
                            * (diff_terms - self.coeff_eps * dist_matrix).sum()
                            / (b * t * self.lmax * self.mmax)
                        )

            return loss

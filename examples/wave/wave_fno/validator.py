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

import matplotlib.pyplot as plt
from torch import FloatTensor
from physicsnemo.utils.logging import LaunchLogger


class WaveValidator:
    """Grid Validator for wave equation predictions.

    Compares model prediction against ground truth, computes loss, and logs
    a side-by-side visualization of the initial condition, truth, prediction,
    and point-wise error.

    Parameters
    ----------
    loss_fun : torch.nn.Module
        Loss function for validation error
    font_size : float, optional
        Font size for plots
    """

    def __init__(self, loss_fun, font_size: float = 28.0):
        self.criterion = loss_fun
        self.font_size = font_size
        self.headers = ("initial u(0)", "truth u(T)", "prediction", "abs error")

    def compare(
        self,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
        logger: LaunchLogger,
    ) -> float:
        """Compare prediction to ground truth and log visualization.

        Parameters
        ----------
        invar : FloatTensor
            Initial condition input
        target : FloatTensor
            Ground truth solution at time T
        prediction : FloatTensor
            Model prediction
        step : int
            Current epoch/step for labeling
        logger : LaunchLogger
            Logger for figure output

        Returns
        -------
        float
            Validation loss
        """
        loss = self.criterion(prediction, target)

        # Extract first sample for plotting
        invar_np = invar.cpu().numpy()[0, 0, :, :]
        target_np = target.cpu().numpy()[0, 0, :, :]
        pred_np = prediction.detach().cpu().numpy()[0, 0, :, :]
        error_np = abs(pred_np - target_np)

        plt.close("all")
        plt.rcParams.update({"font.size": self.font_size})
        fig, ax = plt.subplots(1, 4, figsize=(15 * 4, 15), sharey=True)
        im = []
        im.append(ax[0].imshow(invar_np, cmap="RdBu_r"))
        im.append(ax[1].imshow(target_np, cmap="RdBu_r"))
        im.append(ax[2].imshow(pred_np, cmap="RdBu_r"))
        im.append(ax[3].imshow(error_np, cmap="hot"))

        for ii in range(len(im)):
            fig.colorbar(
                im[ii], ax=ax[ii], location="bottom", fraction=0.046, pad=0.04
            )
            ax[ii].set_title(self.headers[ii])

        logger.log_figure(figure=fig, artifact_file=f"validation_step_{step:03d}.png")

        return loss

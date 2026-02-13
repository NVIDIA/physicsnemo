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
import threading
import os


class GridValidator:
    """Grid Validator

    The validator compares model output and target, inverts normalisation and plots a sample

    Parameters
    ----------
    loss_fun : MSELoss
        loss function for assessing validation error
    norm : Dict, optional
        mean and standard deviation for each channel to normalise input and target
    font_size : float, optional
        font size used in figures

    """

    def __init__(
        self,
        font_size: float = 28.0,
        output_dir: str = "./plots/",
    ):
        self.font_size = font_size
        self.headers = ("true", "prediction", "error")
        self._plot_thread = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_figure(
        self, target: FloatTensor, prediction: FloatTensor, step: int, resolution: int
    ):
        target = target.cpu().numpy().reshape(-1, resolution, resolution)[0, :, :]
        prediction = (
            prediction.reshape(-1, resolution, resolution)
            .detach()
            .cpu()
            .numpy()[0, :, :]
        )

        plt.close("all")
        plt.rcParams.update({"font.size": self.font_size})
        fig, ax = plt.subplots(1, 3, figsize=(15 * 3, 15), sharey=True)
        im = []
        im.append(ax[0].imshow(target))
        im.append(ax[1].imshow(prediction))
        im.append(ax[2].imshow((prediction - target)))

        for ii in range(len(im)):
            fig.colorbar(im[ii], ax=ax[ii], location="bottom", fraction=0.046, pad=0.04)
            ax[ii].set_title(self.headers[ii])

        plt.savefig(f"{self.output_dir}/validation_step_{step:03d}.png")

    def _plot_figure_thread(self, target, prediction, step, resolution):
        self.plot_figure(target, prediction, step, resolution)

    def make_plot(
        self,
        prediction: FloatTensor,
        target: FloatTensor,
        step: int,
        resolution: int,
    ) -> float:
        """compares model output, target and plots everything

        Parameters
        ----------
        invar : FloatTensor
            input to model
        target : FloatTensor
            ground truth
        prediction : FloatTensor
            model output
        step : int
            iteration counter
        logger : LaunchLogger
            logger to which figure is passed

        Returns
        -------
        float
            validation error
        """

        # Wait for previous plot thread if still running
        if self._plot_thread is not None and self._plot_thread.is_alive():
            self._plot_thread.join()

        # Start new plot thread
        self._plot_thread = threading.Thread(
            target=self._plot_figure_thread,
            args=(target, prediction, step, resolution),
        )
        self._plot_thread.start()

        return

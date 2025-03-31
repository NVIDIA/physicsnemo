# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import torch
import os.path
import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Dict
from torch import FloatTensor, Tensor
from torch.nn import MSELoss
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.datapipes.benchmarks.kernels.initialization import (
    init_uniform_random_4d,
)
from physicsnemo.datapipes.benchmarks.kernels.utils import (
    fourier_to_array_batched_2d,
    threshold_3d,
)


class NestedDarcyDataset:
    """Nested Darcy Dataset

    A Dataset class for loading nested Darcy data generated with generate_nested_darcy.py
    during training. The method takes care of loading the correct level and associated
    information from its parent level.

    Parameters
    ----------
    data_path : str
        Path to numpy dict file containing the data
    level : int, optional
        Refinement level which shall be loaded
    norm : Dict, optional
        mean and standard deviation for each channel to normalise input and target
    log : PythonLogger
        logger for command line output

    """

    def __init__(
        self,
        mode: str,
        data_path: str = None,
        model_name: str = None,
        norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        log: PythonLogger = None,
        parent_prediction: FloatTensor = None,
    ) -> None:
        self.dist = DistributedManager()
        self.data_path = os.path.abspath(data_path)
        self.model_name = model_name
        # self.level = level
        self.norm = norm
        self.log = log
        self.mode = mode
        assert self.mode in [
            "train",
            "eval",
        ], "mode in NestedDarcyDataset must be train or eval."

        if mode == "eval" and int(self.model_name[-1]) > 0:
            assert (
                parent_prediction is not None
            ), f"pass parent result to evaluate level {int(self.model_name[-1])}"
            parent_prediction = parent_prediction.detach().cpu().numpy()
        self.load_dataset(parent_prediction)

    def load_dataset(self, parent_prediction: FloatTensor = None) -> None:
        try:
            contents = np.load(self.data_path, allow_pickle=True).item()
        except IOError as err:
            self.log.error(f"Unable to find or load file {self.data_path}")
            exit()

        # load input varibales, copy to device and normalise
        dat = contents["fields"]
        self.ref_fac = contents["meta"]["ref_fac"]
        self.buffer = contents["meta"]["buffer"]
        self.fine_res = contents["meta"]["fine_res"]

        mod = self.model_name
        perm, darc, par_pred, self.position = [], [], [], {}
        for id, samp in dat.items():
            if int(mod[-1]) > 0:
                self.position[id] = {}
            for jd, fields in samp[mod].items():
                perm.append(fields["permeability"][None, None, ...])
                darc.append(fields["darcy"][None, None, ...])

                if int(mod[-1]) > 0:  # if not on global level
                    xy_size = perm[-1].shape[-1]
                    pos = fields["pos"]
                    self.position[id][jd] = pos
                    if self.mode == "eval":
                        parent = parent_prediction[int(id), 0, ...]
                    elif self.mode == "train":
                        parent = (
                            samp[f"ref{int(mod[-1])-1}"]["0"]["darcy"]
                            - self.norm["darcy"][0]
                        ) / self.norm["darcy"][1]
                    par_pred.append(
                        parent[
                            pos[0] : pos[0] + xy_size,
                            pos[1] : pos[1] + xy_size,
                        ][None, None, ...]
                    )

        perm = (
            np.concatenate(perm, axis=0) - self.norm["permeability"][0]
        ) / self.norm["permeability"][1]
        darc = (np.concatenate(darc, axis=0) - self.norm["darcy"][0]) / self.norm[
            "darcy"
        ][1]

        if int(mod[-1]) > 0:
            par_pred = np.concatenate(par_pred, axis=0)
            perm = np.concatenate((par_pred, perm), axis=1)

        self.invars = torch.from_numpy(perm).float().to(self.dist.device)
        self.outvars = torch.from_numpy(darc).float().to(self.dist.device)

        self.length = self.invars.size()[0]

    def __getitem__(self, idx: int):
        return {"permeability": self.invars[idx, ...], "darcy": self.outvars[idx, ...]}

    def __len__(self):
        return self.length


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
        loss_fun: MSELoss,
        norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        font_size: float = 28.0,
    ) -> None:
        self.norm = norm
        self.criterion = loss_fun
        self.font_size = font_size
        self.headers = ("invar", "truth", "prediction", "relative error")

    def compare(
        self,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
        logger: LaunchLogger,
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
        loss = self.criterion(prediction, target)
        norm = self.norm

        # pick first sample from batch
        invar = invar * norm["permeability"][1] + norm["permeability"][0]
        target = target * norm["darcy"][1] + norm["darcy"][0]
        prediction = prediction * norm["darcy"][1] + norm["darcy"][0]
        invar = invar.cpu().numpy()[0, -1, :, :]
        target = target.cpu().numpy()[0, 0, :, :]
        prediction = prediction.detach().cpu().numpy()[0, 0, :, :]

        plt.close("all")
        plt.rcParams.update({"font.size": self.font_size})
        fig, ax = plt.subplots(1, 4, figsize=(15 * 3.5, 15), sharey=True)
        im = []
        im.append(ax[0].imshow(invar))
        im.append(ax[1].imshow(target))
        im.append(ax[2].imshow(prediction))
        im.append(ax[3].imshow((prediction - target) / norm["darcy"][1]))

        for ii in range(len(im)):
            fig.colorbar(im[ii], ax=ax[ii], location="bottom", fraction=0.046, pad=0.04)
            ax[ii].set_title(self.headers[ii])

        logger.log_figure(figure=fig, artifact_file=f"validation_step_{step:03d}.png")

        return loss


def PlotNestedDarcy(dat: dict, idx: int) -> None:
    """Plot fields from the nested Darcy case

    Parameters
    ----------
    dat : dict
        dictionary containing fields
    target : FloatTensor
        index of example to plot
    """
    fields = dat[str(idx)]
    n_insets = len(fields["ref1"])

    fig, ax = plt.subplots(n_insets + 1, 4, figsize=(20, 5 * (n_insets + 1)))

    vmin = fields["ref0"]["0"]["darcy"].min()
    vmax = fields["ref0"]["0"]["darcy"].max()

    ax[0, 0].imshow(fields["ref0"]["0"]["permeability"])
    ax[0, 0].set_title("permeability glob")
    ax[0, 1].imshow(fields["ref0"]["0"]["darcy"], vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("darcy glob")
    ax[0, 2].axis("off")
    ax[0, 3].axis("off")

    for ii in range(n_insets):
        loc = fields["ref1"][str(ii)]
        inset_size = loc["darcy"].shape[1]
        ax[ii + 1, 0].imshow(loc["permeability"])
        ax[ii + 1, 0].set_title(f"permeability fine {ii}")
        ax[ii + 1, 1].imshow(loc["darcy"], vmin=vmin, vmax=vmax)
        ax[ii + 1, 1].set_title(f"darcy fine {ii}")
        ax[ii + 1, 2].imshow(
            fields["ref0"]["0"]["permeability"][
                loc["pos"][0] : loc["pos"][0] + inset_size,
                loc["pos"][1] : loc["pos"][1] + inset_size,
            ]
        )
        ax[ii + 1, 2].set_title(f"permeability zoomed {ii}")
        ax[ii + 1, 3].imshow(
            fields["ref0"]["0"]["darcy"][
                loc["pos"][0] : loc["pos"][0] + inset_size,
                loc["pos"][1] : loc["pos"][1] + inset_size,
            ],
            vmin=vmin,
            vmax=vmax,
        )
        ax[ii + 1, 3].set_title(f"darcy zoomed {ii}")

    fig.tight_layout()
    plt.savefig(f"sample_{idx:02d}.png")
    plt.close()


@wp.kernel
def fourier_to_array_batched_2d_cropped(
    array: wp.array3d(dtype=float),
    fourier: wp.array4d(dtype=float),
    nr_freq: int,
    lx: int,
    ly: int,
    bounds: wp.array3d(dtype=int),
    fill_val: int,
):  # pragma: no cover
    """Array of Fourier amplitudes to batched 2d spatial array

    Parameters
    ----------
    array : wp.array3d
        Spatial array
    fourier : wp.array4d
        Array of Fourier amplitudes
    nr_freq : int
        Number of frequencies in Fourier array
    lx : int
        Grid size x
    ly : int
        Grid size y
    x_start : int
        lowest x-index
    y_start : int
        lowest y-index
    """
    b, p, x, y = wp.tid()

    if bounds[b, p, 0] == fill_val:
        return

    x += bounds[b, p, 0]
    y += bounds[b, p, 1]

    array[b, x, y] = 0.0
    dx = 6.28318 / wp.float32(lx)
    dy = 6.28318 / wp.float32(ly)
    rx = dx * wp.float32(x)
    ry = dy * wp.float32(y)
    for i in range(nr_freq):
        for j in range(nr_freq):
            ri = wp.float32(i)
            rj = wp.float32(j)
            ss = fourier[0, b, i, j] * wp.sin(ri * rx) * wp.sin(rj * ry)
            cs = fourier[1, b, i, j] * wp.cos(ri * rx) * wp.sin(rj * ry)
            sc = fourier[2, b, i, j] * wp.sin(ri * rx) * wp.cos(rj * ry)
            cc = fourier[3, b, i, j] * wp.cos(ri * rx) * wp.cos(rj * ry)
            wp.atomic_add(
                array, b, x, y, 1.0 / (wp.float32(nr_freq) ** 2.0) * (ss + cs + sc + cc)
            )


class DarcyInset2D(Darcy2D):
    """2D Darcy flow benchmark problem datapipe.

    This datapipe continuously generates solutions to the 2D Darcy equation with variable
    permeability. All samples are generated on the fly and is meant to be a benchmark
    problem for testing data driven models. Permeability is drawn from a random Fourier
    series and threshold it to give a piecewise constant function. The solution is obtained
    using a GPU enabled multi-grid Jacobi iterative method.

    Parameters
    ----------
    resolution : int, optional
        Resolution to run simulation at, by default 256
    batch_size : int, optional
        Batch size of simulations, by default 64
    nr_permeability_freq : int, optional
        Number of frequencies to use for generating random permeability. Higher values
        will give higher freq permeability fields., by default 5
    max_permeability : float, optional
        Max permeability, by default 2.0
    min_permeability : float, optional
        Min permeability, by default 0.5
    max_iterations : int, optional
        Maximum iterations to use for each multi-grid, by default 30000
    convergence_threshold : float, optional
        Solver L-Infinity convergence threshold, by default 1e-6
    iterations_per_convergence_check : int, optional
        Number of Jacobi iterations to run before checking convergence, by default 1000
    nr_multigrids : int, optional
        Number of multi-grid levels, by default 4
    normaliser : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary with keys `permeability` and `darcy`. The values for these keys are two floats corresponding to mean and std `(mean, std)`.
    device : Union[str, torch.device], optional
        Device for datapipe to run place data on, by default "cuda"

    Raises
    ------
    ValueError
        Incompatable multi-grid and resolution settings
    """

    def __init__(
        self,
        resolution: int = 256,
        batch_size: int = 64,
        nr_permeability_freq: int = 5,
        max_permeability: float = 2.0,
        min_permeability: float = 0.5,
        max_iterations: int = 30000,
        convergence_threshold: float = 1e-6,
        iterations_per_convergence_check: int = 1000,
        nr_multigrids: int = 4,
        normaliser: Union[Dict[str, Tuple[float, float]], None] = None,
        device: Union[str, torch.device] = "cuda",
        max_n_insets: int = 3,
        fine_res: int = 32,
        fine_permeability_freq: int = 10,
        min_offset: int = 48,
        ref_fac: int = None,
        min_dist_frac: float = 1.7,
        fill_val: int = -99999,
    ):
        super().__init__(
            resolution,
            batch_size,
            nr_permeability_freq,
            max_permeability,
            min_permeability,
            max_iterations,
            convergence_threshold,
            iterations_per_convergence_check,
            nr_multigrids,
            normaliser,
            device,
        )

        self.max_n_insets = max_n_insets
        self.fine_res = fine_res
        self.fine_freq = fine_permeability_freq
        self.ref_fac = ref_fac
        assert (
            resolution % self.ref_fac == 0
        ), "simulation res must be multiple of ref_fac"

        # force inset on coarse grid
        if not min_offset % self.ref_fac == 0:
            min_offset += self.ref_fac - min_offset % self.ref_fac
        self.beg_min = min_offset
        self.beg_max = resolution - min_offset - fine_res - self.ref_fac
        self.bounds = None
        self.min_dist_frac = min_dist_frac
        self.fill_val = fill_val

        assert (
            self.max_n_insets <= 3
        ), f"at most 3 insets supported, change max_n_insets accordingly"
        assert (self.beg_max - self.beg_min) % ref_fac == 0, "lsdhfgn3x!!!!"

    def initialize_batch(self) -> None:
        """Initializes arrays for new batch of simulations"""

        # initialize permeability
        self.permeability.zero_()
        seed = np.random.randint(np.iinfo(np.uint64).max, dtype=np.uint64)
        wp.launch(
            kernel=init_uniform_random_4d,
            dim=self.fourier_dim,
            inputs=[self.rand_fourier, -1.0, 1.0, seed],
            device=self.device,
        )
        wp.launch(
            kernel=fourier_to_array_batched_2d,
            dim=self.dim,
            inputs=[
                self.permeability,
                self.rand_fourier,
                self.nr_permeability_freq,
                self.resolution,
                self.resolution,
            ],
            device=self.device,
        )

        rr = np.random.randint(
            low=0,
            high=(self.beg_max - self.beg_min) // self.ref_fac,
            size=(self.batch_size, self.max_n_insets, 2),
        )
        n_insets = np.random.randint(
            low=1,
            high=rr.shape[1] + 1,
            size=(self.batch_size,),
        )

        # check that regions do not overlap and have distance
        min_dist = self.min_dist_frac * self.fine_res // self.ref_fac + 1
        print("adjusting inset positions")
        for ib in range(self.batch_size):
            if n_insets[ib] <= 1:
                rr[ib, 1:, :] = self.fill_val
                continue
            else:
                while (
                    abs(rr[ib, 0, 0] - rr[ib, 1, 0]) < min_dist
                    and abs(rr[ib, 0, 1] - rr[ib, 1, 1]) < min_dist
                ):
                    rr[ib, 0, :] = np.random.randint(
                        low=0,
                        high=(self.beg_max - self.beg_min) // self.ref_fac,
                        size=(2,),
                    )
                    rr[ib, 1, :] = np.random.randint(
                        low=0,
                        high=(self.beg_max - self.beg_min) // self.ref_fac,
                        size=(2,),
                    )
            if n_insets[ib] <= 2:
                rr[ib, 2:, :] = self.fill_val
                continue
            else:
                while (
                    abs(rr[ib, 0, 0] - rr[ib, 2, 0]) < min_dist
                    and abs(rr[ib, 0, 1] - rr[ib, 2, 1]) < min_dist
                ) or (
                    abs(rr[ib, 1, 0] - rr[ib, 2, 0]) < min_dist
                    and abs(rr[ib, 1, 1] - rr[ib, 2, 1]) < min_dist
                ):
                    rr[ib, 2, :] = np.random.randint(
                        low=0,
                        high=(self.beg_max - self.beg_min) // self.ref_fac,
                        size=(2,),
                    )
        print("done")

        rr = np.where(rr != self.fill_val, (rr * self.ref_fac) + self.beg_min, rr)
        self.bounds = wp.array(rr, dtype=int, device=self.device)

        wp.launch(
            kernel=fourier_to_array_batched_2d_cropped,
            dim=(self.batch_size, self.bounds.shape[1], self.fine_res, self.fine_res),
            inputs=[
                self.permeability,
                self.rand_fourier,
                self.fine_freq,
                self.fine_res,
                self.fine_res,
                self.bounds,
                self.fill_val,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=threshold_3d,
            dim=self.dim,
            inputs=[
                self.permeability,
                0.0,
                self.min_permeability,
                self.max_permeability,
            ],
            device=self.device,
        )

        # zero darcy arrays
        self.darcy0.zero_()
        self.darcy1.zero_()

    def batch_generator(self) -> Tuple[Tensor, Tensor]:
        # run simulation
        self.generate_batch()

        # convert warp arrays to pytorch
        permeability = wp.to_torch(self.permeability)
        darcy = wp.to_torch(self.darcy0)

        # add channel dims
        permeability = torch.unsqueeze(permeability, axis=1)
        darcy = torch.unsqueeze(darcy, axis=1)

        # crop edges by 1 from multi-grid
        permeability = permeability[:, :, : self.resolution, : self.resolution]
        darcy = darcy[:, :, : self.resolution, : self.resolution]

        # normalize values
        if self.normaliser is not None:
            permeability = (
                permeability - self.normaliser["permeability"][0]
            ) / self.normaliser["permeability"][1]
            darcy = (darcy - self.normaliser["darcy"][0]) / self.normaliser["darcy"][1]

        # CUDA graphs static copies
        if self.output_k is None:
            self.output_k = permeability
            self.output_p = darcy
        else:
            self.output_k.data.copy_(permeability)
            self.output_p.data.copy_(darcy)

        return {"permeability": self.output_k, "darcy": self.output_p}

    def __iter__(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Yields
        ------
        Iterator[Tuple[Tensor, Tensor, Tensor]]
            Infinite iterator that returns a batch of (permeability, darcy pressure)
            fields of size [batch, resolution, resolution]
        """
        # infinite generator
        while True:
            batch = self.batch_generator()
            batch["inset_pos"] = wp.to_torch(self.bounds)
            yield batch

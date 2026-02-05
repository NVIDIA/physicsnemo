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

from os.path import isdir
from os import mkdir
import numpy as np
from utils import DarcyInset2D, PlotNestedDarcy


def nested_darcy_generator() -> None:
    """Dataset Generator for the nested Darcy Problem

    This script generates the training, validation and out-of-sample data sets
    for the nested FNO problem and stores them in ./data, where trainer and
    inferencer will find it.
    """
    out_dir = "./data/"
    file_names = ["training_data.npy", "validation_data.npy", "out_of_sample.npy"]
    sample_size = [8192, 2048, 2048]
    max_batch_size = 128
    resolution = 1024
    glob_res = 256
    fine_res = 128
    buffer = 32
    permea_freq = 3
    max_n_insets = 2
    fine_permeability_freq = 2
    min_dist_frac = 1.8
    device = "cuda"
    n_plots = 10
    fill_val = -99999

    perm_norm = (0.0, 1.0)
    darc_norm = (0.0, 1.0)

    if not isdir(out_dir):
        mkdir(out_dir)

    assert resolution % glob_res == 0, "resolution needs to be multiple of glob_res"
    ref_fac = resolution // glob_res
    inset_size = fine_res + 2 * buffer
    min_offset = (fine_res * (ref_fac - 1) + 1) // 2 + buffer * ref_fac

    # force inset on coarse grid
    if not min_offset % ref_fac == 0:
        min_offset += ref_fac - min_offset % ref_fac

    for dset in range(len(file_names)):
        # compute batch size and number of iterations
        batch_size = min(max_batch_size, sample_size[dset])
        nr_iterations = (sample_size[dset] - 1) // max_batch_size + 1

        datapipe = DarcyInset2D(
            resolution=resolution,
            batch_size=batch_size,
            nr_permeability_freq=permea_freq,
            max_permeability=2.0,
            min_permeability=0.5,
            max_iterations=30000,
            iterations_per_convergence_check=10,
            nr_multigrids=3,
            normaliser={"permeability": perm_norm, "darcy": darc_norm},
            device=device,
            max_n_insets=max_n_insets,
            fine_res=fine_res,
            fine_permeability_freq=fine_permeability_freq,
            min_offset=min_offset,
            ref_fac=ref_fac,
            min_dist_frac=min_dist_frac,
            fill_val=fill_val,
        )

        dat = {}
        samp_ind = -1
        for _, sample in zip(range(nr_iterations), datapipe):
            permea = sample["permeability"].cpu().detach().numpy()
            darcy = sample["darcy"].cpu().detach().numpy()
            pos = (sample["inset_pos"].cpu().detach().numpy()).astype(int)
            assert (np.where(pos == fill_val, 0, pos) % ref_fac).sum() == 0, (
                "inset off coarse grid"
            )

            # crop out refined region, allow for surrounding area, save in extra array
            for ii in range(batch_size):
                samp_ind += 1
                samp_str = str(samp_ind)

                # global fields
                dat[samp_str] = {
                    "ref0": {
                        "0": {
                            "permeability": permea[ii, 0, ::ref_fac, ::ref_fac],
                            "darcy": darcy[ii, 0, ::ref_fac, ::ref_fac],
                        }
                    }
                }

                # insets
                dat[samp_str]["ref1"] = {}
                for pp in range(pos.shape[1]):
                    if pos[ii, pp, 0] == fill_val:
                        continue
                    xs = pos[ii, pp, 0] - buffer
                    ys = pos[ii, pp, 1] - buffer

                    dat[samp_str]["ref1"][str(pp)] = {
                        "permeability": permea[
                            ii, 0, xs : xs + inset_size, ys : ys + inset_size
                        ],
                        "darcy": darcy[
                            ii, 0, xs : xs + inset_size, ys : ys + inset_size
                        ],
                        "pos": (pos[ii, pp, :] - min_offset) // ref_fac,
                    }
        meta = {"ref_fac": ref_fac, "buffer": buffer, "fine_res": fine_res}

        np.save(out_dir + file_names[dset], {"meta": meta, "fields": dat})

        # plot some fields
        for idx in range(n_plots):
            PlotNestedDarcy(dat, idx)


if __name__ == "__main__":
    nested_darcy_generator()

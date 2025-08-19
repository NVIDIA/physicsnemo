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

"""
This code runs the data processing in parallel to load OpenFoam files, process them 
and save in the npy format for faster processing in the DoMINO datapipes. Several 
parameters such as number of processors, input and output paths, etc. can be 
configured in config.yaml in the data_processing tab.
"""

from openfoam_datapipe import OpenFoamDataset
from physicsnemo.utils.domino.utils import *
import multiprocessing
import hydra, time, os
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def process_files(*args_list):
    ids = args_list[0]
    processor_id = args_list[1]
    fm_data = args_list[2]
    output_dir = args_list[3]
    for j in ids:
        fname = fm_data.filenames[j]
        if len(os.listdir(os.path.join(fm_data.data_path, fname))) == 0:
            print(f"Skipping {fname} - empty.")
            continue
        outname = os.path.join(output_dir, fname)
        print("Filename:%s on processor: %d" % (outname, processor_id))
        filename = f"{outname}.npy"
        if os.path.exists(filename):
            print(f"Skipping {filename} - already exists.")
            continue
        start_time = time.time()
        data_dict = fm_data[j]
        np.save(filename, data_dict)
        print("Time taken for %d = %f" % (j, time.time() - start_time))


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")
    phase = "train"
    volume_variable_names = list(cfg.variables.volume.solution.keys())
    num_vol_vars = 0
    for j in volume_variable_names:
        if cfg.variables.volume.solution[j] == "vector":
            num_vol_vars += 3
        else:
            num_vol_vars += 1

    surface_variable_names = list(cfg.variables.surface.solution.keys())
    num_surf_vars = 0
    for j in surface_variable_names:
        if cfg.variables.surface.solution[j] == "vector":
            num_surf_vars += 3
        else:
            num_surf_vars += 1

    fm_data = OpenFoamDataset(
        cfg.data_processor.input_dir,
        kind=cfg.data_processor.kind,
        volume_variables=volume_variable_names,
        surface_variables=surface_variable_names,
        model_type=cfg.model.model_type,
    )
    output_dir = cfg.data_processor.output_dir
    create_directory(output_dir)
    n_processors = cfg.data_processor.num_processors

    num_files = len(fm_data)
    ids = np.arange(num_files)
    num_elements = int(num_files / n_processors) + 1
    process_list = []
    ctx = multiprocessing.get_context("spawn")
    for i in range(n_processors):
        if i != n_processors - 1:
            sf = ids[i * num_elements : i * num_elements + num_elements]
        else:
            sf = ids[i * num_elements :]
        # print(sf)
        process = ctx.Process(target=process_files, args=(sf, i, fm_data, output_dir))

        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()


if __name__ == "__main__":
    main()

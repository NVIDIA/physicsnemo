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

import sys
import tarfile
import uuid
from multiprocessing import Pool
from pathlib import Path
from typing import List, Literal, Optional, Union
import webdataset as wds
import numpy as np
from src.data.drivaernet_datamodule import DrivAerNetDataset, DrivAerNetPreprocessor
from src.data.components.webdataset_utils import from_numpy
from torch.multiprocessing import set_start_method


class DrivAerNetToWebdataset:
    """
    Convert DrivAerNetDataset to a webdataset
    """

    def __init__(
        self,
        dataset: DrivAerNetDataset,
        output_path: Union[str, Path],
        tarfile_name: str = "data.tar",
    ):
        self.dataset = dataset
        self.tarfile_name = tarfile_name
        self.output_path = Path(output_path)
        self.temp_path = self.output_path / str(uuid.uuid4())[:8]
        self.temp_path.mkdir(exist_ok=True)
        print(
            f"Saving DrivAerNetWebdataset to {self.temp_path} for {self.dataset.data_path} phase: {self.dataset.phase}"
        )
        # Create a text file in the temp_path and print dataset information
        with open(self.temp_path / "info.txt", "w") as f:
            f.write(f"Dataset: {self.dataset.data_path}\n")
            f.write(f"Phase: {self.dataset.phase}\n")
            f.write(f"Number of items: {len(self.dataset)}\n")

    def _save_item(self, idx: int):
        print(f"Saving item {idx}/{len(self.dataset)}")
        item = self.dataset[idx]
        # Save to 0 padded index
        np.savez_compressed(
            self.temp_path / f"{idx:06d}.npz",
            **item,
        )

    def save(self, num_processes: int = 1):
        # Save each item in as a numpy file in parallel
        assert num_processes > 0, (
            f"num_processes should be greater than 0, got {num_processes}"
        )
        if num_processes < 2:
            for idx in range(len(self.dataset)):
                print(f"Saving item {idx}/{len(self.dataset)}")
                self._save_item(idx)
        elif num_processes > 1:
            with Pool(num_processes) as p:
                p.map(self._save_item, range(len(self.dataset)))

        # Compress the dataset to a tar file
        print("Compressing the dataset to a tar file")
        self.output_path = self.output_path.expanduser()
        self.output_path.mkdir(exist_ok=True)
        tar_path = self.output_path / self.tarfile_name
        with tarfile.open(tar_path, "w") as tar:
            for file in self.temp_path.glob("*.npz"):
                tar.add(file, arcname=file.name)


def convert_to_webdataset(
    data_path: str,
    out_path: Optional[str] = "~/datasets/drivaer_webdataset",
    phase: Optional[Literal["train", "val", "test"]] = "train",
    num_processes: Optional[int] = 2,
):
    set_start_method("spawn", force=True)
    # Add the parent directory to the path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

    # Please modify this to create appropriate data. Make sure to not use normalizer twice.
    preprocessors = [
        DrivAerNetPreprocessor(num_points=-1),
    ]

    # Create separate paths for train/text, with and without spoiler
    dataset = DrivAerNetDataset(data_path, phase=phase, preprocessors=preprocessors)
    output_path = Path(out_path).expanduser()
    output_path.mkdir(exist_ok=True)
    # Save to numpy
    DrivAerNetToWebdataset(
        dataset,
        output_path,
        tarfile_name=f"{phase}.tar",
    ).save(num_processes=num_processes)


def compute_pressure_stats(*data_paths: List[str]):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(sorted(data_paths)),
        wds.tarfile_to_samples(),
        wds.map(lambda x: from_numpy(x, "npz")),
    )
    num_points = []
    means = []
    vars = []

    # visualize the progress bar using tqdm
    import tqdm

    for item in tqdm.tqdm(dataset):
        p = item["time_avg_pressure"]
        num_points.append(p.shape[0])
        means.append(p.mean().item())
        vars.append(p.var().item())
    # Compute normalization function
    mean = np.mean(means)
    std = np.sqrt(np.mean(vars))
    print(f"Mean: {mean}, std: {std}")
    # Save the parameters as a text file
    with open("pressure_normalization.txt", "w") as f:
        f.write(f"Mean: {mean}, std: {std}")


if __name__ == "__main__":
    import fire

    fire.Fire(convert_to_webdataset)

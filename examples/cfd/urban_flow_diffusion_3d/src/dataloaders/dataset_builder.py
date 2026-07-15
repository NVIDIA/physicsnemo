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

import torch, sys

from torch.utils.data import DataLoader, DistributedSampler
from physicsnemo.distributed import DistributedManager
from src.dataloaders.dataset_utils import get_precision
from src.dataloaders.uflow_dataset import UflowDataset2D, UflowDataset3D


def get_dataset_and_dataloader(cfg, Train=True, seed=42):
    """
    Builds a dataset and dataloader for 2D or 3D UFlow datasets.

    Args:
        cfg: cfg from Hydra config
        Train: Whether to use DistributedSampler
        seed: Seed for reproducibility
        #TODO: Not sure if calling all the configs directly is good idea, consider shifting this to main train script
    Returns:
        dataloader, dataset (both as objects)
    """
    ds_type = cfg.dataset.type.lower()
    datatype = get_precision(cfg.train.perf.fp_optimizations)
    # print(datatype)
    if ds_type in ("uf-downsampled", "uf-full-res"):
        dim = len(cfg.dataset.img_res)
        dataset_cls = UflowDataset3D if dim == 3 else UflowDataset2D
    else:
        raise NotImplementedError(f"Unknown dataset type: {cfg.dataset.type}")

    mins = [cfg.dataset.u_comp.min, cfg.dataset.v_comp.min]
    maxs = [cfg.dataset.u_comp.max, cfg.dataset.v_comp.max]
    if hasattr(cfg.dataset, "w_comp"):
        mins.append(cfg.dataset.w_comp.min)
        maxs.append(cfg.dataset.w_comp.max)

    dataset_obj = dataset_cls(
        data_path=cfg.dataset.data_dir,
        ds_ratio=cfg.dataset.ds_ratio,
        normalize=cfg.dataset.normalize,
        mins=mins,
        maxs=maxs,
        datatype=datatype,
    )

    # --- Sampler for distributed training ---
    sampler = None
    shuffle = Train

    if Train:
        dist = DistributedManager()
        sampler = DistributedSampler(
            dataset=dataset_obj,
            rank=dist.rank,
            num_replicas=dist.world_size,
            seed=seed,
            shuffle=shuffle,
        )

    batch_size = (
        cfg.train.hp.batch_size_per_gpu
        if hasattr(cfg.train.hp, "batch_size_per_gpu")
        else 4
    )
    num_workers = (
        cfg.train.perf.dataloader_workers
        if hasattr(cfg.train.perf, "dataloader_workers")
        else 4
    )

    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=Train,
    )

    return dataloader, dataset_obj

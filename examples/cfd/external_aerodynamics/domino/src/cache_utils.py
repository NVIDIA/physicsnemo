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

"""Shared preprocessing for cache generation and full-data validation."""

from pathlib import Path

from omegaconf import DictConfig
from utils import get_keys_to_read, load_scaling_factors

from physicsnemo.datapipes.cae.cae_dataset import CAEDataset
from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe


def create_cache_dataset(cfg: DictConfig) -> DoMINODataPipe:
    """Load all processed points using the same normalization as training."""
    volume_factors, surface_factors = load_scaling_factors(cfg)
    keys, optional_keys = get_keys_to_read(cfg, cfg.model.model_type)
    datapipe = DoMINODataPipe(
        input_path=cfg.data_processor.output_dir,
        model_type=cfg.model.model_type,
        phase="train",
        grid_resolution=cfg.model.interp_res,
        normalize_coordinates=cfg.data.normalize_coordinates,
        sampling=False,
        sample_in_bbox=cfg.data.sample_in_bbox,
        volume_factors=volume_factors,
        surface_factors=surface_factors,
        scaling_type=cfg.model.normalization,
        bounding_box_dims=cfg.data.bounding_box,
        bounding_box_dims_surf=cfg.data.bounding_box_surface,
        num_surface_neighbors=cfg.model.num_neighbors_surface,
        surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        caching=True,
        gpu_preprocessing=cfg.data.gpu_preprocessing,
        gpu_output=False,
    )
    datapipe.set_dataset(
        CAEDataset(
            data_dir=cfg.data_processor.output_dir,
            keys_to_read=keys,
            keys_to_read_if_available=optional_keys,
            output_device=datapipe.preproc_device,
        )
    )
    return datapipe


def cache_filename(dataset: DoMINODataPipe, index: int) -> str:
    """Identify the source file without adding string metadata to tensor outputs."""
    return Path(dataset.dataset._filenames[index]).name + "_cached.npy"

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

"""Data loading, validation, and preprocessing utilities."""

from data.dataloader import (
    ReservoirDataset,
    collate_fn,
    create_dataloaders,
    get_dataset_info,
)
from data.file_resolution import resolve_data_files
from data.gpu_prefetch import GPUPrefetcher
from data.mask_detection import MaskResult, detect_mask
from data.normalization import NormStats, compute_norm_stats, normalize_sample
from data.scalar_utils import (
    create_mionet_collate_fn,
    detect_scalar_channels,
    log_scalar_detection_results,
    verify_scalar_consistency,
)
from data.validation import (
    detect_dimensions,
    get_dimension_info,
    print_validation_summary,
    validate_batch_dimensions,
    validate_sample_dimensions,
)

__all__ = [
    # Dataloader
    "ReservoirDataset",
    "collate_fn",
    "create_dataloaders",
    "get_dataset_info",
    # File resolution
    "resolve_data_files",
    # Mask detection
    "detect_mask",
    "MaskResult",
    # Normalization
    "compute_norm_stats",
    "normalize_sample",
    "NormStats",
    # GPU prefetch
    "GPUPrefetcher",
    # Validation
    "detect_dimensions",
    "validate_batch_dimensions",
    "validate_sample_dimensions",
    "print_validation_summary",
    "get_dimension_info",
    # Scalar utils
    "detect_scalar_channels",
    "verify_scalar_consistency",
    "create_mionet_collate_fn",
    "log_scalar_detection_results",
]

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

r"""
Normalization utilities for flood prediction datasets.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer


def collect_all_fields(
    dataset, 
    expect_target: bool = True
) -> Union[
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]]],
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]], List[torch.Tensor]]
]:
    r"""
    Collect all fields from a dataset into lists.

    Parameters
    ----------
    dataset : Dataset
        Dataset to collect fields from.
    expect_target : bool, optional, default=True
        Whether to expect target field.

    Returns
    -------
    Tuple[List[torch.Tensor], ...]
        Tuple of lists: (geometry, static, boundary, dynamic, target, [cell_area]).
        If cell_area is found, returns 6-tuple, otherwise 5-tuple.

    Raises
    ------
    KeyError
        If required fields are missing.
    """
    geometry_list = []
    static_list = []
    boundary_list = []
    dynamic_list = []
    target_list = []
    cell_area_list = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        # Validate required fields
        required_fields = ["geometry", "static", "boundary", "dynamic"]
        missing_fields = [field for field in required_fields if field not in sample]
        if missing_fields:
            raise KeyError(f"Sample {i} missing required fields: {missing_fields}")
        
        geometry_list.append(sample["geometry"])
        static_list.append(sample["static"])
        boundary_list.append(sample["boundary"])
        dynamic_list.append(sample["dynamic"])
        if expect_target:
            target_list.append(sample.get("target", None))
        if "cell_area" in sample:
            cell_area_list.append(sample["cell_area"])

    # Return cell_area if it was found
    if cell_area_list:
        return geometry_list, static_list, boundary_list, dynamic_list, target_list, cell_area_list
    else:
        return geometry_list, static_list, boundary_list, dynamic_list, target_list


def stack_and_fit_transform(
    geom_list: List[torch.Tensor],
    static_list: List[torch.Tensor],
    boundary_list: List[torch.Tensor],
    dyn_list: List[torch.Tensor],
    tgt_list: List[Optional[torch.Tensor]],
    normalizers: Optional[Dict[str, UnitGaussianNormalizer]] = None,
    fit_normalizers: bool = True
) -> Tuple[Dict[str, UnitGaussianNormalizer], Dict[str, torch.Tensor]]:
    r"""
    Stack field lists into tensors and apply normalization.

    Parameters
    ----------
    geom_list : List[torch.Tensor]
        List of geometry tensors.
    static_list : List[torch.Tensor]
        List of static feature tensors.
    boundary_list : List[torch.Tensor]
        List of boundary condition tensors.
    dyn_list : List[torch.Tensor]
        List of dynamic feature tensors.
    tgt_list : List[Optional[torch.Tensor]]
        List of target tensors.
    normalizers : Dict[str, UnitGaussianNormalizer], optional
        Dict of existing normalizers (if fit_normalizers=False).
    fit_normalizers : bool, optional, default=True
        Whether to fit new normalizers.

    Returns
    -------
    Tuple[Dict[str, UnitGaussianNormalizer], Dict[str, torch.Tensor]]
        Tuple of (normalizers dict, big_tensors dict).

    Raises
    ------
    ValueError
        If lists are empty or have incompatible shapes.
    """
    # Filter out None values before stacking to avoid errors
    geom_list = [g for g in geom_list if g is not None] if geom_list else []
    static_list = [s for s in static_list if s is not None] if static_list else []
    boundary_list = [b for b in boundary_list if b is not None] if boundary_list else []
    dyn_list = [d for d in dyn_list if d is not None] if dyn_list else []
    tgt_list = [t for t in tgt_list if t is not None] if tgt_list else []
    
    geometry_big = torch.stack(geom_list, dim=0) if geom_list else None
    static_big = torch.stack(static_list, dim=0) if static_list else None
    boundary_big = torch.stack(boundary_list, dim=0) if boundary_list else None
    dynamic_big = torch.stack(dyn_list, dim=0) if dyn_list else None
    target_big = torch.stack(tgt_list, dim=0) if tgt_list else None

    if normalizers is None:
        normalizers = {}

    if geometry_big is not None:
        if fit_normalizers:
            geometry_norm = UnitGaussianNormalizer(dim=[0, 1])
            geometry_norm.fit(geometry_big)
            geometry_big = geometry_norm.transform(geometry_big)
            normalizers["geometry"] = geometry_norm
        else:
            geometry_big = normalizers["geometry"].transform(geometry_big)

    if static_big is not None:
        if fit_normalizers:
            static_norm = UnitGaussianNormalizer(dim=[0, 1])
            static_norm.fit(static_big)
            static_big = static_norm.transform(static_big)
            normalizers["static"] = static_norm
        else:
            static_big = normalizers["static"].transform(static_big)

    if boundary_big is not None:
        if fit_normalizers:
            boundary_norm = UnitGaussianNormalizer(dim=[0, 1, 2])
            boundary_norm.fit(boundary_big)
            boundary_big = boundary_norm.transform(boundary_big)
            normalizers["boundary"] = boundary_norm
        else:
            boundary_big = normalizers["boundary"].transform(boundary_big)

    if target_big is not None:
        if fit_normalizers:
            target_norm = UnitGaussianNormalizer(dim=[0, 1, 2])
            target_norm.fit(target_big)
            target_big = target_norm.transform(target_big)
            normalizers["target"] = target_norm
        else:
            target_big = normalizers["target"].transform(target_big)

    if dynamic_big is not None:
        if "target" in normalizers and normalizers["target"] is not None:
            # Use target normalizer for dynamic fields if available
            dynamic_big = normalizers["target"].transform(dynamic_big)
            normalizers["dynamic"] = normalizers["target"]
        elif fit_normalizers:
            # Create separate normalizer for dynamic if target is unavailable
            dynamic_norm = UnitGaussianNormalizer(dim=[0, 1, 2])
            dynamic_norm.fit(dynamic_big)
            dynamic_big = dynamic_norm.transform(dynamic_big)
            normalizers["dynamic"] = dynamic_norm
        elif "dynamic" in normalizers:
            # Use existing dynamic normalizer if available
            dynamic_big = normalizers["dynamic"].transform(dynamic_big)

    big_tensors = {
        "geometry": geometry_big,
        "static": static_big,
        "boundary": boundary_big,
        "dynamic": dynamic_big,
        "target": target_big,
    }
    return normalizers, big_tensors


def transform_with_existing_normalizers(
    geom_list: List[torch.Tensor],
    static_list: List[torch.Tensor],
    boundary_list: List[torch.Tensor],
    dyn_list: List[torch.Tensor],
    normalizers: Dict[str, UnitGaussianNormalizer]
) -> Dict[str, torch.Tensor]:
    r"""
    Transform data lists using existing normalizers.

    Parameters
    ----------
    geom_list : List[torch.Tensor]
        List of geometry tensors.
    static_list : List[torch.Tensor]
        List of static feature tensors.
    boundary_list : List[torch.Tensor]
        List of boundary condition tensors.
    dyn_list : List[torch.Tensor]
        List of dynamic feature tensors.
    normalizers : Dict[str, UnitGaussianNormalizer]
        Dict of normalizers to use.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dict of transformed tensors.

    Raises
    ------
    KeyError
        If required normalizers are missing.
    ValueError
        If lists are empty.
    """
    if not normalizers:
        raise ValueError("normalizers dict cannot be empty")
    transformed = {}
    data_map = {"geometry": geom_list, "static": static_list, "boundary": boundary_list, "dynamic": dyn_list}
    
    for key, data_list in data_map.items():
        if data_list and key in normalizers:
            big_tensor = torch.stack(data_list, dim=0)
            transformed[key] = normalizers[key].transform(big_tensor)
    
    return transformed


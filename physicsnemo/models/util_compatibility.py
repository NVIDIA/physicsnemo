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
from typing import Any, Dict


def convert_ckp_apex(
    ckp_args_dict: Dict[str, Any],
    model_args: Dict[str, Any],
    model_dict: Dict[str, Any],
) -> Dict[str, Any]:

    """Utility for converting Apex GroupNorm-related keys in a checkpoint.

    This function modifies the checkpoint arguments and model dictionary
    to ensure compatibility when switching between Apex-optimized models
    and standard PyTorch models.

    Parameters
    ----------
    ckp_args_dict : Dict[str, Any]
        Dictionary of checkpoint arguments (e.g., configuration parameters saved during training).
    model_args : Dict[str, Any]
        Dictionary of model initialization arguments that may need updating.
    model_dict : Dict[str, Any]
        Dictionary containing model state_dict (weights) loaded from checkpoint.

    Returns
    -------
    Dict[str, Any]
        Updated model_dict with necessary key modifications applied for compatibility.

    Raises
    ------
    KeyError
        If essential expected keys are missing during the conversion process.
    """

    apex_in_ckp = ("use_apex_gn" in ckp_args_dict["__args__"].keys()) and (
        ckp_args_dict["__args__"]["use_apex_gn"]
    )
    apex_in_workflow = (
        (model_args is not None)
        and ("use_apex_gn" in model_args.keys())
        and (model_args["use_apex_gn"])
    )

    filtered_state_dict = {}
    # case1: try to use non-optimized ckp in optimized workflow
    if (not apex_in_ckp) and apex_in_workflow:
        # transfer GN weight & bias to apex GN weight & bias
        for key, value in model_dict.items():
            is_duplicate = False
            for norm_layer in ["norm0", "norm1", "norm2", "aux_norm"]:
                if f"{norm_layer}.weight" in key:
                    new_key = key.replace(
                        f"{norm_layer}.weight", f"{norm_layer}.gn.weight"
                    )
                    filtered_state_dict[new_key] = value
                    is_duplicate = True
                elif f"{norm_layer}.bias" in key:
                    new_key = key.replace(f"{norm_layer}.bias", f"{norm_layer}.gn.bias")
                    filtered_state_dict[new_key] = value
                    is_duplicate = True
            if not is_duplicate:
                filtered_state_dict[key] = value

    # case2: try to use optimized ckp in non-optimized workflow
    elif apex_in_ckp and (not apex_in_workflow):
        # transfer apex GN weight & bias to GN weight & bias
        for key, value in model_dict.items():
            is_duplicate = False
            for norm_layer in ["norm0", "norm1", "norm2", "aux_norm"]:
                if f"{norm_layer}.gn.weight" in key:
                    new_key = key.replace(
                        f"{norm_layer}.gn.weight", f"{norm_layer}.weight"
                    )
                    filtered_state_dict[new_key] = value
                    is_duplicate = True
                elif f"{norm_layer}.bias" in key:
                    new_key = key.replace(f"{norm_layer}.gn.bias", f"{norm_layer}.bias")
                    filtered_state_dict[new_key] = value
                    is_duplicate = True
            if not is_duplicate:
                filtered_state_dict[key] = value
    else:
        # no need to convert ckp
        return model_dict

    return filtered_state_dict

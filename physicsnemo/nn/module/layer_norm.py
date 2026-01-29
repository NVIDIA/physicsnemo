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

import importlib
import os
import warnings

import torch
from torch import nn

from physicsnemo.core.version_check import check_version_spec

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)


def remove_extra_state_hook_for_torch(
    module: nn.Module,
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
) -> None:
    """
    Pre-hook to remove Transformer Engine's extra state from the state_dict when loading into a PyTorch LayerNorm.

    This function scans the state_dict for any keys that match the pattern '{prefix}norm._extra_state'
    and removes them. These keys are specific to Transformer Engine's LayerNorm and are not needed
    (and may cause errors) when loading into a standard PyTorch LayerNorm.

    Args:
        module (nn.Module): The module into which the state_dict is being loaded.
        state_dict (dict): The state dictionary being loaded.
        prefix (str): The prefix for parameters in this module.
        local_metadata (dict): Metadata for this module.
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict function.
        missing_keys (list): List of missing keys.
        unexpected_keys (list): List of unexpected keys.
        error_msgs (list): List of error messages.
    """
    # Go through the state dict, and for any keys that have
    # prefix + "norm._extra_state", remove those.
    # They are extra from transformer engine and not needed in the
    # torch layernorm.
    keys_to_remove = [
        key for key in state_dict if key.startswith(prefix + "_extra_state")
    ]
    for key in keys_to_remove:
        del state_dict[key]


def ignore_missing_extra_state_key(
    module: nn.Module, incompatible_keys: torch.nn.modules.module._IncompatibleKeys
) -> None:
    """
    Post-hook to ignore missing 'ln.norm._extra_state' key when loading state_dict.

    This function removes 'ln.norm._extra_state' from the list of missing keys in
    the IncompatibleKeys object. This is useful when loading a checkpoint saved
    from a Transformer Engine LayerNorm into a PyTorch LayerNorm, where this extra
    state is not present or needed.

    Args:
        module (nn.Module): The module into which the state_dict is being loaded.
        incompatible_keys: An object with a 'missing_keys' attribute (typically torch.nn.modules.module._IncompatibleKeys).
    """
    # Remove 'ln.norm._extra_state' from the missing keys:
    problem_key = "ln._extra_state"
    if problem_key in incompatible_keys.missing_keys:
        incompatible_keys.missing_keys.remove(problem_key)


def get_layer_norm_class() -> nn.Module:
    """
    Dynamically pick the layer norm provider based on availability of transformer engine.
    If transformer engine is available, it will use the transformer engine implementation of
    LayerNorm. Otherwise, it will use the pytorch implementation of LayerNorm.

    Override the default behavior by setting the PHYSICSNEMO_FORCE_TE environment variable.
    """

    # This is to allow users to force the use of TE or pytorch layer norm
    force_te_setting = os.environ.get("PHYSICSNEMO_FORCE_TE")
    te_available = (
        TE_AVAILABLE  # make a local copy to avoid changing the global variable
    )

    # Can't use transformer engine without cuda:
    if not torch.cuda.is_available():
        te_available = False

    # Let the users force the setting no matter what:
    if force_te_setting is not None:
        if force_te_setting.lower() == "true" or force_te_setting.lower() == "1":
            te_available = True
        elif force_te_setting.lower() == "false" or force_te_setting.lower() == "0":
            te_available = False
        else:
            # In this scenario, the variable PHYSICSNEMO_FORCE_TE was set, but not
            # to a value we expect.  Emit a warning:
            warnings.warn(
                f"The PHYSICSNEMO_FORCE_TE environment variable was set to an invalid value. "
                f"Expected 'True' or 'False', but got '{force_te_setting}'. "
                "Ignoring the variable and using the default behavior.",
                UserWarning,
                stacklevel=2,
            )

    if te_available:
        te = importlib.import_module("transformer_engine.pytorch")
        base = te.LayerNorm
    else:
        base = nn.LayerNorm

    class LayerNorm(base):
        """
        Wrapper around layer norm utilities.

        This class will default to using the transformer engine implementation of
        LayerNorm - it is significantly faster in the backwards pass.

        If transformer engine is not available, it will fall back to the
        pytorch implementation of LayerNorm.

        Additionally, this class registers pre or post hooks to allow you to
        train with / without transformer engine, and run inference
        with / without transformer engine.

        .. note::
            Transformer engine adds additional state parameters that affect
            fp8 stability. **Do NOT** switch from transformer engine to pytorch
            or from pytorch to transformer engine with a checkpoint if you
            are using fp8 precision in the layer norm regions.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if te_available:
                self.register_load_state_dict_post_hook(ignore_missing_extra_state_key)
            else:
                self.register_load_state_dict_pre_hook(
                    remove_extra_state_hook_for_torch
                )

    return LayerNorm


LayerNorm = get_layer_norm_class()

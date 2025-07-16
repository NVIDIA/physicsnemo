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

import os
import torch
from torch import nn

try:
    import transformer_engine.pytorch as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

# This is to allow users to force the use of TE or pytorch layer norm
force_te_setting = os.environ.get("PHYSICSNEMO_FORCE_TE")
if force_te_setting is not None:
    if force_te_setting.lower() == "true" or force_te_setting.lower() == "1":
        TE_AVAILABLE = True
    elif force_te_setting.lower() == "false" or force_te_setting.lower() == "0":
        TE_AVAILABLE = False

def remove_extra_state_hook_for_torch(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
    
    
    # Go through the state dict, and for any keys that have
    # prefix + "norm._extra_state", remove those.
    # They are extra from transformer engine and not needed in the 
    # torch layernorm.
    keys_to_remove = [key for key in state_dict if key.startswith(prefix + "norm._extra_state")]
    for key in keys_to_remove:
        del state_dict[key]
    

    

def ignore_missing_extra_state_key(module, incompatible_keys) -> None:
    
    # Remove 'ln.norm._extra_state' from the missing keys:

    problem_key = "ln.norm._extra_state"
    if problem_key in incompatible_keys.missing_keys:
        incompatible_keys.missing_keys.remove(problem_key)
    
class LayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_te = TE_AVAILABLE
        
        
        # TE uses an extra state to manage fp8 scaling
        # It shows up in the state dict, making the two
        # layers incompatiple with each other
        # https://github.com/NVIDIA/TransformerEngine/issues/458

        # As a workaround, we we're loading a te-trained layer norm
        # into torch layer norm, remove that state:        
        
        if self.use_te:
            self.norm = te.LayerNorm(*args, **kwargs)
            self.register_load_state_dict_post_hook(ignore_missing_extra_state_key)
        else:
            self.norm = nn.LayerNorm(*args, **kwargs)
            self.register_load_state_dict_pre_hook(remove_extra_state_hook_for_torch)
        
    def forward(self, x):
        return self.norm(x)

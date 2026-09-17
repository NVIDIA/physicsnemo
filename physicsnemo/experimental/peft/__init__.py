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

"""Native LoRA for PhysicsNeMo models (``physicsnemo.experimental.peft``)."""

from physicsnemo.experimental.peft.apply import (
    ApplyResult,
    apply_lora,
    resolve_targets,
)
from physicsnemo.experimental.peft.config import LoRAConfig
from physicsnemo.experimental.peft.io import load_adapter, save_adapter
from physicsnemo.experimental.peft.lora import (
    LoRALayer,
    LoRALinear,
    is_lora_layer,
    register_lora_wrapper,
    wrappable_types,
)
from physicsnemo.experimental.peft.merge import merge_lora
from physicsnemo.experimental.peft.utils import (
    print_trainable_parameters,
    set_adapter_enabled,
    split_params_for_optimizer,
)

__all__ = [
    "LoRAConfig",
    "apply_lora",
    "resolve_targets",
    "ApplyResult",
    "merge_lora",
    "save_adapter",
    "load_adapter",
    "LoRALayer",
    "LoRALinear",
    "is_lora_layer",
    "register_lora_wrapper",
    "wrappable_types",
    "print_trainable_parameters",
    "set_adapter_enabled",
    "split_params_for_optimizer",
]

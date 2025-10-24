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

from typing import Union

from physicsnemo import Module


default_model_params = {
    "AFNO": {
        "patch_size": (8, 8),
        "embed_dim": 768,
        "depth": 12,
        "num_blocks": 8,
    }
}


def setup_model(
    model_type: str = "AFNO", model_name: Union[str, None] = None, **model_cfg
) -> Module:
    """Setup model from config dict."""
    model_kwargs = default_model_params[model_type].copy()
    model_kwargs.update(model_cfg)

    model = Module.instantiate({"__name__": model_type, "__args__": model_kwargs})

    if model_name is not None:
        model.meta.name = model_name

    return model

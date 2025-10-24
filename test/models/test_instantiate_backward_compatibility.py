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


import math

import pytest

import physicsnemo


@pytest.mark.parametrize(
    "model",
    [
        {
            "__name__": "AFNO",
            "__module__": "modulus.models.afno.afno",  # Modulus namespace
            "__args__": {
                "inp_shape": [128, 128],
                "in_channels": 3,
                "out_channels": 2,
                "patch_size": [16, 16],
                "embed_dim": 256,
                "depth": 4,
                "mlp_ratio": 4.0,
                "drop_rate": 0.0,
                "num_blocks": 16,
                "sparsity_threshold": 0.01,
                "hard_thresholding_fraction": 1.0,
            },
        },
        {
            "__name__": "StormCastUNet",
            "__module__": "modulus.models.diffusion.unet",
            "__args__": {
                "img_resolution": [512, 640],
                "img_in_channels": 127,
                "img_out_channels": 99,
                "use_fp16": False,
                "sigma_min": 0,
                "sigma_max": math.inf,
                "sigma_data": 0.5,
                "model_type": "SongUNet",
                "embedding_type": "zero",
                "channel_mult": [1, 2, 2, 2, 2],
                "attn_resolutions": [],
                "additive_pos_embed": False,
            },
        },
    ],
)
def test_instantiate_backward_compatibility(model):
    """Test instantiation of a model from a dictionary coming from modulus namespace."""
    model = physicsnemo.models.Module.instantiate(model)

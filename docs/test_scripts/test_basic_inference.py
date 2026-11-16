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

# [imports]
import torch

import physicsnemo
from physicsnemo.models.fno.fno import FNO

# [imports]

# [code]

model = FNO(
    in_channels=1,
    out_channels=1,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=2,
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
    padding=5,
).to("cuda")

# Save the checkpoint. For demo, we will just save untrained checkpoint,
# but in typical workflows is saved after model training.
model.save("untrained_checkpoint.mdlus")

# Inference code

# The parameters to instantitate the model will be loaded from the checkpoint
model_inf = physicsnemo.Module.from_checkpoint("untrained_checkpoint.mdlus").to("cuda")

# put the model in evaluation mode
model_inf.eval()

# run inference
with torch.inference_mode():
    input = torch.ones(8, 1, 256, 256).to("cuda")
    output = model_inf(input)
    print(output.shape)

# [code]

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
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.metrics.general.mse import mse
from physicsnemo.models.fno.fno import FNO

# [imports]

# [code]
normaliser = { # Dictionary with mean and std of the permeability and darcy fields
    "permeability": (1.25, 0.75), 
    "darcy": (3.63e-2, 2.44e-2),
}
dataloader = Darcy2D(
    resolution=256, batch_size=64, nr_permeability_freq=5, normaliser=normaliser
)
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: 0.85**step
)

# run for 20 iterations
dataloader = iter(dataloader)
for i in range(20):
    batch = next(dataloader)
    truth = batch["darcy"]
    pred = model(batch["permeability"])
    loss = mse(pred, truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    print(f"Iteration: {i}. Loss: {loss.detach().cpu().numpy()}")
# [code]

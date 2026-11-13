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
from torch.nn.parallel import DistributedDataParallel

import physicsnemo
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.general.mse import mse
from physicsnemo.models.fno.fno import FNO
from physicsnemo.utils import StaticCaptureTraining

# [imports]

# [code]


def main():
    # Initialize the DistributedManager. This will automatically
    # detect the number of processes the job was launched with and
    # set those configuration parameters appropriately.
    DistributedManager.initialize()

    # Get instance of the DistributedManager
    dist = DistributedManager()

    normaliser = {
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
    ).to(dist.device)

    # Set up DistributedDataParallel if using more than a single process.
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[
                    dist.local_rank
                ],  # Set the device_id to be the local rank of this process on this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.85**step
    )

    # Create training step function with optimization wrapper
    # StaticCaptureTraining calls `backward` on the loss and
    # `optimizer.step()` so you don't have to do that
    # explicitly.
    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        cuda_graph_warmup=11,
    )
    def training_step(invar, outvar):
        predvar = model(invar)
        loss = mse(predvar, outvar)
        return loss

    # run for 20 iterations
    dataloader = iter(dataloader)
    for i in range(20):
        batch = next(dataloader)
        truth = batch["darcy"]
        input = batch["permeability"]
        loss = training_step(input, truth)
        scheduler.step()


if __name__ == "__main__":
    main()

# [code]

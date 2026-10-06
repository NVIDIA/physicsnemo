# [imports]
import torch

# [imports]
# [pytorch model]
import torch.nn as nn

import physicsnemo
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.metrics.general.mse import mse


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)

        self.dec1 = self.upconv_block(128, 64)
        self.dec2 = self.upconv_block(64, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return self.final(x)


# [pytorch model]

# [physicsnemo model]

from dataclasses import dataclass

import torch.nn as nn

from physicsnemo.core import ModelMetaData, Module


@dataclass
class MdlsUNetMetaData(ModelMetaData):
    name: str = "MdlsUNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


MdlsUNet = Module.from_torch(UNet, meta=MdlsUNetMetaData)

# [physicsnemo model]


# [code]

import time

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining

# CUDA graph capture checks the distributed manager, even for a single process
DistributedManager.initialize()

normaliser = {
    "permeability": (1.25, 0.75),
    "darcy": (4.52e-2, 2.79e-2),
}
dataloader = Darcy2D(
    resolution=256, batch_size=8, nr_permeability_freq=5, normaliser=normaliser
)
model = MdlsUNet().to("cuda")

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
# [code]

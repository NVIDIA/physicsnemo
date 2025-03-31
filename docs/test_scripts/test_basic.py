# [imports]
import torch

import physicsnemo
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.metrics.general.mse import mse
from physicsnemo.models.fno.fno import FNO

# [imports]

# [code]
normaliser = {
    "permeability": (1.25, 0.75),
    "darcy": (4.52e-2, 2.79e-2),
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
for i in range(20):
    batch = next(iter(dataloader))
    true = batch["darcy"]
    pred = model(batch["permeability"])
    loss = mse(pred, true)
    loss.backward()
    optimizer.step()
    scheduler.step()

    print(f"Iteration: {i}. Loss: {loss.detach().cpu().numpy()}")
# [code]

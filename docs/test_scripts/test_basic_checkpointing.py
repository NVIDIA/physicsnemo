# [imports]
import torch

import physicsnemo
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
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

# load the epoch and optimizer, model and scheduler parameters from the checkpoint if
# it exists. Here we will use the `load_checkpoint` function to load the checkpoint,
# optimizer, and scheduler parameters from the checkpoint.
loaded_epoch = load_checkpoint(
    "./checkpoints",
    models=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device="cuda",
)

# we will setup the training to run for 20 epochs each epoch running for 5 iterations
# starting with the loaded epoch
dataloader = iter(dataloader)
for i in range(max(1, loaded_epoch), 20):
    # this would be iterations through different batches
    for _ in range(5):
        batch = next(dataloader)
        true = batch["darcy"]
        pred = model(batch["permeability"])
        loss = mse(pred, true)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # save checkpoint every 5th epoch
    if i % 5 == 0:
        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=i,
        )
# [code]

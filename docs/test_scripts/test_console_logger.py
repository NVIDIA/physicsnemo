# [imports]
import torch

import physicsnemo
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.launch.logging import LaunchLogger, PythonLogger
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

# Initialize the logger
logger = PythonLogger("main")  # General python logger
LaunchLogger.initialize()

# Use logger methods to track various information during training
logger.info("Starting Training!")

# we will setup the training to run for 20 epochs each epoch running for 5 iterations
dataloader = iter(dataloader)
for i in range(20):
    # wrap the epoch in launch logger to control frequency of output for console logs
    with LaunchLogger("train", epoch=i) as launchlog:
        # this would be iterations through different batches
        for _ in range(5):
            batch = next(dataloader)
            truth = batch["darcy"]
            pred = model(batch["permeability"])
            loss = mse(pred, truth)
            loss.backward()
            optimizer.step()
            scheduler.step()
            launchlog.log_minibatch({"Loss": loss.detach().cpu().numpy()})

        launchlog.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})


logger.info("Finished Training!")
# [code]

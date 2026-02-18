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

"""
Minimal example to show how the active learning workflow can be put together,
comprising the minimum of model training, and running some query strategy
to select data samples for labeling.

This example implements a simple MLP that takes in 2D coordinates and
outputs logits for a binary classification task. The active learning
workflow here uses a query strategy that looks for samples that have
the highest classification uncertainty (i.e. closest to 0.5), and iterates
by adding those samples to the training set. Ideally, if uncertainty is
well-adjusted to this problem, then the query strategy will select samples
that are more likely to improve the model's general performance, as compared
to a random selection baseline.
"""

import queue
import time

import torch
from moon_data import MoonsDataset
from moon_strategies import ClassifierUQQuery, DummyLabelStrategy, F1Metrology
from torch import nn

from physicsnemo import ModelMetaData, Module
from physicsnemo.active_learning import Driver, registry
from physicsnemo.active_learning import config as c
from physicsnemo.active_learning.loop import DefaultTrainingLoop

torch.manual_seed(216167)


@registry.register("MLP")
class MLP(Module):
    """
    Define a trivial MLIP model that will classify a 2D coordinate
    into one of two classes, producing logits as the output.

    There is nothing to configure here, so focus on the active learning
    components.
    """

    def __init__(self):
        super().__init__(meta=ModelMetaData(amp=False))
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor, shape [B, 2] for a batch
            size of B.

        Returns
        -------
        torch.Tensor
            The output tensor, shape [B, 1] for a batch
            size of B. Remember to ``squeeze`` the output.
        """
        return self.layers(x)


# this implements the `TrainingProtocol` interface
@registry.register("training_step")
def training_step(model: MLP, data: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Implements the training logic for a single batch of data.

    Parameters
    ----------
    model: MLP
        The model to train.
    data: tuple[torch.Tensor, torch.Tensor]
        The data to train on.

    Returns
    -------
    torch.Tensor
        The loss tensor.
    """
    x, y = data
    logits = model(x).squeeze()
    loss = model.loss_fn(logits, y)
    return loss


def main():
    """
    Configure an end-to-end active learning workflow.

    The code below primarily demonstrates how to compose things together
    to form the full workflow. There are three configurations structures
    that ultimately dictate the behavior of ``Driver``, which orchestrates
    the workflow:

    1. ``TrainingConfig``: everything to do with the model training process.
    2. ``StrategiesConfig``: comprises the query, label, and metrology strategies.
    3. ``DriverConfig``: decides things like batch size, logging, and ``DistributedManager``.

    The workflow should completely quickly: an `active_learning_logs` folder will
    be created, and within it, run-specific logs. You will find the model weights,
    alongside JSON logs of the process and from the ``F1Metrology`` strategy, which
    will records how precision/recall progresses as more data points are added to the
    strategy.
    """
    # instantiate the model and data
    dataset = MoonsDataset()
    uq_model = MLP()

    # configure how training/fine-tuning is done within active learning
    training_config = c.TrainingConfig(
        train_datapool=dataset,
        optimizer_config=c.OptimizerConfig(
            torch.optim.SGD,
            optimizer_kwargs={"lr": 0.01},
        ),
        # configure different times for initial training and subsequent
        # fine-tuning
        max_training_epochs=30,
        max_fine_tuning_epochs=30,
        # this configures the training loop
        train_loop_fn=DefaultTrainingLoop(
            use_progress_bars=False,
            enable_static_capture=False,
        ),
    )
    # this configuration packs all the strategy components together
    strategy_config = c.StrategiesConfig(
        query_strategies=[ClassifierUQQuery(max_samples=10)],
        queue_cls=queue.Queue,
        label_strategy=DummyLabelStrategy(),
        metrology_strategies=[F1Metrology()],
    )
    # this driver class handles the active learning loop
    driver_config = c.DriverConfig(
        batch_size=16,
        max_active_learning_steps=70,
        fine_tuning_lr=0.005,
        device=torch.device("cpu"),  # set to other accelerators if needed
    )
    driver = Driver(
        config=driver_config,
        learner=uq_model,
        strategies_config=strategy_config,
        training_config=training_config,
    )
    # our model doesn't implement a `training_step` method but in principle
    # it could be implemented, and we wouldn't need to pass the step function here
    driver(train_step_fn=training_step)

    # just some sanity checks
    if not (
        len(dataset.train_indices)
        == int(dataset.initial_samples * dataset.total_samples)
        + driver_config.max_active_learning_steps
        * strategy_config.query_strategies[0].max_samples
    ):
        raise RuntimeError(
            "Number of samples added to the training pool inconsistent with expected value."
        )

    # restart the driver from a checkpoint; in practice the path would be provided
    # train_datapool must be provided since it's not serialized
    # learner must nominally have the same architecture as the one used to create the checkpoint
    new_driver = Driver.load_checkpoint(
        driver.log_dir / "checkpoints" / "step_42" / "labeling",
        learner=uq_model,
        train_datapool=dataset,
    )
    assert new_driver.active_learning_step_idx == 42
    # enable this to re-run the driver training: be aware that this will overwrite subsequent checkpoints!!
    RERUN = True
    if RERUN:
        new_driver.logger.info(
            f"Rerunning driver from checkpoint {new_driver.last_checkpoint}"
        )
        time.sleep(5)
        new_driver(train_step_fn=training_step)


if __name__ == "__main__":
    main()

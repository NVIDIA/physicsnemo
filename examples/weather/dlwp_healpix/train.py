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

import logging
import os
import sys

import hydra
import numpy as np
import torch as th
from physicsnemo.distributed import DistributedManager
from hydra.utils import instantiate

from physicsnemo import Module
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg):
    """Train DLWP HEALPix weather model using the techniques described in the
    paper "Advancing Parsimonious Deep Learning Weather Prediction using the HEALPix Mesh".
    """
    # Initialize distributed
    DistributedManager.initialize()
    dist = DistributedManager()

    # set device globally to be sure that no spurious context are created on gpu 0:
    th.cuda.set_device(dist.device)

    # Initialize logger.
    os.makedirs(".logs", exist_ok=True)
    logger = PythonLogger(name="train")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging(file_name=f".logs/train_{dist.rank}.log")
    logger0.info(f"experiment working directory: {os.getcwd()}")

    # Seed
    if cfg.seed is not None:
        th.manual_seed(cfg.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Data module
    data_module = instantiate(cfg.data.module)

    # Model
    input_channels = len(cfg.data.input_variables)
    output_channels = (
        len(cfg.data.output_variables)
        if cfg.data.output_variables is not None
        else input_channels
    )
    constants_arr = data_module.constants
    n_constants = (
        0 if constants_arr is None else len(constants_arr.keys())
    )  # previously was 0 but with new format it is 1

    decoder_input_channels = int(cfg.data.get("add_insolation", 0))
    cfg.model["input_channels"] = input_channels
    cfg.model["output_channels"] = output_channels
    cfg.model["n_constants"] = n_constants
    cfg.model["decoder_input_channels"] = decoder_input_channels

    # convert Hydra cfg to pure dicts so they can be saved using physicsnemo
    model = instantiate(cfg.model, _convert_="all")
    model.batch_size = cfg.batch_size
    model.learning_rate = cfg.learning_rate

    # Instantiate PyTorch modules (with state dictionaries from checkpoint if given)
    criterion = instantiate(cfg.trainer.criterion)
    optimizer = instantiate(cfg.trainer.optimizer, params=model.parameters())
    lr_scheduler = (
        instantiate(cfg.trainer.lr_scheduler, optimizer=optimizer)
        if cfg.trainer.lr_scheduler is not None
        else None
    )

    # setup startup values
    epoch = 1
    val_error = th.inf
    iteration = 0
    epochs_since_improved = 0

    # Prepare training under consideration of checkpoint if given
    if cfg.get("checkpoint_name", None) is not None:
        checkpoint_path = Path(
            cfg.get("output_dir"),
            "tensorboard",
            "checkpoints",
            "training-state-" + cfg.get("checkpoint_name") + ".mdlus",
        )
        optimizer_path = Path(
            cfg.get("output_dir"),
            "tensorboard",
            "checkpoints",
            "optimizer-state-" + cfg.get("checkpoint_name") + ".ckpt",
        )
        if checkpoint_path.exists():
            logger0.info(f"Loading checkpoint: {checkpoint_path}")
            model = Module.from_checkpoint(str(checkpoint_path))
            checkpoint = th.load(optimizer_path, map_location=dist.device)
            if not cfg.get("load_weights_only"):
                # Load optimizer
                optimizer = instantiate(
                    cfg.trainer.optimizer, params=model.parameters()
                )
                optimizer_state_dict = checkpoint["optimizer_state_dict"]
                optimizer.load_state_dict(optimizer_state_dict)
                # Move tensors to the appropriate device as in https://github.com/pytorch/pytorch/issues/2830
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if th.is_tensor(v):
                            state[k] = v.to(device=dist.device)
                # Optionally load scheduler
                if cfg.trainer.lr_scheduler is not None:
                    lr_scheduler = instantiate(
                        cfg.trainer.lr_scheduler, optimizer=optimizer
                    )
                    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                else:
                    lr_scheduler = None
            epoch = checkpoint["epoch"]
            val_error = checkpoint["val_error"]
            iteration = checkpoint["iteration"]
            epochs_since_improved = (
                checkpoint["epochs_since_improved"]
                if "epochs_since_improved" in checkpoint.keys()
                else 0
            )
        else:
            logger0.info(
                f"Checkpoint not found, weights not loaded. Requested path: {checkpoint_path}"
            )

    # Instantiate the trainer and fit the model
    logger0.info("Model initialized")
    trainer = instantiate(
        cfg.trainer,
        model=model,
        data_module=data_module,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=dist.device,
    )
    logger0.info(f"starting training")
    trainer.fit(
        epoch=epoch,
        validation_error=val_error,
        iteration=iteration,
        epochs_since_improved=epochs_since_improved,
    )


if __name__ == "__main__":
    train()
    print("Done.")

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

import glob
import logging
import os
import re

import numpy as np
import torch as th

logger = logging.getLogger(__name__)


# TODO switch over to physicsnemo checkpoint system
def write_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    iteration: int,
    val_error: float,
    epochs_since_improved: int,
    dst_path: str,
    keep_n_checkpoints: int = 5,
):
    """
    Writes a checkpoint including model, optimizer, and scheduler state dictionaries along with current epoch,
    iteration, and validation error to file.

    :param model: The network model
    :param optimizer: The pytorch optimizer
    :param scheduler: The pytorch learning rate scheduler
    :param epoch: Current training epoch
    :param iteration: Current training iteration
    :param val_error: The validation error of the current training
    :param epochs_since_improved: The number of epochs since the validation error improved
    :param dst_path: Path where the checkpoint is written to
    :param keep_n_checkpoints: Number of best checkpoints that will be saved (worse checkpoints are overwritten)
    """
    root_path = os.path.join(
        dst_path,
        "checkpoints",
    )
    # root_path = os.path.dirname(ckpt_dst_path)
    ckpt_dst_path = os.path.join(
        root_path,
        f"training-state-epoch-{str(epoch).zfill(4)}-val_loss="
        + "{:.4E}".format(val_error)
        + ".mdlus",
    )
    os.makedirs(root_path, exist_ok=True)

    model.save(ckpt_dst_path)
    model.save(os.path.join(root_path, "training-state-last.mdlus"))

    opt_dst_path = os.path.join(
        root_path,
        f"optimizer-state-epoch-{str(epoch).zfill(4)}-val_loss="
        + "{:.4E}".format(val_error)
        + ".ckpt",
    )
    th.save(
        obj={
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "iteration": iteration,
            "val_error": val_error,
            "epochs_since_improved": epochs_since_improved,
        },
        f=opt_dst_path,
    )
    th.save(
        obj={
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "iteration": iteration,
            "val_error": val_error,
            "epochs_since_improved": epochs_since_improved,
        },
        f=os.path.join(root_path, "optimizer-state-last.ckpt"),
    )

    # Only keep top n checkpoints
    ckpt_paths = np.array(glob.glob(root_path + "/training-state-epoch-*.mdlus"))
    if len(ckpt_paths) > keep_n_checkpoints + 1:
        worst_path = ""
        worst_error = -np.infty
        for ckpt_path in ckpt_paths:
            if "NAN" in ckpt_path:
                os.remove(ckpt_path)
                try:
                    os.remove(ckpt_path.replace("training", "optimizer"))
                except FileNotFoundError:
                    pass
                continue
            # Read the scientific number from the checkpoint name and perform update if appropriate
            curr_error = float(
                re.findall("-?\d*\.?\d+E[+-]?\d+", os.path.basename(ckpt_path))[0]
            )
            if curr_error > worst_error:
                worst_path = ckpt_path
                worst_error = curr_error
        os.remove(worst_path)
        try:
            os.remove(
                worst_path.replace("training", "optimizer").replace("mdlus", "ckpt")
            )
        except FileNotFoundError:
            pass

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

import torch
import os
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
import dgl
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch.nn.functional as F
from physicsnemo.models.meshgraphnet import MeshGraphNet
import matplotlib.pyplot as plt
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import LaunchLogger, PythonLogger
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.distributed import DistributedManager

from utils import (
    create_datasets,
    _custom_collate,
    get_rotation_matrix,
    create_edges,
    compute_mean_var,
)


def prepare_input(
    pos: np.ndarray,
    forces: np.ndarray,
    box_size: float,
    rotation_matrix: np.ndarray = None,
    add_random_noise: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform transformations on the input for data augmentation

    Parameters
    ----------
    pos : np.ndarray
        Coordinates of the atoms. [N, 3]
    forces : np.ndarray
        True force components on each atom. [N, 3]
    box_size : float
        Bounding box for the periodic domain
    rotation_matrix : np.ndarray, optional
        Rotation matrix to rotate the coordinates and forces. [3, 3], by default None
    add_random_noise : bool, optional
        Whether to add a random displacement to the coordinates, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Transformed coordinates and forces
    """

    pos = np.mod(pos, box_size)
    off = np.mean(pos, axis=0)

    # Rotate the whole system. As the interatomic distance remains unchanged,
    # the forces can just be rotated using the same transformation
    if rotation_matrix is not None:
        pos = pos - off
        pos = np.matmul(pos, rotation_matrix)
        pos += off
        forces = np.matmul(forces, rotation_matrix)

    if add_random_noise:
        pos = pos + np.random.randn(*pos.shape) * 0.005
        pos = np.mod(pos, box_size)
    off_2 = np.min(pos, axis=0)
    pos = pos - off_2

    return pos, forces


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    dataset, test_dataset = create_datasets(
        to_absolute_path(os.path.join("./", "lj_data")), test_size=0.1
    )

    if dist.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset,
            num_workers=1,
            batch_size=1,
            sampler=sampler,
            collate_fn=_custom_collate,
        )
    else:
        dataloader = DataLoader(
            dataset,
            num_workers=1,
            batch_size=1,
            shuffle=True,
            collate_fn=_custom_collate,
        )

    test_dataloader = DataLoader(
        test_dataset,
        num_workers=1,
        batch_size=1,
        shuffle=True,
        collate_fn=_custom_collate,
    )

    model = MeshGraphNet(
        input_dim_nodes=cfg.model.input_dim_nodes,
        input_dim_edges=cfg.model.input_dim_edges,
        output_dim=cfg.model.output_dim,
        processor_size=cfg.model.processor_size,
        mlp_activation_fn=cfg.model.mlp_activation_fn,
        num_layers_node_processor=cfg.model.num_layers_node_processor,
        num_layers_edge_processor=cfg.model.num_layers_edge_processor,
        num_layers_node_encoder=None,  # No node encoder
        num_layers_node_decoder=cfg.model.num_layers_node_decoder,
        hidden_dim_edge_encoder=cfg.model.hidden_dim_edge_encoder,
    ).to(dist.device)

    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr.start_lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr.gamma)

    LaunchLogger.initialize(use_mlflow=True)

    # define constants
    distance_threshold = cfg.distance_threshold
    box_size = cfg.box_size
    force_mean, force_sd = compute_mean_var(
        to_absolute_path(os.path.join("./", "lj_data"))
    )

    # Attempt to load latest checkpoint if one exists
    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    for epoch in range(max(1, loaded_epoch + 1), cfg.max_epochs + 1):
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(dataloader), epoch_alert_freq=1
        ) as log:
            model.train()
            for data in dataloader:
                # Create edges
                pos = data[0][
                    0
                ]  # Select first element of the list (works only for batchsize 1)
                forces = data[1][0]
                r = get_rotation_matrix()

                pos, forces = prepare_input(
                    pos, forces, box_size, r, add_random_noise=True
                )
                src, dst, edge_features = create_edges(
                    pos, distance_threshold, box_size
                )
                g = dgl.graph((src, dst)).to(dist.device)

                node_fea = torch.ones(
                    size=(pos.shape[0], cfg.model.hidden_dim_edge_encoder)
                ).to(dist.device)
                edge_fea = (
                    torch.tensor(np.array(edge_features), dtype=torch.float32)
                    .view(-1, 4)
                    .to(dist.device)
                )

                out = model(node_fea, edge_fea, g)
                true_out = torch.tensor(
                    (forces - force_mean) / force_sd, dtype=torch.float32
                ).to(dist.device)

                optimizer.zero_grad()

                # L1 loss to encourage network to learn minimal message-passing required
                # for force prediction.
                # Regularization of penalize the total sum of forces.
                loss = F.l1_loss(out, true_out) + 0.001 * torch.mean(out).abs()
                loss.backward()
                optimizer.step()
                scheduler.step()
                log.log_minibatch({"Mini-batch loss": loss.detach()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if dist.rank == 0:
            save_checkpoint(
                "./checkpoints",
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

            with LaunchLogger("valid", epoch=epoch) as log:
                with torch.no_grad():
                    model.eval()
                    val_loss = 0

                    forces_pair = []
                    cosines = []
                    for data in test_dataloader:
                        pos = data[0][0]
                        forces = data[1][0]
                        pos, forces = prepare_input(
                            pos,
                            forces,
                            box_size,
                            rotation_matrix=None,
                            add_random_noise=False,
                        )

                        # Create edges
                        src, dst, edge_features = create_edges(
                            pos, distance_threshold, box_size
                        )
                        g = dgl.graph((src, dst)).to(dist.device)
                        node_fea = torch.ones(
                            size=(pos.shape[0], cfg.model.hidden_dim_edge_encoder)
                        ).to(dist.device)
                        edge_fea = (
                            torch.tensor(np.array(edge_features), dtype=torch.float32)
                            .view(-1, 4)
                            .to(dist.device)
                        )

                        out = model(node_fea, edge_fea, g)
                        true_out = torch.tensor(
                            (forces - force_mean) / force_sd, dtype=torch.float32
                        ).to(dist.device)

                        val_loss += F.mse_loss(out, true_out).detach()

                        out_np = out.detach().cpu().numpy()
                        true_out_np = true_out.detach().cpu().numpy()
                        forces_pair.append((out_np, true_out_np))

                        # Compute the angle of predicted forces
                        dot_product = np.sum(out_np * true_out_np, axis=1)
                        out_np_mag = np.linalg.norm(out_np, axis=1)
                        true_out_np_mag = np.linalg.norm(true_out_np, axis=1)

                        cosine = dot_product / (out_np_mag * true_out_np_mag)
                        cosines.append(cosine)

                    plt.clf()
                    plt.figure(figsize=(5, 5))

                    # Compute the total force vector
                    for force_system in forces_pair:
                        pred, true = force_system
                        pred_total = pred[:, 0] + pred[:, 1] + pred[:, 2]
                        true_total = true[:, 0] + true[:, 1] + true[:, 2]
                        pred_total = (pred_total * force_sd + force_mean) / 1000
                        true_total = (true_total * force_sd + force_mean) / 1000
                        plt.scatter(pred_total, true_total, s=5, c="black")

                    cosine_percentage = (
                        np.concatenate(cosines, axis=0) > 0.995
                    ).sum() / np.concatenate(cosines, axis=0).shape[0]

                    # plot y=x line
                    x = np.linspace(-0.5, 0.5, 50)
                    y = x
                    plt.plot(x, y, color="blue", linestyle="--")

                    plt.text(
                        1,
                        -19,
                        f"Cosine Percentage: {round(cosine_percentage, 3)}",
                        fontsize=8,
                    )
                    plt.xlim([-0.5, 0.5])
                    plt.ylim([-0.5, 0.5])
                    plt.gca().set_aspect("equal")
                    plt.savefig(f"results_figure_{epoch}.png")

                    log.log_epoch({"Validation loss": val_loss})
                    log.log_epoch({"Cosine percentage": cosine_percentage})


if __name__ == "__main__":
    main()

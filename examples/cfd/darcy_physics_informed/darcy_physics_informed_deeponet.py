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

from itertools import chain
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.utils.checkpoint import save_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils import HDF5MapStyleDataset


def validation_step(graph, dataloader, epoch):
    """Validation Step"""

    with torch.no_grad():
        loss_epoch = 0
        for data in dataloader:
            invar, outvar, x_invar, y_invar = data
            out = graph.forward(
                {"k_prime": invar[:, 0].unsqueeze(dim=1), "x": x_invar, "y": y_invar}
            )

            deepo_out_u = out["u"]

            loss_epoch += F.mse_loss(outvar, deepo_out_u)

        # convert data to numpy
        outvar = outvar.detach().cpu().numpy()
        predvar = deepo_out_u.detach().cpu().numpy()

        # plotting
        fig, ax = plt.subplots(1, 3, figsize=(25, 5))

        d_min = np.min(outvar[0, 0])
        d_max = np.max(outvar[0, 0])

        im = ax[0].imshow(outvar[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(predvar[0, 0], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(np.abs(predvar[0, 0] - outvar[0, 0]))
        plt.colorbar(im, ax=ax[2])

        ax[0].set_title("True")
        ax[1].set_title("Pred")
        ax[2].set_title("Difference")

        fig.savefig(f"results_{epoch}.png")
        plt.close()
        return loss_epoch / len(dataloader)


class MdlsSymWrapper(Arch):
    """
    Wrapper model to convert PhysicsNeMo model to PhysicsNeMo-Sym model.

    PhysicsNeMo Sym relies on the inputs/outputs of the model being dictionary of tensors.
    This wrapper converts the input dictionary of tensors to a tensor inputs that can
    be processed by the PhysicsNeMo model that operate on tensors. Appropriate
    transformations are performed in the forward pass of the model to translate between
    these two input/output definitions.

    These transformations can differ based on the models. For e.g. typically for a fully
    connected network, the input tensors are combined by concatenating them along
    appropriate dimension before passing them as an input to the PhysicsNeMo model.
    During the output, the process is reversed, the output tensor from pytorch model is
    split across appropriate dimensions and then converted to a dictionary with
    appropriate keys to produce the final output.

    Having the model wrapped in a wrapper like this allows gradient computation using
    the PhysicsNeMo Sym's optimized gradient computing backend.

    For more details on PhysicsNeMo Sym models, refer:
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/tutorials/simple_training_example.html#using-custom-models-in-physicsnemo
    For more details on Key class, refer:
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.html#module-physicsnemo.sym.key
    """

    def __init__(
        self,
        input_keys=[Key("k"), Key("x"), Key("y")],
        output_keys=[Key("k_prime"), Key("u")],
        trunk_net=None,
        branch_net=None,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
        )

        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def forward(self, dict_tensor: Dict[str, torch.Tensor]):
        # Concatenate x, y inputs to feeed in the trunk network which has a MLP
        xy_input_shape = dict_tensor["x"].shape
        xy = self.concat_input(
            {
                k: dict_tensor[k].view(xy_input_shape[0], -1, 1) for k in ["x", "y"]
            },  # flatten the coordinate dimensions
            ["x", "y"],
            detach_dict=self.detach_key_dict,
            dim=-1,  # concat along the last dimension to form the feature vector.
        )
        fc_out = self.trunk_net(xy)

        # Pass the k-prime for the FNO input
        fno_out = self.branch_net(dict_tensor["k_prime"])

        # reshape the fc_out
        fc_out = fc_out.view(
            xy_input_shape[0], -1, xy_input_shape[-2], xy_input_shape[-1]
        )

        # multiply the outputs of branch and trunk networks to get the final output
        out = fc_out * fno_out

        return self.split_output(
            out, self.output_key_dict, dim=1
        )  # Split along the channel dimension to get a dictionary of tensors


@hydra.main(version_base="1.3", config_path="conf", config_name="config_deeponet.yaml")
def main(cfg: DictConfig):

    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LaunchLogger.initialize()

    # Use Diffusion equation for the Darcy PDE
    forcing_fn = 1.0 * 4.49996e00 * 3.88433e-03  # after scaling
    darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)

    dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/train.hdf5"), device=device
    )
    validation_dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/validation.hdf5"), device=device
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model_branch = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    )

    model_trunk = FullyConnected(
        in_features=cfg.model.fc.in_features,
        out_features=cfg.model.fc.out_features,
        layer_size=cfg.model.fc.layer_size,
        num_layers=cfg.model.fc.num_layers,
    )

    # Define k-prime as an auxiliary variable that is a copy of k.
    # Having k as the output of the model will allow gradients of k (for pde loss)
    # to be computed using Sym's gradient backend
    model = MdlsSymWrapper(
        input_keys=[Key("k_prime"), Key("x"), Key("y")],
        output_keys=[Key("k"), Key("u")],
        trunk_net=model_trunk,
        branch_net=model_branch,
    ).to(device)

    phy_informer = PhysicsInformer(
        required_outputs=["diffusion_u"],
        equations=darcy,
        grad_method="autodiff",
        device=device,
    )

    optimizer = torch.optim.Adam(
        chain(model_branch.parameters(), model_trunk.parameters()),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0,
        fused=True if torch.cuda.is_available() else False,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    for epoch in range(cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            for data in dataloader:
                optimizer.zero_grad()
                outvar = data[1]

                coords = torch.stack([data[2], data[3]], dim=1).requires_grad_(True)
                # compute forward pass
                out = model.forward(
                    {
                        "k_prime": data[0][:, 0].unsqueeze(dim=1),
                        "x": coords[:, 0:1],
                        "y": coords[:, 1:2],
                    }
                )

                residuals = phy_informer.forward(
                    {
                        "coordinates": coords,
                        "u": out["u"],
                        "k": out["k"],
                    }
                )
                pde_out_arr = residuals["diffusion_u"]

                # Boundary condition
                pde_out_arr = F.pad(
                    pde_out_arr[..., 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))

                # Compute data loss
                deepo_out_u = out["u"]
                deepo_out_k = out["k"]
                loss_data = F.mse_loss(outvar, deepo_out_u) + F.mse_loss(
                    data[0][:, 0].unsqueeze(dim=1), deepo_out_k
                )

                # Compute total loss
                loss = loss_data + cfg.physics_weight * loss_pde

                # Backward pass and optimizer and learning rate update
                loss.backward()
                optimizer.step()
                scheduler.step()
                log.log_minibatch(
                    {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach()}
                )

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(model, validation_dataloader, epoch)
            log.log_epoch({"Validation error": error})

        save_checkpoint(
            "./checkpoints",
            models=[model_branch, model_trunk],
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()

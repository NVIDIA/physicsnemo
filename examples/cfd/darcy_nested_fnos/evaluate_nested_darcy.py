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

import hydra
from torch import cat, FloatTensor
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.fno import FNO
from physicsnemo.utils import StaticCaptureEvaluateNoGrad
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.utils import load_checkpoint

from utils import NestedDarcyDataset, PlotNestedDarcy


def plot_assembled(perm, darc):
    """Utility for plotting"""
    headers = ["permeability", "darcy"]
    plt.rcParams.update({"font.size": 28})
    fig, ax = plt.subplots(1, 2, figsize=(15 * 2, 15), sharey=True)
    im = []
    im.append(ax[0].imshow(perm))
    im.append(ax[1].imshow(darc))

    for ii in range(len(im)):
        fig.colorbar(im[ii], ax=ax[ii], location="bottom", fraction=0.046, pad=0.04)
        ax[ii].set_title(headers[ii])

    fig.savefig(join("./", f"test_test.png"))


def EvaluateModel(
    cfg: DictConfig,
    model_name: str,
    norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
    parent_result: FloatTensor = None,
    log: PythonLogger = None,
):
    """Utility for running inference on trained model"""
    # define model and load weights
    dist = DistributedManager()
    log.info(f"evaluating model {model_name}")
    model_cfg = cfg.arch[model_name]
    model = FNO(
        in_channels=model_cfg.fno.in_channels,
        out_channels=model_cfg.decoder.out_features,
        decoder_layers=model_cfg.decoder.layers,
        decoder_layer_size=model_cfg.decoder.layer_size,
        dimension=model_cfg.fno.dimension,
        latent_channels=model_cfg.fno.latent_channels,
        num_fno_layers=model_cfg.fno.fno_layers,
        num_fno_modes=model_cfg.fno.fno_modes,
        padding=model_cfg.fno.padding,
    ).to(dist.device)
    load_checkpoint(
        path=f"./checkpoints/best/{model_name}", device=dist.device, models=model
    )

    # prepare data for inference
    dataset = NestedDarcyDataset(
        mode="eval",
        data_path=cfg.inference.inference_set,
        model_name=model_name,
        norm=norm,
        log=log,
        parent_prediction=parent_result,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.inference.batch_size, shuffle=False)
    with open_dict(cfg):
        cfg.ref_fac = dataset.ref_fac
        cfg.fine_res = dataset.fine_res
        cfg.buffer = dataset.buffer

    # store positions of insets if refinement level > 0, ie if not global model
    if int(model_name[-1]) > 0:
        pos = dataset.position
    else:
        pos = None

    # define forward method
    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    # evaluate and invert normalisation
    invars, result = [], []
    for batch in dataloader:
        invars.append(batch["permeability"])
        result.append(forward_eval(batch["permeability"]))
    invars = cat(invars, dim=0).detach()
    result = cat(result, dim=0).detach()

    return pos, invars, result


def AssembleSolutionToDict(cfg: DictConfig, perm: dict, darcy: dict, pos: dict):
    """Assemble solution to easily interpretable dict"""
    dat, idx = {}, 0
    for ii in range(perm["ref0"].shape[0]):
        samp = str(ii)
        dat[samp] = {
            "ref0": {
                "0": {
                    "permeability": perm["ref0"][ii, 0, ...],
                    "darcy": darcy["ref0"][ii, 0, ...],
                }
            }
        }

        # insets
        dat[samp]["ref1"] = {}
        for ins, ps in pos["ref1"][samp].items():
            dat[samp]["ref1"][ins] = {
                "permeability": perm["ref1"][idx, 1, ...],
                "darcy": darcy["ref1"][idx, 0, ...],
                "pos": ps,
            }
            idx += 1

    if cfg.inference.save_result:
        np.save(
            "./nested_darcy_results.npy",
            dat,
        )
    return dat


def AssembleToSingleField(cfg: DictConfig, dat: dict):
    """Assemble multiple fields to a single dict"""
    ref_fac = cfg.ref_fac
    glob_size = dat["0"]["ref0"]["0"]["darcy"].shape[0]
    inset_size = dat["0"]["ref1"]["0"]["darcy"].shape[0]
    size = ref_fac * glob_size
    min_offset = (cfg.fine_res * (ref_fac - 1) + 1) // 2 + cfg.buffer * ref_fac

    perm = np.zeros((len(dat), size, size), dtype=np.float32)
    darc = np.zeros_like(perm)
    for ii, (_, field) in enumerate(dat.items()):
        # extract global premeability and expand to size x size
        perm[ii, ...] = np.kron(
            field["ref0"]["0"]["permeability"],
            np.ones((ref_fac, ref_fac), dtype=field["ref0"]["0"]["permeability"].dtype),
        )
        darc[ii, ...] = np.kron(
            field["ref0"]["0"]["darcy"],
            np.ones((ref_fac, ref_fac), dtype=field["ref0"]["0"]["darcy"].dtype),
        )

        # overwrite refined regions
        for __, inset in field["ref1"].items():
            pos = inset["pos"] * ref_fac + min_offset
            perm[
                ii, pos[0] : pos[0] + inset_size, pos[1] : pos[1] + inset_size
            ] = inset["permeability"]
            darc[
                ii, pos[0] : pos[0] + inset_size, pos[1] : pos[1] + inset_size
            ] = inset["darcy"]

    return {"permeability": perm, "darcy": darc}, ref_fac


def GetRelativeL2(pred, tar):
    """Compute L2 error"""
    div = 1.0 / tar["darcy"].shape[0] * tar["darcy"].shape[1]
    err = pred["darcy"] - tar["darcy"]

    l2_tar = np.sqrt(np.einsum("ijk,ijk->i", tar["darcy"], tar["darcy"]) * div)
    l2_err = np.sqrt(np.einsum("ijk,ijk->i", err, err) * div)

    return np.mean(l2_err / l2_tar)


def ComputeErrorNorm(cfg: DictConfig, pred_dict: dict, log: PythonLogger, ref0_pred):
    """Compute relative L2-norm of error"""
    # assemble ref1 and ref2 solutions alongside gound truth to single scalar field
    log.info("computing relative L2-norm of error...")
    tar_dict = np.load(cfg.inference.inference_set, allow_pickle=True).item()["fields"]
    pred, ref_fac = AssembleToSingleField(cfg, pred_dict)
    tar = AssembleToSingleField(cfg, tar_dict)[0]

    assert np.all(
        tar["permeability"] == pred["permeability"]
    ), "Permeability from file is not equal to analysed permeability"

    # compute l2 norm of error
    rel_l2_err = GetRelativeL2(pred, tar)
    log.log(f"    ...which is {rel_l2_err}.")

    if cfg.inference.get_ref0_error_norm:
        ref0_pred = np.kron(
            ref0_pred, np.ones((ref_fac, ref_fac), dtype=ref0_pred.dtype)
        )
        rel_l2_err = GetRelativeL2({"darcy": ref0_pred}, tar)
        log.log(f"The error with ref_0 only would be {rel_l2_err}.")

    return


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def nested_darcy_evaluation(cfg: DictConfig) -> None:
    """Inference of the nested 2D Darcy flow benchmark problem.

    This inference script consecutively evaluates the models of nested FNO for the
    nested Darcy problem, taking into account the result of the model associated
    with the parent level. All results are stored in a numpy file and a selection
    of samples can be plotted in the end.
    """
    # initialize monitoring, models and normalisation
    DistributedManager.initialize()  # Only call this once in the entire script!
    log = PythonLogger(name="darcy_fno")

    model_names = sorted(list(cfg.arch.keys()))
    norm = {
        "permeability": (
            cfg.normaliser.permeability.mean,
            cfg.normaliser.permeability.std,
        ),
        "darcy": (cfg.normaliser.darcy.mean, cfg.normaliser.darcy.std),
    }

    # evaluate models and revoke normalisation
    perm, darcy, pos, result, ref0_pred = {}, {}, {}, None, None
    for name in model_names:
        position, invars, result = EvaluateModel(cfg, name, norm, result, log)
        perm[name] = (
            (invars * norm["permeability"][1] + norm["permeability"][0])
            .detach()
            .cpu()
            .numpy()
        )
        darcy[name] = (
            (result * norm["darcy"][1] + norm["darcy"][0]).detach().cpu().numpy()
        )
        pos[name] = position

        if cfg.inference.get_ref0_error_norm and int(name[-1]) == 0:
            ref0_pred = np.copy(darcy[name]).squeeze()

    # port solution format to dict structure like in input files
    pred_dict = AssembleSolutionToDict(cfg, perm, darcy, pos)

    # compute error norm
    if cfg.inference.get_error_norm:
        ComputeErrorNorm(cfg, pred_dict, log, ref0_pred)

    # plot some fields
    if cfg.inference.n_plots > 0:
        log.info("plotting results")
        for idx in range(cfg.inference.n_plots):
            PlotNestedDarcy(pred_dict, idx)


if __name__ == "__main__":
    nested_darcy_evaluation()

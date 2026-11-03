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

"""Generate samples from a trained diffusion model for any dataset.

Uses the training config to build the validation dataset and the model
conditions, generates an ensemble of samples for each requested dataset index,
and saves the denormalized predictions and targets to a Zarr store, optionally
with plots. The inference options are in ``config/inference/default.yaml``.
Example::

    python inference.py --config-name reference_downscaling +inference=default \\
        inference.diffusion_checkpoint=<ckpt_dir>/EDMPreconditioner.0.<step>.mdlus \\
        inference.indices=[0,10,20] inference.ensemble_size=4

Other config overrides, e.g. ``sampler.args.num_steps=32``, can be added as usual.
"""

import os

import hydra
import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from physicsnemo.core import Module
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler

from datasets import dataset_classes
from utils.config import MainConfig
from utils.nn import (
    build_network_condition_and_target,
    diffusion_model_forward,
    unpack_batch,
)
from utils.plots import inference_plot


@hydra.main(version_base=None, config_path="config", config_name=None)
def main(cfg: DictConfig) -> None:
    """Generate and save samples as configured by ``cfg``; see the module docstring."""
    if "inference" not in cfg:
        raise ValueError(
            "Add the inference options with +inference=default, see "
            "config/inference/default.yaml"
        )
    inf_cfg = cfg.inference
    cfg = OmegaConf.to_container(cfg, resolve=True)
    del cfg["inference"]
    cfg = MainConfig(**cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(inf_cfg.seed)

    # Dataset, built as in training
    dataset_kwargs = cfg.dataset.__dict__.copy()
    dataset_cls = dataset_classes[dataset_kwargs.pop("name")]
    dataset_kwargs.pop("loader", None)
    dataset = dataset_cls(dataset_kwargs, train=False)
    invariants = dataset.get_invariants()
    if invariants is not None:
        invariants = torch.as_tensor(invariants, device=device)[None]
    state_channels = dataset.state_channels()
    unknown_vars = set(inf_cfg.plot_variables) - set(state_channels)
    if unknown_vars:
        raise ValueError(f"Unknown plot_variables {unknown_vars}: not state channels")

    # Models
    net = Module.from_checkpoint(inf_cfg.diffusion_checkpoint).eval().to(device)
    regression_net = None
    if "regression" in cfg.model.diffusion_conditions:
        regression_path = inf_cfg.regression_checkpoint or cfg.model.regression_weights
        regression_net = Module.from_checkpoint(regression_path).eval().to(device)
    elif inf_cfg.save_regression:
        raise ValueError("save_regression requires a regression+diffusion model")
    use_mask_tokens = cfg.model.architecture == "dit" and bool(
        cfg.model.hyperparameters.get("use_nan_mask_tokens", False)
    )

    sampler_args = cfg.sampler.args.__dict__
    scheduler = EDMNoiseScheduler(
        sigma_min=sampler_args.get("sigma_min", 0.002),
        sigma_max=sampler_args.get("sigma_max", 80.0),
        rho=sampler_args.get("rho", 7.0),
    )
    amp_dtype = {"amp-bf16": torch.bfloat16, "amp-fp16": torch.float16}.get(
        cfg.training.perf.fp_optimizations
    )

    os.makedirs(inf_cfg.output_dir, exist_ok=True)
    output_path = os.path.join(inf_cfg.output_dir, "samples.zarr")
    dims = ("y", "x")
    (lat, lon) = (dataset.latitude(), dataset.longitude())
    has_lat_lon = not (np.isnan(lat).all() or np.isnan(lon).all())
    with torch.no_grad():
        for i, idx in enumerate(inf_cfg.indices):
            print(f"Generating {inf_cfg.ensemble_size} sample(s) for index {idx}...")
            batch = torch.utils.data.default_collate([dataset[idx]])
            (background, target, mask, lead_time_label, scalar_conditions) = (
                unpack_batch(batch, device)
            )
            invalid_mask = None
            if use_mask_tokens and mask is not None:
                invalid_mask = (mask < 0.5).any(dim=1, keepdim=True)

            with torch.autocast(
                device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                (condition, _, reg_out) = build_network_condition_and_target(
                    background,
                    target,
                    invariants,
                    lead_time_label=lead_time_label,
                    scalar_conditions=scalar_conditions,
                    regression_net=regression_net,
                    condition_list=cfg.model.diffusion_conditions,
                    regression_condition_list=cfg.model.regression_conditions,
                )
                ensemble = []
                for _ in range(inf_cfg.ensemble_size):
                    pred = diffusion_model_forward(
                        net,
                        condition,
                        shape=target.shape,
                        scheduler=scheduler,
                        dtype=target.dtype,
                        device=device,
                        sampler_args=sampler_args,
                        lead_time_label=lead_time_label,
                        invalid_mask=invalid_mask,
                    ).float()
                    if reg_out is not None:
                        pred += reg_out.float()
                    ensemble.append(dataset.denormalize_state(pred.cpu().numpy())[0])

            prediction = np.stack(ensemble)
            target = dataset.denormalize_state(target.cpu().numpy())[0]

            # Save predictions and targets in physical units, appending along "index"
            data_vars = {
                "prediction": (
                    ("index", "ensemble", "channel", *dims),
                    prediction[None],
                ),
                "target": (("index", "channel", *dims), target[None]),
            }
            if inf_cfg.save_regression:
                regression = dataset.denormalize_state(reg_out.float().cpu().numpy())
                data_vars["regression"] = (("index", "channel", *dims), regression)
            ds = xr.Dataset(
                data_vars, coords={"index": [idx], "channel": state_channels}
            )
            if i == 0:
                if has_lat_lon:
                    ds = ds.assign_coords(lat=(dims, lat), lon=(dims, lon))
                ds.to_zarr(output_path, mode="w")
            else:
                ds.to_zarr(output_path, append_dim="index")

            for var in inf_cfg.plot_variables:
                c = state_channels.index(var)
                fig = inference_plot(prediction[0, c], target[c], f"{var}, index {idx}")
                fig.savefig(os.path.join(inf_cfg.output_dir, f"{var}_{idx}.png"))
                plt.close(fig)

    print(f"Saved samples to {output_path}")


if __name__ == "__main__":
    main()

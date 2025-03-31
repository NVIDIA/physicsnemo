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
import os
from physicsnemo.metrics.crps import kcrps
import netCDF4 as nc
import torch
import tqdm
import numpy as np
import csv
import shutil


def rank_(truth: torch.Tensor, img: torch.Tensor):
    img.sort(0)
    rank = torch.zeros_like(truth, dtype=torch.int32)
    for e in range(img.size(0)):
        rank += torch.where(img[e] < truth, 1, 0)

    e, b, x = img.shape
    counts = torch.zeros([b, e + 1], device=img.device)
    for r in range(img.size(0) + 1):
        counts[..., r] = torch.sum(rank == r, dim=-1)

    return counts


n_samples = 5000
conditional_samples = "/path/to/conditional_samples.nc"


# TODO store this hyperparameter per variable in netCDF
GAMMA = 0.1
# TODO save this metadata alongside model
u10_std = 2.372
v10_std = 4.115
logtp_std = 2.489


def update(t, v, pred, obs, writer, counts_output):
    """
    Compute and write CRPS, MAE, MSE, and rank histogram for a single variable at a single time.
    """
    mask = ~np.isnan(obs)

    pred = pred[:, mask]
    obs = obs[mask]

    # Strange errors occasionally happen when run on CPU
    #  Caught signal 8 (Floating point exception: integer divide by zero)
    # ==== backtrace (tid:2825527) ====
    #  0 0x0000000000042520 __sigaction()  ???:0
    #  1 0x0000000001944597 mkl_vml_serv_GetMinN()  ???:0
    pred = torch.as_tensor(pred).cuda()
    obs = torch.as_tensor(obs).cuda()

    if obs.size == 0:
        return

    if v == "10u":
        pseudo_obs = pred + u10_std * GAMMA * torch.randn_like(pred)
    elif v == "10v":
        pseudo_obs = pred + v10_std * GAMMA * torch.randn_like(pred)
    elif v == "tp":
        logtp = torch.log(pred)  # TODO handle small eps factor
        logtp += logtp_std * GAMMA * torch.randn_like(pred)
        pseudo_obs = logtp.exp()

    # is it averaged spatially
    # TODO clarify in paper that metrics are computed as spatial averages
    # of the available stations (I think)
    # maybe better to copmute average skills for each station and then
    # average

    # crps
    crps = kcrps(pseudo_obs, obs, biased=False)
    crps = crps.mean()

    # # variance (ddof = 1)
    # # unbiased variance w/ 1/(e-1)
    # # s_t^2 in Fortin 2014
    if pseudo_obs.size(0) > 1:
        variance = pseudo_obs.var(correction=1, dim=0).mean()
        writer.writerow([t, v, "variance", variance.cpu().item()])

    # MAE mean
    x = torch.mean(torch.abs((pred.mean(0) - obs)))
    writer.writerow([t, v, "mae_mean", x.cpu().item()])

    x = torch.mean(torch.abs((pred - obs)))
    writer.writerow([t, v, "mae_single", x.cpu().item()])

    # MSE of ensemble mean
    # See equation (3) Fortin
    mse_mean = torch.mean((pred.mean(0) - obs) ** 2)

    # Average MSE of single members
    mse_single = torch.mean((pred - obs) ** 2)

    # rank histogram
    counts = rank_(obs.unsqueeze(0), pseudo_obs.unsqueeze(1))
    counts = counts.squeeze(0)

    # # TODO timestamp is missing from data neal saved
    writer.writerow([t, v, "crps", crps.cpu().item()])
    writer.writerow([t, v, "mse_mean", mse_mean.cpu().item()])
    writer.writerow([t, v, "mse_single", mse_single.cpu().item()])
    counts_output[v] = counts_output.get(v, 0) + counts.cpu().numpy()


def main(input_path, output_path, prediction_group_name: str):

    if os.path.isdir(output_path):
        return

    tmpout = output_path + ".tmp"
    os.makedirs(tmpout, exist_ok=True)
    output_f = open(os.path.join(tmpout, "scores.csv"), "w")
    writer = csv.writer(output_f)

    with nc.Dataset(input_path) as f:
        variables = list(f.groups["truth"].variables)
        times = range(f.dimensions["time"].size)

        counts_output = {}
        try:
            predgroup = f.groups[prediction_group_name]
        except KeyError:
            raise ValueError(f"expected one of {list(f.groups)}")
        for t in tqdm.tqdm(times):
            for v in variables:

                predv = predgroup.variables[v]
                if len(predv.dimensions) == 3:
                    pred = predv[t].data[None]
                else:
                    pred = predv[:, t].data

                obs = f.groups["validation"].variables[v][t].data
                update(t, v, pred, obs, writer, counts_output)

    np.savez(os.path.join(tmpout, "counts"), **counts_output)
    shutil.move(tmpout, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument(
        "-g",
        "--group",
        type=str,
        help="netCDF group used for prediction.",
        default="prediction",
    )
    args = parser.parse_args()
    main(
        input_path=conditional_samples,
        output_path=args.output,
        prediction_group_name=args.group,
    )

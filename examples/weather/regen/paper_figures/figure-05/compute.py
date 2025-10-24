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

import numpy as np
import matplotlib.pyplot as plt
import glob


files_hr = glob.glob("eval_mses/more_rand_mse_for_eval_hr_leave_*.npy")
files_observed = glob.glob("eval_mses/more_rand_mse_for_eval_leave_*.npy")


files_hr = glob.glob("eval_mses/redu_mse_for_eval_hr_leave_*.npy")
files_observed = glob.glob("eval_mses/redu_mse_for_eval_leave_*.npy")


files_hr.sort()
files_observed.sort()


files_hr


mses_hr = []
for f in files_hr:
    e = np.load(f)
    mse = e.mean(axis=0)
    mses_hr.append(mse)
mses_hr = np.array(mses_hr)


mses_observed = []
for f in files_observed:
    e = np.load(f)
    mse = e.mean(axis=0)
    mses_observed.append(mse)
mses_observed = np.array(mses_observed)


mses_hr


cs = ["C00", "C01", "C03"]


pos = np.concatenate([np.linspace(1, 15, 15), np.linspace(46, 50, 5)])


stds_observed = []
for f in files_observed:
    e = np.load(f)
    std = e.std(axis=0)
    stds_observed.append(std)
std_observed = np.array(stds_observed)
err = np.flip(np.sqrt(std_observed), axis=0) / np.sqrt(len(e))


ticks = np.flip(
    50
    - np.concatenate(
        [np.linspace(3, 45, 15, dtype=int), np.linspace(46, 50, 5, dtype=int)]
    )
)
mses_observed = np.flip(mses_observed, axis=0)
mses_hr = np.flip(mses_hr, axis=0)


plt.figure(figsize=(8, 5))
for i in range(3):
    plt.plot(ticks, np.sqrt(mses_observed[:, i]), c=cs[i])
    plt.plot(
        ticks, np.sqrt(mses_hr[:, i]), c=cs[i], linestyle="dotted", label="_nolegend_"
    )
    # plt.plot(ticks, np.sqrt(mses_observed[:,i]), c = cs[i])
    plt.fill_between(
        ticks,
        np.sqrt(mses_observed[:, i]) + err[:, i],
        np.sqrt(mses_observed[:, i]) - err[:, i],
        alpha=0.1,
        color=cs[i],
        label="_nolegend_",
    )

# plt.ylim([ 0,1.6])
plt.legend(
    [
        "10u",
        "10v",
        "tp",
    ]
)
plt.ylabel("RMSE")
plt.xlabel("No. stations included for inference")
x = plt.xticks(ticks, ticks)


np.sqrt(mses_observed[:, i])


# Load an example dataset with long-form data

# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal", hue="region", style="event", data=fmri)


err = np.sqrt(std_observed)


e = np.load("eval_mses/more_rand_mse_for_eval_leave_03.npy")
f = np.load("eval_mses/more_rand_mse_for_eval_leave_21.npy")
g = np.load("eval_mses/more_rand_mse_for_eval_hr_leave_03.npy")
h = np.load("eval_mses/more_rand_mse_for_eval_hr_leave_21.npy")


plt.figure(figsize=(20, 10))
plt.plot(e[:, 0], c="C00")
plt.plot(f[:, 0], c="C00", linestyle="-.")
plt.plot(g[:, 0], c="C01")
plt.plot(h[:, 0], c="C01", linestyle="-.")
plt.ylim([0, 2])
plt.legend(
    [
        "eval pred on 4 stat",
        "eval pred on 20 stat",
        "eval hrrr on 4 stat",
        "eval hrrr on 20 stat",
    ]
)
plt.xlabel("day of year 2017")
plt.ylabel("RMSE")


e = np.load("tuning_std/rands_mse_for_eval_hr.npy")
e.mean(axis=0)


e = np.load("tuning_std/rands_mse_for_eval.npy")
e.mean(axis=0)


import xarray as xr


from utils import find_takeout, find_takeout_random


stat_loc = xr.open_dataarray("station_locations_on_grid.nc")


num_leave = 30
valid = []
for _ in range(num_leave):
    bool_array = stat_loc.values.astype(bool)
    for indices in valid:
        bool_array[indices[0], indices[1]] = False
    valid.append(find_takeout_random(bool_array))


new = [tuple(row) for row in valid]
print(len(np.unique(new, axis=0)), len(valid))


np.save("random_val_stations", np.array(valid))


num_leave = 30
valid = []
for _ in range(num_leave):
    bool_array = stat_loc.values.astype(bool)
    for indices in valid:
        bool_array[indices[0], indices[1]] = False
    valid.append(find_takeout(bool_array))


np.save("redundant_val_stations", np.array(valid))


valid = np.load("more_random_val_stations.npy")


bool_array = stat_loc.values.astype(bool)
for indices in valid:
    bool_array[indices[0], indices[1]] = False
tune = np.zeros_like(bool_array)
for indices in valid:
    tune[indices[0], indices[1]] = True


e = np.load("eval_mses/redu_mse_for_eval_leave_04.npy")
f = np.load("eval_mses/redu_mse_for_eval_leave_20.npy")
g = np.load("eval_mses/redu_mse_for_eval_hr_leave_04.npy")
h = np.load("eval_mses/redu_mse_for_eval_hr_leave_20.npy")


plt.figure(figsize=(20, 10))
plt.plot(e[:, 0], c="C00")
plt.plot(f[:, 0], c="C00", linestyle="-.")
plt.plot(g[:, 0], c="C01")
plt.plot(h[:, 0], c="C01", linestyle="-.")
plt.ylim([0, 2])
plt.legend(
    [
        "eval pred on 4 stat",
        "eval pred on 20 stat",
        "eval hrrr on 4 stat",
        "eval hrrr on 20 stat",
    ]
)
plt.xlabel("day of year 2017")
plt.ylabel("RMSE")


files_hr = glob.glob("eval_mses/rand_pseudo_mse_for_eval_hr_leave_*.npy")
files_observed = glob.glob("eval_mses/rand_pseudo_mse_for_eval_leave_*.npy")

files_hr.sort()
files_observed.sort()


mses_hr = []
for f in files_hr:
    e = np.load(f)
    mse = e.mean(axis=0)
    mses_hr.append(mse)
mses_hr = np.array(mses_hr)
mses_observed = []
for f in files_observed:
    e = np.load(f)
    mse = e.mean(axis=0)
    mses_observed.append(mse)
mses_observed = np.array(mses_observed)


plt.figure(figsize=(8, 5))
for i in range(3):
    plt.plot(ticks, np.sqrt(mses_observed[:, i]), c=cs[i])
    plt.plot(ticks, np.sqrt(mses_hr[:, i]), c=cs[i], linestyle="dotted")
plt.ylim([0, 1.6])
plt.legend(["10u", "10u HRRR", "10v", "10v HRRR", "tp", "tp HRRR"])
plt.ylabel("RMSE")
plt.xlabel("No. pseudo-stations left out for evaluation")
x = plt.xticks(ticks, ticks)


num_stat = [3, 5, 10, 15, 20, 25, 30, 35, 40, 43, 45, 46, 47, 48, 49, 50]


def gather_metric_files(metric, stat):
    """
    Gather the metric files for a given number of stations.
    """
    p = "/path/to/gridded_isd_oklahoma/sweep/"
    files = glob.glob(p + metric + stat + "_*")
    files.sort()
    loaded_arrays = []
    for file in files:
        timestep = np.load(file)
        loaded_arrays.append(timestep)
    return np.array(loaded_arrays)


all_data = []
for n in num_stat:
    all_data.append(gather_metric_files("MSE_OBS_denorm_", str(n)))
all_data = np.array(all_data)


all_hrdata = []
for n in num_stat:
    all_hrdata.append(gather_metric_files("MSE_HRR_denorm_", str(n)))
all_hrdata = np.array(all_hrdata)


y = np.sqrt(all_data.mean(axis=1))
y_h = np.sqrt(all_hrdata.mean(axis=1))
err = np.sqrt(all_data.std(axis=1)) / np.sqrt(all_data.shape[1])
x = 50 - np.array(num_stat)


np.save("figure_data/numstatsweep/y.npy", y)
np.save("figure_data/numstatsweep/y_h.npy", y_h)
np.save("figure_data/numstatsweep/err.npy", err)
np.save("figure_data/numstatsweep/x.npy", x)

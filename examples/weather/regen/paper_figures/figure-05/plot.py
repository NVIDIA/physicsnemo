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
import matplotlib

matplotlib.rcParams.update({"font.size": 14})


y = np.load(
    "../figure_data/numstatsweep/y.npy",
)
y_h = np.load(
    "../figure_data/numstatsweep/y_h.npy",
)
err = np.load(
    "../figure_data/numstatsweep/err.npy",
)
x = np.load(
    "../figure_data/numstatsweep/x.npy",
)

fig = plt.figure(figsize=(15, 5))

for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.plot(x, y[:, i], c="C0", label="SDA")
    ax.plot(x, y_h[:, i], c="C0", linestyle="dotted", label="HRRR")
    # plt.plot(ticks, np.sqrt(mses_observed[:,i]), c = cs[i])
    ax.fill_between(
        x,
        y[:, i] + err[:, i],
        y[:, i] - err[:, i],
        alpha=0.1,
        color="C0",
        label="_nolegend_",
    )

    # ax.set_xticks(range(12), range(1,13))
    ax.set_ylabel(["RMSE [m/s]", "RMSE [m/s]", "RMSE [mm/h]"][i])
    ax.set_xlabel("No. stations included for inference")
    ax.set_title(["10u", "10v", "tp"][i])
    ax.legend(loc=1)
plt.tight_layout()

plt.savefig("../figures/numstatsweep.pdf", format="pdf", dpi=300)

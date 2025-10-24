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
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"saving figure to {path}.pdf ")
    plt.savefig(path + ".pdf")


SUBPANEL_WIDTH = 4
SUBPANEL_HEIGHT = 3

import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"

figsize = (SUBPANEL_WIDTH, SUBPANEL_HEIGHT)


def calculate_r_squared(x, y):
    # Compute the mean of y
    y_mean = np.mean(y)

    # Compute total sum of squares (TSS)
    tss = np.sum((y - y_mean) ** 2)

    # Compute residual sum of squares (RSS)
    rss = np.sum((y - x) ** 2)

    # Compute R-squared
    r_squared = 1 - (rss / tss)

    return r_squared


def read_score(output_path):
    return pd.read_csv(
        os.path.join(output_path, "scores.csv"),
        names=["t", "variable", "metric", "value"],
    )


def maybe_time(year, month, day, hour):
    try:
        return pd.Timestamp(year=year, month=month, day=day, hour=hour)
    except ValueError:
        return None


scoring_target = "figure_data/scores/paper"


def get_scores():
    """
    Get the scores for the SDA and HRRR models.
    """
    missing_samples = np.load("./figure_data/missing_samples.npy", allow_pickle=True)
    missing_samples = missing_samples.astype("datetime64[us]")
    time_index = pd.date_range("2017-01-01", periods=8640, freq=pd.Timedelta(1, "h"))
    time_missing = [t in missing_samples for t in time_index]
    time_df = pd.DataFrame(
        dict(t=np.arange(len(time_index)), missing=time_missing, time=time_index)
    )

    scores = {
        "sda": scoring_target,
        "hrrr": "figure_data/scores/hrrr",
    }

    dfs = []

    for source in scores:
        df = read_score(scores[source])
        df = df.join(time_df.set_index("t"), on="t")
        df = df[~df.missing]
        df["source"] = source
        dfs.append(df)

    df = pd.concat(dfs)
    return df


df = get_scores()
df = df.set_index(
    [
        "source",
        "time",
        "variable",
        "metric",
    ]
).value.unstack(["metric", "variable"])
df

# %%
PROJECT = "figure_data"
n_samples = 5000

output_path = scoring_target

# %%
for field in ["10u", "10v", "tp"]:
    plt.figure(figsize=figsize)
    p = df.loc["sda"]
    month = p.index.month
    sns.pointplot(x=month, y=np.sqrt(p["mse_mean"][field]), label="SDA Ensemble Mean")
    sns.pointplot(
        x=month,
        y=np.sqrt(p["mse_single"][field]),
        label="SDA Single Member",
    )
    p = df.loc["hrrr"]
    month = p.index.month
    sns.pointplot(
        x=month,
        y=np.sqrt(p["mse_single"][field]),
        label="HRRR",
    )
    plt.title(field)
    plt.ylabel("RMSE")
    plt.xlabel("Month")
    plt.legend()
    plt.tight_layout()
    savefig(f"figures/rmse/{field}")

# %%
num_ensemble = 15
# see eq 14 of Fortin  e.t al (2014)
spread = df["variance"] * (num_ensemble + 1) / num_ensemble
skill = df["mse_mean"]


# %%
def plot_spread_skill(spread, skill, unit, n_samples=400, lim=None):
    samples = np.random.choice(
        len(spread), n_samples, replace=False
    )  # to avoid over plotting
    lim = lim or [spread.min(), spread.max()]
    plt.plot(lim, lim, "k--")
    plt.plot(spread.iloc[samples], skill.iloc[samples], ".")
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel(r"Ensemble spread ($\frac{R+1}{R} s_t^2$)")
    plt.ylabel("Error (MSE of ensemble mean)")


def spread_skill(field, unit, lim, n_samples):
    plt.figure(figsize=figsize)
    plot_spread_skill(
        spread[field], skill[field], unit=unit, n_samples=n_samples, lim=lim
    )
    plt.title(field + f" [{unit}]")
    plt.tight_layout()
    savefig(f"figures/spread-skill/{field}")


# %%
spread_skill("tp", "$mm^2/h^2$", [0, 10], 1000)

# %%
spread_skill("10u", "$m^2/s^2$", [0, 5], 200)

# %%
spread_skill("10v", "$m^2/s^2$", [0, 5], 200)

# %% [markdown]
# # Rank histograms

# %%
arrs = np.load(os.path.join(output_path, "counts.npz"))

for v in arrs:
    plt.figure(figsize=figsize)
    plt.bar(np.arange(17), arrs[v] / arrs[v].sum())
    plt.title(v)
    plt.xlabel("Rank of observation")
    plt.ylabel("Probability")
    plt.tight_layout()
    savefig(f"figures/rank-histogram/{v}")

# %% [markdown]
# # Table

# %%
df

# %%
df = get_scores()
averages = df.groupby(["variable", "metric", "source"]).mean().value
print(averages.unstack(["source"]))

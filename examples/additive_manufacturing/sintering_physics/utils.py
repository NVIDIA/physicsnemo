# © Copyright 2023 HP Development Company, L.P.
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


import json
import math
import os

import numpy as np
import torch

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the Tensorflow library. Install the "
        + "package at: https://www.tensorflow.org/install"
    )

NUM_PARTICLE_TYPES = 3
KINEMATIC_PARTICLE_ID = 0  # refers to anchor point
METAL_PARTICLE_ID = 2  # refers to normal particles
ANCHOR_PLANE_PARTICLE_ID = 1  # refers to anchor plane


class Stats:
    """
    Represents statistical attributes with methods for device transfer.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def to(self, device):
        """Transfers the mean and standard deviation to a specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def cast(v):
    return np.array(v, dtype=np.float64)


def _read_metadata(data_path):
    """reads metadata"""
    with open(os.path.join(data_path, "metadata.json"), "rt") as fp:
        return json.load(fp)


def _combine_std(std_x, std_y):
    """combine standard deviation with l2 norm"""
    return np.sqrt(std_x**2 + std_y**2)


def tf2torch(t):
    """
    Converts a TensorFlow tensor to a PyTorch tensor.
    """
    t = torch.from_numpy(t.numpy())
    return t


def torch2tf(t):
    """
    Converts a PyTorch tensor to a TensorFlow tensor.
    """
    t = tf.convert_to_tensor(t.cpu().numpy())
    return t


def get_kinematic_mask(particle_types):
    """Returns a boolean mask, set to true for kinematic (obstacle) particles."""
    # return tf.equal(particle_types, KINEMATIC_PARTICLE_ID)
    # return size: num_particles_in_batch

    return particle_types == torch.ones(particle_types.shape) * KINEMATIC_PARTICLE_ID


def get_metal_mask(particle_types):
    """Returns a boolean mask, set to true for metal particles."""
    # get free particles
    return particle_types == torch.ones(particle_types.shape) * METAL_PARTICLE_ID


def get_anchor_z_mask(particle_types):
    """
    Generates a mask identifying anchor plane particles in a tensor of particle types.
    """
    # get anchor plane particles
    return particle_types == torch.ones(particle_types.shape) * ANCHOR_PLANE_PARTICLE_ID


def cos_theta(p1, p2):
    """compute cosine of two non-zero vectors"""
    return (torch.dot(p1, p2)) / (
        (torch.sqrt(torch.dot(p1, p1))) * (math.sqrt(torch.dot(p2, p2)))
    )


def weighted_square_error(y_pre, y, device):
    """
    Calculates a weighted square error for predictions, emphasizing larger errors
    by sorting and applying diminishing weights.
    """
    k = y_pre - y
    print("weighted_square_error k shape: ", k.shape)

    k = k.view(-1)
    k = torch.square(k)
    sorted, indices = torch.sort(k, descending=True)
    print("weight: ", sorted.size())
    n = sorted.size()[0]

    weights = []
    dw = 1.0 / n
    for i in range(n):
        weights.append(dw)
        dw = dw * 0.99
    weights = torch.FloatTensor(weights).to(device)

    out = weights * sorted
    print("weighted_square_error out shape: ", out.shape)

    out = torch.mean(out)
    # out = torch.sum(out)
    print("mean out: ", out, out.shape)
    return out


def weighted_loss(loss_, device):
    """
    Computes a loss value where individual components are weighted, with higher weights
    assigned to larger loss components.
    """
    loss_ = loss_.view(-1)
    sorted, indices = torch.sort(loss_, descending=True)
    n = sorted.size()[0]

    weights = []
    dw = 1.0 / n
    for i in range(n):
        weights.append(dw)
        dw = dw * 0.99
    weights = torch.FloatTensor(weights).to(device)

    out = weights * sorted

    out = torch.sum(out)
    print("out: ", out)
    return out

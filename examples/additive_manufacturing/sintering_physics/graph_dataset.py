# ignore_header_test
# ruff: noqa: E402

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


import functools
import os

import tree

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the Tensorflow library. Install the "
        + "package at: https://www.tensorflow.org/install"
    )

from reading_utils import parse_serialized_simulation_example, split_trajectory
from utils import _read_metadata, tf2torch

INPUT_SEQUENCE_LENGTH = 5  # calculate the last 5 velocities. [options: 5, 10]
PREDICT_LENGTH = 1  # [options: 5]
LOSS_DECAY_FACTOR = 0.6

NUM_PARTICLE_TYPES = 3
KINEMATIC_PARTICLE_ID = 0  # refers to anchor point
METAL_PARTICLE_ID = 2  # refers to normal particles
ANCHOR_PLANE_PARTICLE_ID = 1  # refers to anchor plane


def batch_concat(dataset, batch_size):
    """We implement batching as concatenating on the leading axis."""

    # We create a dataset of datasets of length batch_size.
    windowed_ds = dataset.window(batch_size)

    # The plan is then to reduce every nested dataset by concatenating. We can
    # do this using tf.data.Dataset.reduce. This requires an initial state, and
    # then incrementally reduces by running through the dataset

    # Get initial state. In this case this will be empty tensors of the
    # correct shape.
    initial_state = tree.map_structure(
        lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
            shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype
        ),
        dataset.element_spec,
    )

    # We run through the nest and concatenate each entry with the previous state.
    def reduce_window(initial_state, ds):
        """reduce every nested dataset by concatenating, done using tf.data.Dataset.reduce"""
        return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

    return windowed_ds.map(
        lambda *x: tree.map_structure(reduce_window, initial_state, x)
    )


def get_input_fn(data_path, batch_size, prefetch_buffer_size, mode, split):
    """Gets the learning simulation input function for tf.estimator.Estimator.

    Args:
      data_path: the path to the dataset directory.
      batch_size: the number of graphs in a batch.
      mode: either 'one_step_train', 'one_step' or 'rollout'
      split: either 'train', 'valid' or 'test.

    Returns:
      The input function for the learning simulation model.
    """

    def input_fn():
        """Gets the learning simulation input function for tf.estimator"""
        # Load the metadata of the dataset.
        metadata = _read_metadata(data_path)

        # Create a tf.data.Dataset from the TFRecord.
        # todo: try data exists
        ds = tf.data.TFRecordDataset([os.path.join(data_path, f"{split}.tfrecord")])
        ds = ds.map(
            functools.partial(parse_serialized_simulation_example, metadata=metadata)
        )

        if mode.startswith("one_step"):
            # Splits an entire trajectory into chunks of n steps. (n=INPUT_SEQUENCE_LENGTH)
            # Previous steps are used to compute the input velocities
            split_with_window = functools.partial(
                split_trajectory,
                window_length=INPUT_SEQUENCE_LENGTH,
                predict_length=PREDICT_LENGTH,
            )
            ds = ds.flat_map(split_with_window)
            # Splits a chunk into input steps and target steps
            ds = ds.map(prepare_inputs)
            # If in train mode, repeat dataset forever and shuffle.
            if mode == "one_step_train":
                ds.prefetch(buffer_size=prefetch_buffer_size)
                ds = ds.repeat()
                ds = ds.shuffle(512)

            # Custom batching on the leading axis.
            print("before apply batch_concat ds: ", ds)
            ds = batch_concat(ds, batch_size)
        elif mode == "rollout":
            if not batch_size == 1:
                raise ValueError("Rollout evaluation only available for batch size 1")

            ds = ds.map(prepare_rollout_inputs)
        else:
            raise ValueError(f"mode: {mode} not recognized")

        return ds

    return input_fn


def prepare_inputs(tensor_dict):
    """Prepares a single stack of inputs by calculating inputs and targets.

    Computes n_particles_per_example, which is a tensor that contains information
    about how to partition the axis - i.e. which nodes belong to which graph.

    Adds a batch axis to `n_particles_per_example` and `step_context` so they can
    later be batched using `batch_concat`. This batch will be the same as if the
    elements had been batched via stacking.

    Note that all other tensors have a variable size particle axis,
    and in this case they will simply be concatenated along that
    axis.



    Args:
      tensor_dict: A dict of tensors containing positions, and step context (
      if available).

    Returns:
      A tuple of input features and target positions.

    """
    predict_length = PREDICT_LENGTH

    pos = tensor_dict["position"]
    pos = tf.transpose(pos, perm=[1, 0, 2])

    # The target position is the final step of the stack of positions.
    target_position = pos[:, -predict_length:]

    # Remove the target from the input.
    tensor_dict["position"] = pos[:, :-predict_length]

    # Compute the number of particles per example.
    num_particles = tf.shape(pos)[0]
    # Add an extra dimension for stacking via concat.
    tensor_dict["n_particles_per_example"] = num_particles[tf.newaxis]

    num_edges = tf.shape(tensor_dict["senders"])[0]
    tensor_dict["n_edges_per_example"] = num_edges[tf.newaxis]

    if "step_context" in tensor_dict:
        # Take the input global context. We have a stack of global contexts,
        # and we take the penultimate since the final is the target.

        # Method: input the entire sequence of sintering profile
        tensor_dict["step_context"] = tf.reshape(tensor_dict["step_context"], [1, -1])

    # if mode== inference:
    #     if "step_context" in tensor_dict:
    #         tensor_dict["step_context"] = tensor_dict["step_context"][-predict_length - 1]
    #         # Add an extra dimension for stacking via concat.
    #         tensor_dict["step_context"] = tensor_dict["step_context"][tf.newaxis]

    print(
        "prepare inputs, tensor_dict['step_context'] shape: ",
        tensor_dict["step_context"].shape,
    )

    return tensor_dict, target_position


def prepare_rollout_inputs(context, features):
    """Prepares an inputs trajectory for rollout."""
    out_dict = {**context}

    pos = tf.transpose(features["position"], [1, 0, 2])
    # The target position is the final step of the stack of positions.
    target_position = pos[:, -1]

    #  can change whether to Remove the target from the input, with: out_dict['position'] = pos[:, :-1]
    out_dict["position"] = pos
    #  if mode == "inference
    #     out_dict["position"] = pos[:, :-1]

    # Compute the number of nodes
    out_dict["n_particles_per_example"] = [tf.shape(pos)[0]]
    out_dict["n_edges_per_example"] = [tf.shape(context["senders"])[0]]
    if "step_context" in features:
        out_dict["step_context"] = tf.dtypes.cast(features["step_context"], tf.float64)

    out_dict["is_trajectory"] = tf.constant([True], tf.bool)
    return out_dict, target_position


class GraphDataset:
    """
    A dataset class for graph-based models, handling data loading and iteration
    for training or evaluation in different modes.
    """

    # todo: update the size
    def __init__(
        self,
        size=1000,
        mode="one_step_train",
        split="train",
        data_path="None",
        batch_size=1,
        prefetch_buffer_size=100,
    ):
        self.mode = mode
        self.dataset = get_input_fn(
            data_path, batch_size, prefetch_buffer_size, mode=mode, split=split
        )()

        if mode == "rollout":
            # test / inference with test data size:
            self.size = len(list(self.dataset))
        else:
            # train
            self.size = size

        self.dataset = iter(self.dataset)
        self.pos = 0

    def __len__(self):
        return self.size

    def __next__(self):
        # print("get next ds: pos/ size: ", self.pos, self.size)
        if self.pos < self.size:
            features, targets = self.dataset.get_next()
            for key in features:
                if key != "key":
                    features[key] = tf2torch(features[key])

            targets = tf2torch(targets)
            self.pos += 1
            return features, targets
        else:
            raise StopIteration

    def __iter__(self):
        return self

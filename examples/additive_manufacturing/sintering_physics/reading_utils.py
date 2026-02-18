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

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the Tensorflow library. Install the "
        + "package at: https://www.tensorflow.org/install"
    )

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    "position": tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT["step_context"] = tf.io.VarLenFeature(
    tf.string
)

_FEATURE_DTYPES = {
    "position": {"in": np.float64, "out": tf.float64},
    "step_context": {"in": np.float64, "out": tf.float64},
}

_CONTEXT_FEATURES = {
    "key": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "particle_type": tf.io.VarLenFeature(tf.string),
    "senders": tf.io.VarLenFeature(tf.string),
    "receivers": tf.io.VarLenFeature(tf.string),
    # 'temperature': tf.io.VarLenFeature(tf.string)
}


def convert_to_tensor(x, encoded_dtype):
    """Convert inputs to tensor"""
    if len(x) == 1:
        out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
    else:
        out = []
        for el in x:
            out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
    out = tf.convert_to_tensor(np.array(out))
    return out


def parse_serialized_simulation_example(example_proto, metadata):
    """
    Parses a serialized simulation tf.SequenceExample.

    Args:
      example_proto: A string encoding of the tf.SequenceExample proto.
      metadata: A dict of metadata for the dataset.

    Returns:
      context: A dict, with features that do not vary over the trajectory.
      parsed_features: A dict of tf.Tensors representing the parsed examples
        across time, where axis zero is the time axis.

    """
    if "context_mean" in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION

    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description,
    )

    for feature_key, item in parsed_features.items():
        print("feature_key", feature_key)
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]["in"]
        )
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]["out"]
        )

    # There is an extra frame at the beginning so we can calculate pos change
    # for all frames used in the paper.
    position_shape = [metadata["sequence_length"] + 1, -1, metadata["dim"]]
    print(f"\n\nposition shape: {position_shape}")
    print(f"parsed_features['position'] shape: {parsed_features['position'].shape}")

    # Reshape positions to correct dim:
    parsed_features["position"] = tf.reshape(
        parsed_features["position"], position_shape
    )

    # Set correct shapes of the remaining tensors.
    sequence_length = metadata["sequence_length"] + 1
    if "context_mean" in metadata:
        context_feat_len = len(metadata["context_mean"])
        parsed_features["step_context"] = tf.reshape(
            parsed_features["step_context"], [sequence_length, context_feat_len]
        )

    # Decode particle type explicitly
    print("decode particle_type")
    context["particle_type"] = tf.py_function(
        functools.partial(convert_fn, encoded_dtype=np.int64),
        inp=[context["particle_type"].values],
        Tout=[tf.int64],
    )
    context["particle_type"] = tf.reshape(context["particle_type"], [-1])

    context["senders"] = tf.py_function(
        functools.partial(convert_fn, encoded_dtype=np.int64),
        inp=[context["senders"].values],
        Tout=[tf.int64],
    )
    context["senders"] = tf.reshape(context["senders"], [-1])

    context["receivers"] = tf.py_function(
        functools.partial(convert_fn, encoded_dtype=np.int64),
        inp=[context["receivers"].values],
        Tout=[tf.int64],
    )
    context["receivers"] = tf.reshape(context["receivers"], [-1])

    return context, parsed_features


def split_trajectory(context, features, window_length=7, predict_length=10):
    """Splits trajectory into sliding windows."""
    # Our strategy is to make sure all the leading dimensions are the same size,
    # then we can use from_tensor_slices.

    trajectory_length = features["position"].get_shape().as_list()[0]

    # We then stack window_length position changes so the final
    # trajectory length will be - window_length +1 (the 1 to make sure we get
    # the last split).
    input_trajectory_length = trajectory_length - window_length - predict_length + 1

    model_input_features = {}
    # Prepare the context features per step.
    # Repeat the particle types for each window step
    model_input_features["particle_type"] = tf.tile(
        tf.expand_dims(context["particle_type"], axis=0), [input_trajectory_length, 1]
    )

    model_input_features["senders"] = tf.tile(
        tf.expand_dims(context["senders"], axis=0), [input_trajectory_length, 1]
    )

    model_input_features["receivers"] = tf.tile(
        tf.expand_dims(context["receivers"], axis=0), [input_trajectory_length, 1]
    )

    # todo: change the hard-coded trajectory length to be the entire global context (/ sintering profile) sequence length
    # sequence length here is the default sintering 2-stage total length
    # trajectory_length = 14 + 24

    # Process the parsed_features
    if "step_context" in features:
        global_stack = []
        for idx in range(input_trajectory_length):
            # append all the previous temperature history, use an additional module to concat to final vector as global features
            read_step_context = features["step_context"][: idx + window_length]
            zero_pad = tf.zeros(
                [trajectory_length - read_step_context.shape[0] - 1, 1],
                dtype=features["step_context"].dtype,
            )

            read_step_context = tf.concat([read_step_context, zero_pad], 0)
            global_stack.append(read_step_context)
        model_input_features["step_context"] = tf.stack(global_stack)

    pos_stack = [
        features["position"][idx : idx + window_length + predict_length]
        for idx in range(input_trajectory_length)
    ]
    model_input_features["position"] = tf.stack(pos_stack)

    return tf.data.Dataset.from_tensor_slices(model_input_features)

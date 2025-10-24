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

# from physicsnemo.models.mlp import FullyConnected
# import torch
# from physicsnemo.module.derivatives import DerivWrapper

# net = FullyConnected(
#     in_features=2,
#     out_features=2,
# )
# p = net(torch.ones(1000, 2))
# print(p)

# net = DerivWrapper(
#     net,
#     input_keys=["x", "y"],
#     output_keys=["u", "v"],
#     deriv_keys=["u__x", "v__y", "u__x__y"],
# )

# input_dict = {"x": torch.ones(1000, 1), "y": torch.ones(1000, 1)}
# p = net(input_dict)
# print(p["u"][0])

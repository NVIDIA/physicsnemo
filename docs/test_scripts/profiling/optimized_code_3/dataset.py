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

import torch
import numpy as np
from torch.utils.data import Dataset


class RandomNoiseDataset(Dataset):
    """
    Random normal distribution dataset.
    
    Mean AND STD of the distribution is set to the index
    of the sample requested.
    
    Length is hardcoded to 64.
    
    (Don't use this anywhere that isn't an example of how to write non-performant python code!)
    
    """

    def __init__(self, image_shape, ):
        """
        Arguments:
            image_shape (string): Shape of a single example to generate
        """
        self.shape = list(image_shape)


    def __len__(self):
        return 256

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Generate the raw data:
        raw = self.gen_single_image(idx)

        sample = {
            'image' : raw
        }
        
        return sample

    def gen_single_image(self, idx):
        
        return torch.normal(idx, idx, self.shape, device="cuda", dtype=torch.float16 )

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

import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from PIL import Image


# Create a function to convert a figure to a NumPy array
def fig_to_numpy(fig: mpl.figure.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)
    im = np.array(im)
    buf.close()

    # Convert to valid image
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    # if the image has 4 channels, remove the alpha channel
    if im.shape[-1] == 4:
        im = im[..., :3]
    # Convert to uint8 image
    if im.dtype != np.uint8:
        im = (im * 255).astype(np.uint8)
    return im


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)[:, 0:3]

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

from sympy import Symbol, Abs, sign
import numpy as np
from physicsnemo.sym.geometry.geometry import Geometry, csg_curve_naming
from physicsnemo.sym.geometry.curve import SympyCurve
from physicsnemo.sym.geometry.parameterization import (
    Parameterization,
    Parameter,
    Bounds,
)
from physicsnemo.sym.geometry.helper import _sympy_sdf_to_sdf


class Point2D(Geometry):
    """
    2D Point along x and y axis

    Parameters
    ----------
    point : Tuple of int or float
        x and y coordinates of the point
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point, parameterization=Parameterization()):
        # make sympy symbols to use
        x = Symbol("x")
        y = Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({Symbol(csg_curve_naming(0)): (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        pt_1 = SympyCurve(
            functions={"x": point[0], "y": point[1], "normal_x": 1.0, "normal_y": 0},
            area=1.0,
            parameterization=curve_parameterization,
        )
        curves = [pt_1]

        # calculate SDF
        sdf = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 * sign(x - point[0])

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point[0], point[0]),
                Parameter("y"): (point[1], point[1]),
            },
            parameterization=parameterization,
        )

        # initialize
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=1,
            bounds=bounds,
            parameterization=parameterization,
        )

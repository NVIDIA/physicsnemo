# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Protocol
import earth2grid


class Domain(Protocol):
    def numel(self) -> int:
        size = 1
        for n in self.shape():
            size *= n
        return size

    def ndim(self) -> int:
        return len(self.shape())

    def shape(self) -> tuple[int]:
        pass

    @property
    def img_resolution(self) -> int:
        return max(self.shape())

    # TODO add boundary/padding function


class HealPixDomain(Domain):
    def __init__(self, hpx: earth2grid.healpix.Grid):
        self._grid = hpx

    def shape(self) -> tuple[int]:
        return self._grid.shape

    @property
    def img_resolution(self):
        return 2**self._grid.level


class PatchedHealpixDomain(Domain):
    def __init__(self, hpx: earth2grid.healpix.Grid, patch_size: int = 128):
        self._grid = hpx
        self.patch_size = patch_size

    def shape(self) -> tuple[int]:
        return self._grid.shape

    @property
    def img_resolution(self):
        return self.patch_size


class Plane(Domain):
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    def shape(self):
        return (self.ny, self.nx)


class Ring(Domain):
    def __init__(self, n):
        self.n = n

    def shape(self):
        return (self.n,)

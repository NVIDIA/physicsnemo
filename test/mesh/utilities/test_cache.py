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

"""Tests for cache utility functions.

Tests validate the get_cached and set_cached functions used for storing
and retrieving cached computed values in TensorDict structures.
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh.utilities._cache import get_cached, set_cached


class TestGetCached:
    """Tests for get_cached function."""

    def test_get_cached_returns_none_when_not_set(self):
        """Test that get_cached returns None when cache is empty."""
        data = TensorDict({}, batch_size=[10])

        result = get_cached(data, "areas")

        assert result is None

    def test_get_cached_returns_none_for_missing_key(self):
        """Test that get_cached returns None for missing key in existing cache."""
        data = TensorDict({}, batch_size=[10])
        data["_cache"] = TensorDict({"centroids": torch.randn(10, 3)}, batch_size=[10])

        result = get_cached(data, "areas")

        assert result is None

    def test_get_cached_returns_value_when_set(self):
        """Test that get_cached returns the cached value when present."""
        data = TensorDict({}, batch_size=[10])
        cached_value = torch.randn(10, 3)
        data["_cache"] = TensorDict({"centroids": cached_value}, batch_size=[10])

        result = get_cached(data, "centroids")

        assert result is not None
        assert torch.equal(result, cached_value)

    def test_get_cached_empty_tensordict(self):
        """Test get_cached on completely empty TensorDict."""
        data = TensorDict({}, batch_size=[])

        result = get_cached(data, "any_key")

        assert result is None


class TestSetCached:
    """Tests for set_cached function."""

    def test_set_cached_creates_cache_if_missing(self):
        """Test that set_cached creates _cache TensorDict if not present."""
        data = TensorDict({}, batch_size=[10])
        value = torch.randn(10, 3)

        set_cached(data, "centroids", value)

        assert "_cache" in data
        assert "centroids" in data["_cache"].keys()

    def test_set_cached_stores_value(self):
        """Test that set_cached stores the value correctly."""
        data = TensorDict({}, batch_size=[10])
        value = torch.randn(10, 3)

        set_cached(data, "areas", value)

        stored = data[("_cache", "areas")]
        assert torch.equal(stored, value)

    def test_set_cached_overwrites_existing(self):
        """Test that set_cached overwrites existing cached value."""
        data = TensorDict({}, batch_size=[10])
        old_value = torch.randn(10, 3)
        new_value = torch.randn(10, 3)

        set_cached(data, "centroids", old_value)
        set_cached(data, "centroids", new_value)

        stored = data[("_cache", "centroids")]
        assert torch.equal(stored, new_value)
        assert not torch.equal(stored, old_value)

    def test_set_cached_multiple_keys(self):
        """Test that set_cached can store multiple keys."""
        data = TensorDict({}, batch_size=[10])
        centroids = torch.randn(10, 3)
        areas = torch.randn(10)
        normals = torch.randn(10, 3)

        set_cached(data, "centroids", centroids)
        set_cached(data, "areas", areas)
        set_cached(data, "normals", normals)

        assert torch.equal(data[("_cache", "centroids")], centroids)
        assert torch.equal(data[("_cache", "areas")], areas)
        assert torch.equal(data[("_cache", "normals")], normals)


class TestCacheRoundTrip:
    """Tests for set_cached followed by get_cached."""

    def test_roundtrip_scalar(self):
        """Test round-trip with scalar tensor."""
        data = TensorDict({}, batch_size=[])
        value = torch.tensor(42.0)

        set_cached(data, "time", value)
        retrieved = get_cached(data, "time")

        assert retrieved is not None
        assert torch.equal(retrieved, value)

    def test_roundtrip_1d(self):
        """Test round-trip with 1D tensor."""
        data = TensorDict({}, batch_size=[10])
        value = torch.randn(10)

        set_cached(data, "areas", value)
        retrieved = get_cached(data, "areas")

        assert retrieved is not None
        assert torch.equal(retrieved, value)

    def test_roundtrip_2d(self):
        """Test round-trip with 2D tensor."""
        data = TensorDict({}, batch_size=[10])
        value = torch.randn(10, 3)

        set_cached(data, "centroids", value)
        retrieved = get_cached(data, "centroids")

        assert retrieved is not None
        assert torch.equal(retrieved, value)

    def test_roundtrip_3d(self):
        """Test round-trip with 3D tensor."""
        data = TensorDict({}, batch_size=[10])
        value = torch.randn(10, 3, 3)

        set_cached(data, "stress", value)
        retrieved = get_cached(data, "stress")

        assert retrieved is not None
        assert torch.equal(retrieved, value)


class TestCacheWithExistingData:
    """Tests for cache operations with pre-existing data."""

    def test_cache_does_not_affect_existing_data(self):
        """Test that caching doesn't affect existing non-cache data."""
        data = TensorDict({"temperature": torch.randn(10)}, batch_size=[10])
        original_temp = data["temperature"].clone()

        set_cached(data, "areas", torch.randn(10))

        assert torch.equal(data["temperature"], original_temp)

    def test_get_cached_ignores_non_cache_keys(self):
        """Test that get_cached only looks in _cache namespace."""
        data = TensorDict({"areas": torch.randn(10)}, batch_size=[10])

        # Even though "areas" exists at top level, get_cached looks in _cache
        result = get_cached(data, "areas")

        assert result is None

    def test_cache_coexists_with_data(self):
        """Test that cache and regular data coexist."""
        data = TensorDict(
            {
                "temperature": torch.randn(10),
                "pressure": torch.randn(10),
            },
            batch_size=[10],
        )

        set_cached(data, "centroids", torch.randn(10, 3))

        assert "temperature" in data.keys()
        assert "pressure" in data.keys()
        assert "_cache" in data.keys()
        assert get_cached(data, "centroids") is not None


class TestCacheDevices:
    """Tests for device handling in cache operations."""

    def test_cache_cpu(self):
        """Test caching on CPU TensorDict."""
        data = TensorDict({}, batch_size=[10], device="cpu")
        value = torch.randn(10, 3, device="cpu")

        set_cached(data, "centroids", value)
        retrieved = get_cached(data, "centroids")

        assert retrieved is not None
        assert retrieved.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cache_cuda(self):
        """Test caching on CUDA TensorDict."""
        data = TensorDict({}, batch_size=[10], device="cuda")
        value = torch.randn(10, 3, device="cuda")

        set_cached(data, "centroids", value)
        retrieved = get_cached(data, "centroids")

        assert retrieved is not None
        assert retrieved.device.type == "cuda"


class TestCacheDtypes:
    """Tests for dtype handling in cache operations."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.int64, torch.int32]
    )
    def test_cache_various_dtypes(self, dtype):
        """Test caching with various dtypes."""
        data = TensorDict({}, batch_size=[10])
        if dtype in [torch.float32, torch.float64]:
            value = torch.randn(10, dtype=dtype)
        else:
            value = torch.randint(0, 100, (10,), dtype=dtype)

        set_cached(data, "values", value)
        retrieved = get_cached(data, "values")

        assert retrieved is not None
        assert retrieved.dtype == dtype


class TestCacheIntegrationWithMesh:
    """Tests for cache usage patterns similar to Mesh class."""

    def test_cell_data_cache_pattern(self):
        """Test typical cell_data cache pattern used in Mesh."""
        # Simulate cell_data TensorDict
        cell_data = TensorDict({}, batch_size=[100])

        # First access - cache miss
        cached_areas = get_cached(cell_data, "areas")
        assert cached_areas is None

        # Compute and cache
        computed_areas = torch.randn(100)
        set_cached(cell_data, "areas", computed_areas)

        # Second access - cache hit
        cached_areas = get_cached(cell_data, "areas")
        assert cached_areas is not None
        assert torch.equal(cached_areas, computed_areas)

    def test_point_data_cache_pattern(self):
        """Test typical point_data cache pattern used in Mesh."""
        # Simulate point_data TensorDict
        point_data = TensorDict({}, batch_size=[500])

        # Cache point normals
        normals = torch.randn(500, 3)
        set_cached(point_data, "normals", normals)

        # Retrieve
        cached_normals = get_cached(point_data, "normals")
        assert cached_normals is not None
        assert cached_normals.shape == (500, 3)

    def test_multiple_caches_pattern(self):
        """Test multiple cached properties pattern."""
        cell_data = TensorDict({}, batch_size=[100])

        # Cache multiple properties
        set_cached(cell_data, "centroids", torch.randn(100, 3))
        set_cached(cell_data, "areas", torch.randn(100))
        set_cached(cell_data, "normals", torch.randn(100, 3))

        # All should be retrievable
        assert get_cached(cell_data, "centroids") is not None
        assert get_cached(cell_data, "areas") is not None
        assert get_cached(cell_data, "normals") is not None

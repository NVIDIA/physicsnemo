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

import multiprocessing as mp
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
from data_startup import initialize_datasets  # noqa: E402
from surface_geodesic_cache import SurfaceGeodesicCache  # noqa: E402
from test_surface_one_ring import strip  # noqa: E402

OPTIONS = dict(distance_scale=2**0.5, gap_scale=1.0, gap_min=5.0, max_pairs=100000)


def inputs():
    """Return a small mesh and thickness values for exclusion-cache tests."""
    p, f = strip()
    return p, f, torch.ones(len(p)), dict(OPTIONS)


def test_cache_cold_warm_memory_exact_and_rng_independent(tmp_path, monkeypatch):
    """Verify cache cold warm memory exact and RNG independent."""
    import surface_geodesic_cache as module

    args = inputs()
    original = [x.clone() for x in args[:3]]
    state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    reference = SurfaceGeodesicCache().get(*args)
    cold = SurfaceGeodesicCache(tmp_path)
    actual = cold.get(*args)
    assert cold.get(*args) is actual
    assert cold.counts == dict(computed=1, disk_hits=0, memory_hits=1)
    monkeypatch.setattr(
        module,
        "reference_geodesic_exclusions",
        lambda *a, **k: pytest.fail("Recomputed"),
    )
    # Preserve the original source fingerprint while testing that warm loads do
    # not invoke the expensive functional.
    warm = SurfaceGeodesicCache(tmp_path)
    warm.algorithm = cold.algorithm
    torch.testing.assert_close(warm.get(*args), reference, rtol=0, atol=0)
    assert warm.counts == dict(computed=0, disk_hits=1, memory_hits=0)
    for a, b in zip(args[:3], original):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(state[0], torch.get_rng_state())
    assert state[1] == random.getstate()
    after = np.random.get_state()
    assert state[2][0] == after[0] and np.array_equal(state[2][1], after[1])
    assert state[2][2:] == after[2:]


@pytest.mark.parametrize(
    "field", ["positions", "faces", "thickness", *OPTIONS, "algorithm"]
)
def test_cache_invalidates_every_physical_input_option_and_algorithm(tmp_path, field):
    """Verify cache invalidates every physical input option and algorithm."""
    cache = SurfaceGeodesicCache(tmp_path)
    p, f, t, options = inputs()
    cache.get(p, f, t, options)
    if field == "positions":
        p = p * 2
    elif field == "faces":
        f = f.flip(0)
    elif field == "thickness":
        t = t * 2
    elif field == "algorithm":
        cache.algorithm = "different-source-fingerprint"
    else:
        options[field] *= 2
    cache.get(p, f, t, options)
    assert cache.counts["computed"] == 2
    assert len(list(tmp_path.glob("*.pt"))) == 2


@pytest.mark.parametrize(
    "damage", ["checksum", "metadata", "order", "bounds", "dtype", "partial"]
)
def test_corrupt_cache_fails_closed(tmp_path, damage):
    """Verify corrupt cache fails closed."""
    import surface_geodesic_cache as module

    args = inputs()
    SurfaceGeodesicCache(tmp_path).get(*args)
    path = next(tmp_path.glob("*.pt"))
    entry = torch.load(path, weights_only=True)
    if damage == "checksum":
        entry["sha256"] = "wrong"
    elif damage == "metadata":
        entry["metadata"]["options"]["gap_min"] += 1
    elif damage == "order":
        entry["pairs"] = entry["pairs"].flip(1)
    elif damage == "bounds":
        entry["pairs"][0, -1] = len(args[0])
    elif damage == "dtype":
        entry["pairs"] = entry["pairs"].float()
    if damage in ("order", "bounds", "dtype"):
        entry["sha256"] = module._digest(entry["pairs"])
    if damage == "partial":
        path.write_bytes(b"interrupted torch save")
    else:
        torch.save(entry, path)
    with pytest.raises(RuntimeError, match="Invalid geodesic cache"):
        SurfaceGeodesicCache(tmp_path).get(*args)


def test_failed_publish_leaves_no_entry_or_temporary_file(tmp_path, monkeypatch):
    """Verify failed publish leaves no entry or temporary file."""
    import surface_geodesic_cache as module

    def fail(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        SurfaceGeodesicCache(tmp_path).get(*inputs())
    assert all(path.suffix == ".lock" for path in tmp_path.iterdir())


def test_pair_budget_failure_not_cached(tmp_path):
    """Verify pair budget failure not cached."""
    p, f, t, options = inputs()
    options["max_pairs"] = 1
    with pytest.raises(RuntimeError, match="budget"):
        SurfaceGeodesicCache(tmp_path).get(p, f, t, options)
    assert not list(tmp_path.glob("*.pt"))


def test_lock_timeout_and_recovery(tmp_path):
    """Verify lock timeout and recovery."""
    import fcntl

    args = inputs()
    SurfaceGeodesicCache(tmp_path).get(*args)
    path = next(tmp_path.glob("*.lock"))
    with path.open("a+b") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TimeoutError, match="geodesic cache"):
            SurfaceGeodesicCache(tmp_path, timeout_seconds=0.01).get(*args)
    recovered = SurfaceGeodesicCache(tmp_path)
    recovered.get(*args)
    assert recovered.counts["disk_hits"] == 1


def test_cache_fingerprints_the_functional_not_the_no_grad_wrapper():
    """Verify cache fingerprints the functional not the no grad wrapper."""
    import hashlib

    source = (
        CRASH_DIR.parents[2]
        / "physicsnemo/nn/functional/neighbors/reference_geodesic.py"
    )
    assert (
        SurfaceGeodesicCache().algorithm
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )


def _cache_worker(directory, queue):
    cache = SurfaceGeodesicCache(directory)
    value = cache.get(*inputs())
    queue.put((cache.counts, value.tolist()))


def _startup_worker(rank, rendezvous, mode, queue):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=2),
    )
    try:

        def build(stats_mode):
            assert stats_mode == ("compute" if rank == 0 else "load")
            if mode == "slow":
                # Both phases exceed the DEFAULT group's collective deadline.
                # No work may be outstanding there while either rank prepares.
                time.sleep(3)
            if mode == f"failure-{rank}":
                raise ValueError(f"fixture failure on rank {rank}")
            return ("train", "validation")

        try:
            result = initialize_datasets(build, timeout_seconds=30)
            assert result == ("train", "validation")
            tensor = torch.tensor(rank + 1.0)
            dist.all_reduce(tensor)
            assert tensor.item() == 3.0
            queue.put((rank, "ready"))
        except RuntimeError as error:
            queue.put((rank, str(error)))
    finally:
        dist.destroy_process_group()


def _run_workers(target, arguments):
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=target, args=(*args, queue)) for args in arguments
    ]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=55) for _ in processes]
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
        return results
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        queue.close()


def test_cache_concurrent_publish(tmp_path):
    """Verify cache concurrent publish."""
    result = _run_workers(_cache_worker, [(str(tmp_path),)] * 2)
    assert sum(row[0]["computed"] for row in result) == 1
    assert sum(row[0]["disk_hits"] for row in result) == 1
    assert result[0][1] == result[1][1]


@pytest.mark.parametrize("mode", ["slow", "failure-0", "failure-1"])
def test_distributed_cpu_startup_and_error_propagation(tmp_path, mode):
    """Verify distributed cpu startup and error propagation."""
    result = _run_workers(
        _startup_worker,
        [(rank, str(tmp_path / "rendezvous"), mode) for rank in range(2)],
    )
    assert sorted(row[0] for row in result) == [0, 1]
    for _, message in result:
        if mode == "slow":
            assert message == "ready"
        else:
            assert f"fixture failure on rank {mode[-1]}" in message


def test_serial_startup_and_invalid_timeouts():
    """Verify serial startup and invalid timeouts."""
    assert initialize_datasets(lambda mode: mode) == "compute"
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            initialize_datasets(lambda mode: mode, invalid)
        with pytest.raises(ValueError):
            SurfaceGeodesicCache(timeout_seconds=invalid)


def test_validation_config_copy_preserves_interpolations_and_train_settings():
    """Verify validation config copy preserves interpolations and train settings."""
    from copy import deepcopy

    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    with initialize_config_dir(config_dir=str(CRASH_DIR / "conf"), version_base="1.3"):
        config = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=[
                "training.raw_data_dir=/train",
                "training.raw_data_dir_validation=/val",
                "training.num_training_samples=127",
                "datapipe.stats_dir=/stats",
            ],
        )
    validation = deepcopy(config.datapipe)
    with open_dict(validation):
        validation.data_dir = config.training.raw_data_dir_validation
        validation.num_samples = 8
    assert config.datapipe.data_dir == "/train" and config.datapipe.num_samples == 127
    assert validation.data_dir == "/val" and validation.num_samples == 8
    assert validation.contact_geodesic_cache_dir == "/stats/surface_geodesic_cache"

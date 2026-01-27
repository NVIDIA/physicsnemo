# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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
# ruff: noqa: S101
"""Tests for RoundRobinLoader."""

import torch
import torch.utils.data
from datasets.round_robin import RoundRobinLoader


def test_round_robin_loader():
    """Test RoundRobinLoader round-robin interleaving logic."""
    # Create simple datasets
    dataset1 = list(range(0, 10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset2 = list(range(10, 15))  # [10, 11, 12, 13, 14]
    dataset3 = list(range(15, 20))  # [15, 16, 17, 18, 19]

    # Create dataloaders
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=2)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=2)
    loader3 = torch.utils.data.DataLoader(dataset3, batch_size=2)

    # Create RoundRobinLoader
    round_robin = RoundRobinLoader([loader1, loader2, loader3])

    # Test __len__
    assert len(round_robin) == len(loader1) + len(loader2) + len(loader3)
    assert len(round_robin) == 5 + 3 + 3  # 11 total batches

    # Collect all batches
    batches = list(round_robin)

    # Should have 11 batches total
    assert len(batches) == 11

    # First round: one batch from each loader
    assert torch.equal(batches[0], torch.tensor([0, 1]))  # from loader1
    assert torch.equal(batches[1], torch.tensor([10, 11]))  # from loader2
    assert torch.equal(batches[2], torch.tensor([15, 16]))  # from loader3

    # Second round
    assert torch.equal(batches[3], torch.tensor([2, 3]))  # from loader1
    assert torch.equal(batches[4], torch.tensor([12, 13]))  # from loader2
    assert torch.equal(batches[5], torch.tensor([17, 18]))  # from loader3

    # Third round - loader2 and loader3 exhausted, only loader1 continues
    assert torch.equal(batches[6], torch.tensor([4, 5]))  # from loader1
    assert torch.equal(batches[7], torch.tensor([14]))  # last batch from loader2
    assert torch.equal(batches[8], torch.tensor([19]))  # last batch from loader3

    # Remaining batches from loader1
    assert torch.equal(batches[9], torch.tensor([6, 7]))  # from loader1
    assert torch.equal(batches[10], torch.tensor([8, 9]))  # from loader1


def test_round_robin_loader_uneven_lengths():
    """Test RoundRobinLoader with very uneven loader lengths."""
    # Create datasets of very different sizes
    dataset1 = list(range(0, 20))  # 10 batches
    dataset2 = list(range(20, 22))  # 1 batch

    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=2)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=2)

    round_robin = RoundRobinLoader([loader1, loader2])
    batches = list(round_robin)

    # Total should be 11 batches
    assert len(batches) == 11

    # First two batches alternate
    assert torch.equal(batches[0], torch.tensor([0, 1]))
    assert torch.equal(batches[1], torch.tensor([20, 21]))

    # After loader2 is exhausted, only loader1 continues
    assert torch.equal(batches[2], torch.tensor([2, 3]))
    assert torch.equal(batches[3], torch.tensor([4, 5]))


def test_round_robin_loader_empty():
    """Test RoundRobinLoader with empty dataloaders list."""
    round_robin = RoundRobinLoader([])
    batches = list(round_robin)
    assert len(batches) == 0


def test_round_robin_loader_single():
    """Test RoundRobinLoader with a single dataloader."""
    dataset = list(range(0, 10))
    loader = torch.utils.data.DataLoader(dataset, batch_size=3)

    round_robin = RoundRobinLoader([loader])
    batches = list(round_robin)

    # Should have same batches as the original loader
    expected = list(loader)
    assert len(batches) == len(expected)
    for b1, b2 in zip(batches, expected):
        assert torch.equal(b1, b2)


def test_round_robin_loader_epoch_handling():
    """Test that RoundRobinLoader correctly handles epoch transitions with ChunkedDistributedSampler."""
    from datasets.samplers import ChunkedDistributedSampler

    # Create a simple dataset
    dataset = list(range(100))

    # Create samplers for two "workers"
    sampler1 = ChunkedDistributedSampler(
        dataset,
        chunk_size=10,
        num_replicas=2,
        rank=0,
        shuffle=True,
        shuffle_within_chunk=False,
        drop_last=False,
        seed=42,
    )
    sampler2 = ChunkedDistributedSampler(
        dataset,
        chunk_size=10,
        num_replicas=2,
        rank=1,
        shuffle=True,
        shuffle_within_chunk=False,
        drop_last=False,
        seed=42,
    )

    # Create dataloaders
    loader1 = torch.utils.data.DataLoader(dataset, batch_size=5, sampler=sampler1)
    loader2 = torch.utils.data.DataLoader(dataset, batch_size=5, sampler=sampler2)

    # Create RoundRobinLoader
    round_robin = RoundRobinLoader([loader1, loader2])

    # Epoch 1: Collect all batches
    epoch1_batches = list(round_robin)
    epoch1_count = len(epoch1_batches)

    # Verify we got data
    assert epoch1_count > 0, "Should have batches in epoch 1"

    # Check that samplers auto-incremented their epoch
    assert sampler1.epoch == 1, "Sampler 1 should have auto-incremented to epoch 1"
    assert sampler2.epoch == 1, "Sampler 2 should have auto-incremented to epoch 1"

    # Epoch 2: Iterate again - should work and get different shuffle order
    epoch2_batches = list(round_robin)
    epoch2_count = len(epoch2_batches)

    # Should have same number of batches
    assert epoch2_count == epoch1_count, "Should have same number of batches each epoch"

    # Check that samplers auto-incremented again
    assert sampler1.epoch == 2, "Sampler 1 should have auto-incremented to epoch 2"
    assert sampler2.epoch == 2, "Sampler 2 should have auto-incremented to epoch 2"

    # With shuffle=True, the order should be different across epochs
    # (though this is probabilistic, with seed=42 and shuffle=True it should differ)
    # We'll just verify we can iterate multiple times without error
    epoch3_batches = list(round_robin)
    assert len(epoch3_batches) == epoch1_count

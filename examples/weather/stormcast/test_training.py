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

import os
from pathlib import Path
from typing import Literal

from hydra import compose, initialize
from omegaconf import DictConfig
import pytest
import torch
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig,
)
from torch.distributed.tensor import DTensor

try:
    from optimi import StableAdamW

    IS_OPTIMI_AVAILABLE = True
except ImportError:
    IS_OPTIMI_AVAILABLE = False

from physicsnemo.distributed import DistributedManager

import train
from utils import trainer

DistributedManager.initialize()


# Retrieve and fixture configs
def _load_config(config_name: str) -> DictConfig:
    with initialize(version_base=None, config_path="config", job_name="test_training"):
        return compose(config_name=config_name)


@pytest.fixture
def cfg_regression():
    return _load_config(config_name="test_regression_unet.yaml")


@pytest.fixture
def cfg_diffusion():
    return _load_config(config_name="test_diffusion.yaml")


@pytest.fixture
def cfg_diffusion_unet():
    return _load_config(config_name="test_diffusion_unet.yaml")


def _setup_rundir(tmp_path, num_procs):
    # Set up rundir in the temporary directory
    _rundir = tmp_path / "rundir"
    _rundir.mkdir()
    rundir = str(_rundir)

    if num_procs > 1:
        # sync same rundir for all processes
        output_list = [None]
        torch.distributed.barrier()
        torch.distributed.scatter_object_list(output_list, [rundir] * num_procs, src=0)
        rundir = output_list[0]

    return rundir


@pytest.mark.parametrize("net_architecture", ["unet", "dit"])
# @pytest.mark.parametrize("use_regression", [True, False])
@pytest.mark.parametrize("use_regression", [False])
# @pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "domain_parallel_size, force_sharding", [(1, False), (1, True), (2, False)]
)
@pytest.mark.parametrize("fp_optimizations", ["fp32", "amp-bf16"])
# @pytest.mark.parametrize("torch_compile", [True, False])
@pytest.mark.parametrize("torch_compile", [False])
@pytest.mark.parametrize(
    "optimizer", ["adamw", "stableadamw"]
)  # deliberately skipping adam to reduce combinations
@pytest.mark.parametrize("scheduler", [None, "CosineAnnealingLR"])
@pytest.mark.parametrize("sigma_distribution", ["lognormal", "loguniform"])
def test_training(
    tmp_path: Path,
    cfg_regression: DictConfig,
    cfg_diffusion: DictConfig,
    cfg_diffusion_unet: DictConfig,
    *,
    net_architecture: Literal["unet", "dit"],
    use_regression: bool,
    batch_size: int,
    domain_parallel_size: int,
    force_sharding: bool,
    fp_optimizations: Literal["fp32", "amp-fp16", "amp-bf16"],
    torch_compile: bool,
    optimizer: Literal["adam", "adamw", "stableadamw"],
    scheduler: str | None,
    sigma_distribution: Literal["lognormal", "loguniform"],
):
    """Test that training runs with different combinations of parameters."""
    dist = DistributedManager()

    # Skip tests that cannot be run within the present environment
    max_world_size = batch_size * domain_parallel_size
    if dist.world_size > max_world_size:
        pytest.skip(
            f"Skipping: number of processes ({dist.world_size}) > batch_size * domain_parallel_size ({max_world_size})."
        )
    if domain_parallel_size > dist.world_size:
        pytest.skip(
            f"Skipping: not enough processes ({dist.world_size}) to use domain_parallel_size of {domain_parallel_size}."
        )
    sharding = (domain_parallel_size > 1) or force_sharding
    if sharding and torch_compile:
        pytest.skip(
            "Skipping: torch.compile is not supported with ShardTensor for now."
        )
    if (not IS_OPTIMI_AVAILABLE) and (optimizer == "stableadamw"):
        pytest.skip(
            "Skipping: StableAdamW optimizer is not available because optimi is not installed."
        )
    if sharding and (optimizer == "stableadamw"):
        pytest.skip(
            "Skipping: StableAdamW optimizer is not supported with ShardTensor for now."
        )

    # Set up rundir in the temporary directory
    rundir = _setup_rundir(tmp_path, dist.world_size)

    cfg_regression = cfg_regression.copy()
    cfg_diffusion = (
        cfg_diffusion if net_architecture == "dit" else cfg_diffusion_unet
    ).copy()

    # override params from config
    for cfg in [cfg_regression, cfg_diffusion]:
        cfg.model.architecture = net_architecture
        cfg.training.batch_size = batch_size
        cfg.training.domain_parallel_size = domain_parallel_size
        cfg.training.force_sharding = force_sharding
        cfg.training.perf.fp_optimizations = fp_optimizations
        cfg.training.perf.torch_compile = torch_compile
        cfg.training.optimizer.name = optimizer
        cfg.training.scheduler.name = scheduler
        cfg.training.rundir = rundir
    cfg_diffusion.training.loss.sigma_distribution = sigma_distribution

    if use_regression:
        train.main(cfg_regression)

        net_cls = "StormCastUNet" if net_architecture == "unet" else "DiTWrapper"
        ckpt_path = os.path.join(
            rundir, "checkpoints_regression", f"{net_cls}.0.10.mdlus"
        )
        assert os.path.isfile(ckpt_path), "Regression checkpoint not found"
    else:
        if "regression" in cfg_diffusion.model.diffusion_conditions:
            cfg_diffusion.model.diffusion_conditions.remove("regression")

    train.main(cfg_diffusion)

    if dist.world_size > 1:
        torch.distributed.barrier()

    net_cls = "EDMPrecond" if net_architecture == "unet" else "EDMPreconditioner"
    ckpt_path = os.path.join(rundir, "checkpoints_diffusion", f"{net_cls}.0.10.mdlus")
    assert os.path.isfile(ckpt_path), "Diffusion checkpoint not found"


@pytest.mark.parametrize("net_architecture", ["unet", "dit"])
# @pytest.mark.parametrize("use_regression", [True, False])
@pytest.mark.parametrize("use_regression", [False])
@pytest.mark.parametrize(
    "domain_parallel_size_0, batch_size_0, domain_parallel_size_1, batch_size_1",
    [(1, 2, 2, 1), (2, 1, 1, 2), (1, 2, 1, 2), (2, 1, 2, 1), (1, 1, 1, 1)],
)
@pytest.mark.parametrize("scheduler", [None, "CosineAnnealingLR"])
def test_checkpointing(
    tmp_path: Path,
    cfg_regression: DictConfig,
    cfg_diffusion: DictConfig,
    cfg_diffusion_unet: DictConfig,
    *,
    net_architecture: Literal["unet", "dit"],
    use_regression: bool,
    domain_parallel_size_0: int,
    batch_size_0: int,
    domain_parallel_size_1: int,
    batch_size_1: int,
    scheduler: str | None,
):
    """Test that checkpointing works and checkpoints are compatible with different domain parallel sizes."""
    dist = DistributedManager()

    num_procs = domain_parallel_size_0 * batch_size_0
    if num_procs != dist.world_size:
        pytest.skip(
            f"Skipping: this checkpointing test is only run with {num_procs} processes, current: {dist.world_size}."
        )

    rundir = _setup_rundir(tmp_path, num_procs)

    print(f"Rank={dist.rank} rundir={rundir}")

    cfg_regression = cfg_regression.copy()
    cfg_diffusion = (
        cfg_diffusion if net_architecture == "dit" else cfg_diffusion_unet
    ).copy()

    # override params from config
    for cfg in [cfg_regression, cfg_diffusion]:
        cfg.training.batch_size = batch_size_0
        cfg.training.domain_parallel_size = domain_parallel_size_0
        cfg.training.scheduler.name = scheduler
        cfg.training.rundir = rundir

    if use_regression:
        train.main(cfg_regression)
    if "regression" in cfg_diffusion.model.diffusion_conditions:
        cfg_diffusion.model.diffusion_conditions.remove("regression")

    # run for 10 steps first, will produce a checkpoint
    cfg_diffusion.training.total_train_steps = 10
    train.main(cfg_diffusion)

    # this will load the checkpoint from the previous training and continue to 20 steps
    cfg_diffusion.training.batch_size = batch_size_1
    cfg_diffusion.training.domain_parallel_size = domain_parallel_size_1
    cfg_diffusion.training.total_train_steps = 20
    train.main(cfg_diffusion)

    if num_procs > 1:
        torch.distributed.barrier()

    net_cls = "EDMPrecond" if net_architecture == "unet" else "EDMPreconditioner"
    ckpt_path = os.path.join(rundir, "checkpoints_diffusion", f"{net_cls}.0.20.mdlus")
    assert os.path.isfile(ckpt_path), (
        f"Diffusion checkpoint not found on rank {dist.rank}"
    )


def test_checkpoint_integrity(
    tmp_path: Path,
    cfg_diffusion: DictConfig,
    *,
    net_architecture: Literal["unet", "dit"] = "dit",
):
    """Test that model and optimizer states are intact and sharded correctly after checkpoint save/load."""

    dist = DistributedManager()
    if not dist.world_size == 4:
        pytest.skip(
            f"Skipping: test_checkpoint_integrity is only run with exactly 4 processes, current: {dist.world_size}."
        )

    cfg_diffusion.training.domain_parallel_size = 2
    cfg_diffusion.training.batch_size = 2
    cfg_diffusion.training.rundir = _setup_rundir(tmp_path, dist.world_size)

    # create trainer, train a bit and save checkpoint
    t0 = trainer.Trainer(cfg_diffusion.copy())
    for _ in range(5):
        t0.train_step()
    t0.total_steps = 5
    net0 = t0.net
    opt0 = t0.optimizer
    t0.save_checkpoint()

    torch.distributed.barrier()

    # create another trainer, this will load the previous checkpoint
    t1 = trainer.Trainer(cfg_diffusion.copy())
    net1 = t1.net
    opt1 = t1.optimizer

    # get model and optimizer state dicts in full and sharded forms
    options = StateDictOptions(full_state_dict=True)
    (params0, opt_params0) = get_state_dict(net0, opt0, options=options)
    (params1, opt_params1) = get_state_dict(net1, opt1, options=options)

    for key, param0 in params0.items():
        param1 = params1[key]
        assert (param0 == param1).all().cpu().item(), (
            f"Model parameter {key} before and after checkpointing is not equal"
        )

    for key, opt_param0 in opt_params0["state"].items():
        opt_param1 = opt_params0["state"][key]
        for opt_var in opt_param0:
            assert (opt_param0[opt_var] == opt_param1[opt_var]).all().cpu().item(), (
                f"Optimizer parameter {key} before and after checkpointing is not equal"
            )

    # get positional embedding tensors for model and optimizer
    posembed = params1["model.model.tokenizer.pos_embed"]
    opt_posembed = opt_params1["state"]["model.model.tokenizer.pos_embed"]
    posembed_size = posembed.shape[1]

    # check that current rank has the correct slice of the positional embedding
    local_posembed_slice = (
        slice(None),
        slice(0, posembed_size // 2)
        if dist.rank % 2 == 0
        else slice(posembed_size // 2, None),
        slice(None),
    )
    sharded_posembed = posembed[local_posembed_slice]
    opt_sharded_posembed = {
        k: opt_posembed[k][local_posembed_slice] for k in ["exp_avg", "exp_avg_sq"]
    }

    # check that rank 2 has the same pos embed as rank 0 (and likewise for 1 and 3)
    torch.distributed.barrier()
    for shard in [
        sharded_posembed,
        opt_sharded_posembed["exp_avg"],
        opt_sharded_posembed["exp_avg_sq"],
    ]:
        if isinstance(shard, DTensor):
            shard = shard.to_local()
        shard = torch.as_tensor(shard).cpu()

        shard_list = [None for _ in range(dist.world_size)] if dist.rank == 0 else None
        torch.distributed.gather_object(shard, shard_list, dst=0)
        if dist.rank == 0:
            shard_list = [x.clone() for x in shard_list]
            for i in range(dist.world_size):
                for j in range(i + 1, dist.world_size):
                    shards_equal = (shard_list[i] == shard_list[j]).all().cpu().item()
                    if j - i == 2:
                        assert shards_equal, (
                            f"Different positional embedding shards on ranks {i} and {j}"
                        )
                    else:
                        assert not shards_equal, (
                            f"Same positional embedding shards on ranks {i} and {j}"
                        )

        torch.distributed.barrier()

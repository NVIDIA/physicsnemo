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

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import wandb
import dnnlib
from torch_utils import distributed as dist
from utils.diffusions import training_loop
import glob

import warnings

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 1.12.

# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


@click.command()

# Main options.
@click.option(
    "--outdir", help="Where to save the results", metavar="DIR", type=str, required=True
)
@click.option(
    "--precond",
    help="Preconditioning & loss function",
    metavar="vp|ve|edm",
    type=click.Choice(["vp", "ve", "edm"]),
    default="edm",
    show_default=True,
)
@click.option(
    "--config_file", help="Path to config file", metavar="FILE", type=str, required=True
)
@click.option(
    "--config_name",
    help="Name of config to use",
    metavar="NAME",
    type=str,
    required=True,
)
@click.option(
    "--log_to_wandb",
    help="Log to wandb",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--run_id", help="run id", metavar="INT", type=int, default=None, show_default=True
)

# Hyperparameters.
@click.option(
    "--batch",
    help="Total batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=512,
    show_default=True,
)
@click.option(
    "--batch-gpu",
    help="Limit batch size per GPU",
    metavar="INT",
    type=click.IntRange(min=1),
)
@click.option(
    "--cbase", help="Channel multiplier  [default: varies]", metavar="INT", type=int
)
@click.option(
    "--cres",
    help="Channels per resolution  [default: varies]",
    metavar="LIST",
    type=parse_int_list,
)
@click.option(
    "--lr",
    help="Learning rate",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=10e-4,
    show_default=True,
)
@click.option(
    "--ema",
    help="EMA half-life",
    metavar="MIMG",
    type=click.FloatRange(min=0),
    default=0.5,
    show_default=True,
)
@click.option(
    "--dropout",
    help="Dropout probability",
    metavar="FLOAT",
    type=click.FloatRange(min=0, max=1),
    default=0.13,
    show_default=True,
)
@click.option(
    "--augment",
    help="Augment probability",
    metavar="FLOAT",
    type=click.FloatRange(min=0, max=1),
    default=0.12,
    show_default=True,
)
@click.option(
    "--xflip",
    help="Enable dataset x-flips",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)

# Performance-related.
@click.option(
    "--fp16",
    help="Enable mixed-precision training",
    metavar="BOOL",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--ls",
    help="Loss scaling",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=1,
    show_default=True,
)
@click.option(
    "--bench",
    help="Enable cuDNN benchmarking",
    metavar="BOOL",
    type=bool,
    default=True,
    show_default=True,
)
@click.option(
    "--cache",
    help="Cache dataset in CPU memory",
    metavar="BOOL",
    type=bool,
    default=True,
    show_default=True,
)
@click.option(
    "--workers",
    help="DataLoader worker processes",
    metavar="INT",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
)

# I/O-related.
@click.option(
    "--desc", help="String to include in result dir name", metavar="STR", type=str
)
@click.option(
    "--nosubdir", help="Do not create a subdirectory for results", is_flag=True
)
@click.option(
    "--tick",
    help="How often to print progress",
    metavar="KIMG",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
)
@click.option(
    "--snap",
    help="How often to save snapshots",
    metavar="TICKS",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
)
@click.option(
    "--dump",
    help="How often to dump state",
    metavar="TICKS",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
)
@click.option("--seed", help="Random seed  [default: random]", metavar="INT", type=int)
@click.option(
    "--transfer",
    help="Transfer learning from network pickle",
    metavar="PKL|URL",
    type=str,
)
@click.option(
    "--resume", help="Resume from previous training state", metavar="PT", type=str
)
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)
def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam", lr=opts.lr, betas=[0.9, 0.999], eps=1e-8
    )

    # Training options.
    c.ema_halflife_kimg = int(opts.ema * 1000)
    # c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(
        kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump
    )

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Description string.
    desc = f"hrrr-gpus{dist.get_world_size():d}"
    if opts.desc is not None:
        desc += f"-{opts.desc}"

    desc = opts.config_name + "-" + desc

    # Pick output directory.
    cur_run_id = opts.run_id if opts.run_id is not None else 0
    c.run_dir = os.path.join(opts.outdir, f"{cur_run_id}-{desc}")

    # if run_dir exists, then resume training
    if os.path.exists(c.run_dir):
        training_states = sorted(
            glob.glob(os.path.join(c.run_dir, "training-state-*.pt"))
        )
        if training_states:
            print("Resuming training from previous run_dir: " + c.run_dir)
            last_training_state = sorted(
                glob.glob(os.path.join(c.run_dir, "training-state-*.pt"))
            )[-1]
            last_network_snapshot = sorted(
                glob.glob(os.path.join(c.run_dir, "network-snapshot-*.pkl"))
            )[-1]
            last_kimg = int(
                re.fullmatch(
                    r"network-snapshot-(\d+).pkl",
                    os.path.basename(last_network_snapshot),
                ).group(1)
            )
            c.resume_pkl = last_network_snapshot
            c.resume_kimg = last_kimg
            c.resume_state_dump = last_training_state
            print(
                "Resuming training from previous network-snapshot-*.pkl file: "
                + last_network_snapshot
            )
            print(
                "Resuming training from previous training-state-*.pt file: "
                + last_training_state
            )

    # Transfer learning and resume. If a resume or transfer file is specified, it takes precedence over the existing run_dir.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException(
                "--transfer and --resume cannot be specified at the same time"
            )
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r"training-state-(\d+).pt", os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException(
                "--resume must point to training-state-*.pt from a previous training run"
            )
        c.resume_pkl = os.path.join(
            os.path.dirname(opts.resume), f"network-snapshot-{match.group(1)}.pkl"
        )
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Print options.
    if opts.dry_run:
        dist.print0("Dry run; exiting.")
        return

    # Create output directory.
    dist.print0(f"Creating output directory...{c.run_dir}")
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(
            file_name=os.path.join(c.run_dir, "log.txt"),
            file_mode="a",
            should_flush=True,
        )

        if opts.log_to_wandb:
            entity, project = "wandb_entity", "wandb_project"
            entity = entity
            wandb_project = project
            wandb_name = opts.config_name + "_" + desc
            wandb_group = opts.config_name
            os.makedirs(os.path.join(c.run_dir, "wandb"), exist_ok=True)
            wandb.init(
                dir=os.path.join(c.run_dir, "wandb"),
                config=c,
                name=wandb_name,
                group=wandb_group,
                project=wandb_project,
                entity=entity,
                resume=opts.resume,
                mode="online",
            )

    # config options
    c.config_file = opts.config_file
    c.config_name = opts.config_name
    c.log_to_wandb = opts.log_to_wandb

    # Train.
    training_loop.training_loop(**c)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

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

import importlib.util

import torch
import torch.nn as nn

from model import MODEL_REGISTRY

from utils import (
    parse_args,
    print_and_save_results,
    get_csv_filename,
    save_result_incremental,
    end_to_end_benchmark,
)

from physicsnemo.distributed import DistributedManager

# Data parallelism: plain DDP works directly with ShardTensor activations.
from torch.nn.parallel import DistributedDataParallel as DDP

# FSDP2 is only needed when the model itself holds distributed (DTensor)
# parameters, which DDP cannot manage.
from torch.distributed.fsdp import fully_shard

# Imports for Domain Parallelism
from physicsnemo.domain_parallel import scatter_tensor, sync_module_over_mesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import (
    Replicate,
    Shard,
)


def shard_spatial_params(model: nn.Module, domain_mesh, dim_selector) -> None:
    """Shard statically-shaped spatial parameters across the domain mesh.

    Only used on the FSDP2 path (``--fsdp``), where the parameters become
    distributed tensors anyway.  ``dim_selector`` (from the model's
    :class:`ModelSpec`) maps a parameter name to the dim it should be split
    along -- e.g. a positional embedding laid out ``(1, num_patches, C)`` is
    split (not replicated) across the domain ranks to match the
    sequence-sharded activations it is added to.  (Without ``--fsdp`` such
    params stay plain and replicated and ShardTensor auto-promotes them in
    the forward pass.)

    Note carefully: this uses plain ``DTensor`` (``distribute_tensor``), not
    ``ShardTensor``.  Parameters are statically shaped, so DTensor's even-chunk
    sharding is exactly right; ``ShardTensor`` exists for the *activations*,
    whose sharding may be uneven and data-dependent.  ShardTensor interoperates
    with DTensor arguments directly, so the two mix freely in the forward pass.

    ``distribute_tensor`` broadcasts from rank 0 of the mesh by default, so this
    also synchronizes the sharded params across the domain group.
    """
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            shard_dim = dim_selector(name)
            if shard_dim is None:
                continue
            module.register_parameter(
                name,
                nn.Parameter(
                    distribute_tensor(param.data, domain_mesh, [Shard(shard_dim)]),
                    requires_grad=param.requires_grad,
                ),
            )


def main():
    """Main benchmarking script."""
    # Configuration

    args = parse_args()

    image_sizes = list(
        range(args.image_size_start, args.image_size_stop + 1, args.image_size_step)
    )
    device = torch.device("cuda")

    # Generate image sizes based on start, stop, and step
    if args.dimension == 2:
        image_sizes = list(
            range(args.image_size_start, args.image_size_stop + 1, args.image_size_step)
        )
    elif args.dimension == 3:
        image_sizes = list(
            range(
                args.image_size_start,
                args.image_size_stop + 1,
                args.image_size_step,
            )
        )

    # Initialize distributed manager first
    DistributedManager.initialize()
    dm = DistributedManager()

    # Set device based on local rank
    device = dm.device
    torch.cuda.set_device(device)

    # Resolve and validate the parallelism layout against the actual world
    # size.  The mesh must tile the whole job: ddp_size * domain_size ==
    # world_size.  This also guarantees the FSDP2 mesh (the "ddp" axis) is
    # consistent with the domain axis, since FSDP2 below shards over exactly
    # that ddp mesh dimension.
    if dm.world_size % args.domain_size != 0:
        raise ValueError(
            f"World size {dm.world_size} is not divisible by domain size "
            f"{args.domain_size}"
        )
    if args.ddp_size == -1:
        args.ddp_size = dm.world_size // args.domain_size
    if args.ddp_size * args.domain_size != dm.world_size:
        raise ValueError(
            f"ddp_size ({args.ddp_size}) x domain_size ({args.domain_size}) = "
            f"{args.ddp_size * args.domain_size} must equal the world size "
            f"({dm.world_size}). Pass --ddp_size -1 to infer it."
        )
    if args.fsdp and dm.world_size == 1:
        raise ValueError(
            "--fsdp requires a distributed run (world size > 1): no device "
            "mesh exists in a single-process job."
        )

    # Resolve the model choice and check its optional dependencies up front.
    model_spec = MODEL_REGISTRY[args.model]
    for module_name in model_spec.requires:
        if importlib.util.find_spec(module_name) is None:
            raise ImportError(
                f"--model {args.model} requires the optional '{module_name}' "
                f"package. Install it, e.g.: pip install "
                f"nvidia-physicsnemo[cu12,{module_name}-cu12]"
            )

    # Build the mesh whenever the job is distributed at all, so both axes are
    # explicit: DDP is handed the "ddp" mesh group directly (never the default
    # world group), and FSDP2 shards over that same axis.
    ddp_mesh = None
    domain_mesh = None
    if dm.world_size > 1:
        mesh = dm.initialize_mesh(
            mesh_shape=(
                args.ddp_size,
                args.domain_size,
            ),
            mesh_dim_names=["ddp", "domain"],
        )
        ddp_mesh = mesh["ddp"]
        domain_mesh = mesh["domain"]

    num_classes = 1000
    precision_mode = (
        "FP16" if args.use_mixed_precision and torch.cuda.is_available() else "FP32"
    )

    if dm.rank == 0:
        print(f"Device: {device}")
        print(f"Model: {args.model}")
        print(f"Batch size: {args.batch_size}")
        print(f"Domain size: {args.domain_size}")
        print(f"DDP size: {args.ddp_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Precision: {precision_mode}")
        print(f"torch.compile: {args.compile}")
        print("-" * 80)

    results = []

    ddp_size = args.ddp_size
    domain_size = args.domain_size

    # Set up incremental CSV output
    csv_filename = get_csv_filename(args, precision_mode)

    for img_size in image_sizes:
        if dm.rank == 0:
            print(f"\nTesting image size: {img_size}x{img_size}")

        # Each image size traces to different shapes; drop stale compiled
        # graphs so we never hit the recompilation limit across the sweep.
        if args.compile:
            torch._dynamo.reset()

        if args.dimension == 2:
            full_img_size = (img_size, img_size)
        elif args.dimension == 3:
            full_img_size = (img_size, img_size, img_size)

        if args.batch_size % ddp_size != 0 or args.batch_size // ddp_size == 0:
            raise ValueError(
                f"Batch size {args.batch_size} is not divisible by DDP size {ddp_size}"
            )

        # Create synthetic data - scale the batch size down by DDP size.
        x = torch.randn(args.batch_size // ddp_size, 3, *full_img_size, device=device)
        target = torch.randint(
            0, num_classes, (args.batch_size // ddp_size,), device=device
        )

        # Domain Parallel NOTE: we're generating data once per GPU but only keeping the data once per domain.
        # In a real application, you'd do this properly - each GPU would read it's own shard of the data.

        if args.domain_size > 1:
            # When scattering the data, we need to know the global rank of the source
            # But by definition, we use the domain_rank == 0 as the source.  Convert:
            global_rank_of_source = torch.distributed.get_global_rank(
                domain_mesh.get_group(), 0
            )

            # Scatter the input data across the domain:
            x = scatter_tensor(
                x,
                global_rank_of_source,
                domain_mesh,
                placements=(
                    Shard(2),
                ),  # Shard along the 2nd dimension (B C **H** W) which is the Height
                global_shape=x.shape,  # This will be inferred if not provided!
                dtype=x.dtype,  # This will be inferred if not provided!
            )

            target = scatter_tensor(
                target,
                global_rank_of_source,
                domain_mesh,
                placements=(
                    Replicate(),
                ),  # REPLICATE the target - gradients will still be scattered properly.
                global_shape=target.shape,  # This will be inferred if not provided!
                dtype=target.dtype,  # This will be inferred if not provided!
            )

        # The model is a completely vanilla nn.Module in every configuration.
        # ShardTensor activations auto-promote plain weights when they meet in
        # the forward pass, so no distribute_module / model surgery is needed.
        model = model_spec.build(
            img_size=full_img_size, in_channels=3, num_classes=num_classes
        )
        model = model.to(device)

        if args.domain_size > 1:
            if args.fsdp:
                # On the FSDP2 path the parameters become distributed tensors
                # anyway, so split the model's spatial params (e.g. positional
                # embeddings) across the domain to match the sequence-sharded
                # activations they meet.  Without --fsdp they stay plain and
                # replicated: ShardTensor auto-promotes them in the forward
                # pass, which keeps every parameter a plain tensor and lets
                # ordinary DDP manage the model.
                shard_spatial_params(model, domain_mesh, model_spec.spatial_param_dim)

            # Sync the replicated weights (and buffers) across the domain
            # group.  Neither DDP nor FSDP2 will do this for us on the domain
            # axis; DTensor params (sharded above) are already synced by
            # construction and are skipped.
            sync_module_over_mesh(model, domain_mesh)

        if args.compile and args.domain_size > 1:
            # Sharded attention (ring SDPA, neighborhood attention) can't live
            # inside a compiled region; each model's spec compiles the safe
            # regions around it.  Do this on the underlying module, before any
            # FSDP2 wrapping.
            model_spec.compile_regions(model)

        if args.fsdp:
            # Opt-in parameter sharding: shard the weights over the ddp axis
            # with FSDP2.  Gradients of the replicated weights are already
            # reduced over the domain axis by ShardTensor's gradient boundary,
            # so FSDP2 only needs the ddp mesh.  With ddp_size == 1 this is a
            # degenerate (size-1) shard - no communication, but the parameters
            # are uniformly DTensors on every --fsdp configuration.
            fully_shard(model, mesh=ddp_mesh)
        elif args.ddp_size > 1:
            # Replicated data parallelism: all parameters are plain
            # tensors (pos_embed is only distributed on the FSDP2 path),
            # so standard DDP just works - it broadcasts weights at
            # construction, all-reduces gradients in backward, and accepts
            # ShardTensor activations directly.  Pass the ddp mesh group
            # explicitly rather than relying on the default (world) group,
            # so DDP's size always matches the mesh axis.
            model = DDP(
                model,
                device_ids=[dm.local_rank],
                output_device=dm.local_rank,
                process_group=ddp_mesh.get_group(),
            )

        if args.compile and args.domain_size == 1:
            # No sharded attention in the graph: compile the whole model
            # (DDP-wrapped models compile fine via DDPOptimizer).  Fixed
            # shapes per sweep step, so no dynamic tracing.
            model = torch.compile(model, dynamic=False)

        result = end_to_end_benchmark(
            args, model, (x, target), full_img_size, device, num_classes
        )
        results.append(result)

        if dm.rank == 0:
            save_result_incremental(csv_filename, result, args, dm.world_size)
            print(f"Completed image size: {img_size}x{img_size}")

    if dm.rank == 0:
        print_and_save_results(results, args, precision_mode, dm.world_size)


if __name__ == "__main__":
    main()

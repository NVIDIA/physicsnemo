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

"""Main training loop."""

import copy
import json
import os
import pickle  # TODO remove
import time

import numpy as np
import psutil
import torch
from torch.nn.parallel import DistributedDataParallel
from training_stats import default_collector, report, report0
from physicsnemo.utils.diffusion.utils import (
    InfiniteSampler,
    check_ddp_consistency,
    construct_class_by_name,
    copy_params_and_buffers,
    ddp_sync,
    format_time,
    print_module_summary,
)
from misc import open_url

# # weather related
# from .YParams import YParams
# from .dataset import Era5Dataset, CWBDataset, CWBERA5DatasetV2, ZarrDataset

# ----------------------------------------------------------------------------


def training_loop(
    run_dir=".",  # Output directory.
    dataset=None,  # The dataset. Choose from ['cifar10'].
    dataset_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    # data_type=None,
    # data_config=None,
    # task=None,
    dist=None,  # distributed object
    logger0=None,  # rank 0 logger
):
    # Initialize.
    start_time = time.time()
    device = dist.device
    np.random.seed((seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.world_size
    logger0.info(f"batch_gpu: {batch_gpu}")
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.world_size

    # Load dataset
    supported_datasets = ["cifar10", "dfsr"]
    if dataset is None:
        raise RuntimeError("Please specify the dataset.")
    if dataset not in supported_datasets:
        raise ValueError(
            f'Invalid dataset: "{dataset}".' "Supported datasets: {supported_datasets}."
        )
    logger0.info(f"Loading {dataset} dataset...")

    # Load dataset: cifar10
    dataset_obj = construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset
    dataset_sampler = InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )

    # # Load dataset: weather
    # yparams = YParams(data_type + '.yaml', config_name=data_config)
    # if data_type == 'era5':
    #     dataset_obj = Era5Dataset(yparams, yparams.train_data_path, train=True, task=task)
    #     worker_init_fn = None
    # elif data_type == 'cwb':
    #     dataset_obj = CWBDataset(yparams, yparams.train_data_path, train=True, task=task)
    #     worker_init_fn = None
    # elif data_type == 'era5-cwb-v1':
    #     #filelist = os.listdir(path=yparams.cwb_data_dir + '/2018')
    #     #filelist = [name for name in filelist if "2018" in name]
    #     filelist = []
    #     for root, dirs, files in os.walk(yparams.cwb_data_dir):
    #         for file in files:
    #             if '2022' not in file:
    #                 filelist.append(file)
    #     dataset_obj = CWBERA5DatasetV2(yparams, filelist=filelist, chans=list(range(20)), train=True, task=task)
    #     worker_init_fn = dataset_obj.worker_init_fn
    # elif data_type == 'era5-cwb-v2':
    #     dataset_obj = ZarrDataset(yparams, yparams.train_data_path, train=True)
    #     worker_init_fn = None
    # elif data_type == 'era5-cwb-v3':
    #     dataset_obj = ZarrDataset(yparams, yparams.train_data_path, train=True)
    #     #worker_init_fn = dataset_obj.worker_init_fn
    #     worker_init_fn = None

    # dataset_sampler = InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    # dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, worker_init_fn=worker_init_fn, **data_loader_kwargs))

    # img_in_channels = len(yparams.in_channels)   #noise + low-res input
    # if yparams.add_grid:
    #         img_in_channels = img_in_channels + yparams.N_grid_channels

    # img_out_channels = len(yparams.out_channels)

    # if use_mean_input:  #add it to the args and store_true in yaml file
    #     img_in_channels = img_in_channels + yparams.N_grid_channels + img_out_channels

    # Construct network.
    logger0.info("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim,
    )  # cifar10
    # interface_kwargs = dict(img_resolution=yparams.crop_size_x, img_channels=img_out_channels, img_in_channels=img_in_channels, img_out_channels=img_out_channels, label_dim=0)    #weather

    if network_kwargs.class_name == "physicsnemo.models.diffusion.VEPrecond_dfsr_cond":
        # Load dataset scaling parameters to compute physics-informed conditioning variable (PDE residual w.r.t. vorticity)
        interface_kwargs["dataset_mean"] = dataset_obj.stat["mean"]
        interface_kwargs["dataset_scale"] = dataset_obj.stat["scale"]

    net = construct_class_by_name(
        **network_kwargs, **interface_kwargs
    )  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    # net = torch.compile(net)
    # Distributed data parallel
    if dist.world_size > 1:
        ddp = DistributedDataParallel(
            net,
            device_ids=[dist.local_rank],
            broadcast_buffers=dist.broadcast_buffers,
            output_device=dist.device,
            find_unused_parameters=dist.find_unused_parameters,
        )  # broadcast_buffers=True for weather data
    else:
        ddp = net

    if (
        not network_kwargs.class_name
        == "physicsnemo.models.diffusion.VEPrecond_dfsr_cond"
    ):
        if dist.rank == 0:
            with torch.no_grad():
                images = torch.zeros(
                    [
                        batch_gpu,
                        net.img_channels,
                        net.img_resolution,
                        net.img_resolution,
                    ],
                    device=device,
                )
                # img_clean = torch.zeros([batch_gpu, img_out_channels, net.img_resolution, net.img_resolution], device=device)
                # img_lr = torch.zeros([batch_gpu, img_in_channels, net.img_resolution, net.img_resolution], device=device)
                sigma = torch.ones([batch_gpu], device=device)
                labels = torch.zeros([batch_gpu, net.label_dim], device=device)
                # print_module_summary(net, [img_clean, img_lr, sigma, labels], max_nesting=2)
                print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # import pdb; pdb.set_trace()
    # breakpoint()

    # params = net.parameters()
    # print('************************************')
    # print('dist.get_rank()', dist.get_rank())
    # print('net.parameters()', net.parameters())
    # for idx, param in enumerate(net.parameters()):
    #     if idx == 230:
    #         print(f"Parameter {idx}: {param.stride()}")
    #         print(f"Parameter {idx}: {param.shape}")
    #         break
    # print('************************************')

    # Setup optimizer.
    logger0.info("Setting up optimizer...")
    loss_fn = construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss
    optimizer = construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer
    augment_pipe = (
        construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )  # training.augment.AugmentPipe
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # # Import autoresume module
    # #print('os.environ', print(os.environ))
    # # sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
    # SUBMIT_SCRIPTS = '/lustre/fsw/adlr/adlr-others/gpeled/adlr-utils/release/cluster-interface/latest'
    # sys.path.append(SUBMIT_SCRIPTS)
    # #sync autoresums across gpus ...
    # AutoResume = None
    # try:
    #     from userlib.auto_resume import AutoResume
    #     AutoResume.init()
    # except ImportError:
    #     print('AutoResume not imported')

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        logger0.info(f'Loading network weights from "{resume_pkl}"...')
        if dist.rank != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with open_url(resume_pkl, verbose=(dist.rank == 0)) as f:
            data = pickle.load(f)
        if dist.rank == 0:
            torch.distributed.barrier()  # other ranks follow
        copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        del data  # conserve memory
    if resume_state_dump:
        logger0.info(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        )
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # #check num params per gpu
    # with open(f"params_{dist.get_rank()}.txt", "w") as fo:
    #     logger0.info(net.parameters())
    #     for param in net.parameters():
    #         logger0.info(param.size())
    #         #fo.write(f"{name}\t{param.size()}\n")
    # import pdb; pdb.set_trace()

    # Train.
    logger0.info(f"Training for {total_kimg} kimg...")
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    # dist.update_progress(cur_nimg // 1000, total_kimg)  # TODO check if needed
    stats_jsonl = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad()
        for round_idx in range(num_accumulation_rounds):
            with ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):

                # # Fetch training data: weather
                # img_clean, img_lr, labels = next(dataset_iterator)

                # logger0.info(img_clean.shape)
                # logger0.info('max-clean', torch.max(img_clean))
                # logger0.info('min-clean', torch.min(img_clean))
                # logger0.info('mean-clean', torch.mean(img_clean))
                # logger0.info('std-clean', torch.std(img_clean))
                # logger0.info(img_lr.shape)
                # logger0.info('max-lr', torch.max(img_lr))
                # logger0.info('min-lr', torch.min(img_lr))
                # logger0.info('mean-lr', torch.mean(img_lr))
                # logger0.info('std-lr', torch.std(img_lr))
                # import pdb; pdb.set_trace()

                # # Normalization: weather (normalized already in the dataset)
                # img_clean = img_clean.to(device).to(torch.float32).contiguous()   #[-4.5, +4.5]
                # img_lr = img_lr.to(device).to(torch.float32).contiguous()
                # labels = labels.to(device).contiguous()

                # Fetch training data: cifar10
                images, labels = next(dataset_iterator)
                # Normalization: cifar10 (normalized already in the dataset)
                # images = images.to(device).to(torch.float32) / 127.5 - 1
                images = (
                    images.to(device).to(torch.float32)
                    if dataset == "dfsr"
                    else images.to(device).to(torch.float32) / 127.5 - 1
                )
                labels = labels.to(device)

                # loss = loss_fn(net=ddp, img_clean=img_clean, img_lr=img_lr, labels=labels, augment_pipe=augment_pipe)
                loss = loss_fn(
                    net=ddp, images=images, labels=labels, augment_pipe=augment_pipe
                )
                report("Loss/loss", loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                if dataset == "dfsr":
                    loss_sample = (
                        loss.sum()
                        .mul(loss_scaling / batch_gpu_total)
                        .detach()
                        .cpu()
                        .numpy()
                    )

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        # done = cur_nimg >= total_kimg * 1000
        if dataset == "dfsr":
            done = cur_nimg >= total_kimg
            if cur_nimg / batch_size % 500 == 0:
                logger0.info(
                    "Progress in training iterations: loss: {}, iter: {}, cur_nimg: {}, \
                    cur_tick: {}, dist.rank: {}, nimg: {}/{}".format(
                        loss_sample,
                        cur_nimg / batch_size,
                        cur_nimg,
                        cur_tick,
                        dist.rank,
                        cur_nimg,
                        total_kimg,
                    )
                )
        else:
            done = cur_nimg >= total_kimg * 1000
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [
            f"time {format_time(report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        logger0.info(" ".join(fields))

        # ckpt_dir = run_dir

        # print('AutoResume.termination_requested()', AutoResume.termination_requested())
        # print('AutoResume', AutoResume)

        # if AutoResume.termination_requested():
        #     AutoResume.request_resume()
        #     print("Training terminated. Returning")
        #     done = True
        #     #print('dist.get_rank()', dist.get_rank())
        #     #with open(os.path.join(os.path.split(ckpt_dir)[0],'resume.txt'), "w") as f:
        #     with open(os.path.join(ckpt_dir,'resume.txt'), "w") as f:
        #         f.write(os.path.join(ckpt_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
        #         print(os.path.join(ckpt_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
        #         f.close()
        #         #return 0

        # dist.print0('*********************************************')
        # dist.print0('dist.should_stop()', dist.should_stop())
        # dist.print0('done', done)
        # dist.print0('*********************************************')

        # Check for abort.  # TODO: check if needed!
        # if (not done) and dist.should_stop():
        #     done = True
        #     logger0.info()
        #     logger0.info("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            if dataset == "dfsr":
                logger0.info("Saving network snapshot.")
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if dist.world_size > 1:
                        check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.rank == 0:
                with open(
                    os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if dataset == "dfsr":
            save_full_dump = (
                (state_dump_ticks is not None)
                and (done or cur_tick % state_dump_ticks == 0)
                and dist.rank == 0
            )
        else:
            save_full_dump = (
                (state_dump_ticks is not None)
                and (done or cur_tick % state_dump_ticks == 0)
                and cur_tick != 0
                and dist.rank == 0
            )

        if save_full_dump:
            # if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and dist.get_rank() == 0:
            logger0.info("Saving full dump of the training state.")
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg//1000:06d}.pt"),
            )

        # Update logs.
        default_collector.update()
        if dist.rank == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(
                    dict(
                        default_collector.as_dict(),
                        timestamp=time.time(),
                    )
                )
                + "\n"
            )
            stats_jsonl.flush()
        # dist.update_progress(cur_nimg // 1000, total_kimg)  # TODO check if needed

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if not dataset == "dfsr":
        logger0.info()
    logger0.info("Exiting...")

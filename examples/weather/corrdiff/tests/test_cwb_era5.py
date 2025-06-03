# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import torch
# from training.dataset import CWBERA5DatasetV2
# import matplotlib.pyplot as plt
# from YParams import YParams
# import glob
# from torch_utils import misc
# from torch_utils import distributed as dist
# import dnnlib

# data_type = 'era5-cwb'
# data_config = 'full_field_train_crop64'
# params = YParams(data_type + '.yaml', config_name=data_config)

# config = {'batch_size': 8,
#           'shuffle': False,
#           'num_workers': 1}

# # Generators
# # validation_set = CWBERA5DatasetV2(params, cwb_data_dir="/lustre/fsw/sw_climate_fno/cwb-rwrf-pad-2212/all_ranges",
# #                               era5_data_dir="/lustre/fsw/sw_climate_fno/cwb-align/2018",
# #                               filelist=["2018_01.h5", "2018_02.h5"])

# import os
# filelist = os.listdir(path=params.cwb_data_dir)
# filelist = [name for name in filelist if "2018" in name]
# print(filelist)
# validation_set = CWBERA5DatasetV2(params, filelist=filelist)  #["2018_01.h5", "2018_02.h5"])

# # validation_generator = torch.utils.data.DataLoader(validation_set, **config)
# # iter_ = iter(validation_generator)


# #test dataloader
# data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
# dataset_sampler = misc.InfiniteSampler(dataset=validation_set, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=0)
# iter_ = iter(torch.utils.data.DataLoader(dataset=validation_set, sampler=dataset_sampler, batch_size=8, **data_loader_kwargs))

# x, y, _ = next(iter_)

# print('x', x.shape)
# print('y', y.shape)

# idx=0
# plt.figure(figsize=(100, 10))
# row=2
# col=20
# pos_list = []
# for i in range(col):
#     plt.subplot(row, col, i+1)
#     plt.title(f'cwb_{validation_set.target_order[i]}', fontsize=30)
#     pos = plt.imshow(y[idx, i], origin='lower')
#     pos_list.append(pos)
#     plt.colorbar(pos)
# for i in range(col):
#     plt.subplot(row, col, col+i+1)
#     plt.title(f'era5_{validation_set.target_order[i]}', fontsize=30)
#     pos = plt.imshow(x[idx, i], origin='lower')
#     #plt.colorbar(pos_list[i])
#     plt.colorbar(pos)
# plt.savefig('cwb_era5_region.jpg')

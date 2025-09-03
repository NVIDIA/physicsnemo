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

import gdown
import zipfile

url = "https://drive.google.com/uc?id=1jJdTAnhps1EIHDaBfb893fruaLPJzYKI"
output_zip = "./lj_data.zip"
output_dir = "./"

gdown.download(url, output_zip)

with zipfile.ZipFile(output_zip, "r") as zip_ref:
    zip_ref.extractall(output_dir)

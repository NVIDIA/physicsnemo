# ignore_header_test
# Copyright 2023 Stanford University
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

"""
Download dataset
"""

wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/modulus/modulus_datasets-cardiovascular-simulation/versions/0.0/zip -O modulus_datasets-cardiovascular-simulation_0.0.zip
unzip modulus_datasets-cardiovascular-simulation_0.0.zip
unzip cardiovascular_dataset.zip
mv cardiovascular_dataset/* .
rm -r cardiovascular_dataset
rm *.zip

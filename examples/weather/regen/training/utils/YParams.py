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
from ruamel.yaml import YAML
import logging


class YParams:
    """Yaml file parser"""

    def __init__(self, yaml_filename, config_name, print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename) as _file:

            for key, val in YAML().load(_file)[config_name].items():
                if print_params:
                    print(key, val)
                if val == "None":
                    val = None

                self.params[key] = val
                self.__setattr__(key, val)

        if print_params:
            print("---------------------------------------------------")

        # override setattr now so both the dict and the attrs get updated
        self.__setattr__ = self.__custom_setattr__

    def __custom_setattr__(self, key, val):
        self.params[key] = val
        super().__setattr__(key, val)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def update_params(self, config):
        """
        Update the parameters with a new config.
        """
        for key, val in config.items():
            self.params[key] = val
            self.__setattr__(key, val)

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.params.items():
            logging.info(str(key) + " " + str(val))
        logging.info("---------------------------------------------------")

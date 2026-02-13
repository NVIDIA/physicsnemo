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

r"""
Default configuration parameters for DoMINO.

Built as dictionary objects (for JSON serialization) but with
attribute access (for DoMINO's code). Built-in conversion
from Hydra objects as well.
"""

from dataclasses import fields, is_dataclass
from typing import Any


class Config(dict):
    r"""
    A dict subclass that provides attribute-style access to keys.

    Nested dicts are automatically wrapped in Config instances,
    enabling chained attribute access like `config.geometry_rep.geo_conv.base_neurons`.

    Since it inherits from dict, it's JSON serializable out of the box.

    Example usage:
        # Create from nested dict
        params = Config({
            "activation": "gelu",
            "geometry_rep": {
                "geo_conv": {"base_neurons": 32}
            }
        })

        # Attribute access works at all levels
        print(params.activation)  # "gelu"
        print(params.geometry_rep.geo_conv.base_neurons)  # 32

        # Still works as a dict
        print(params["activation"])  # "gelu"

        # JSON serializable
        import json
        json_str = json.dumps(params)

        # Create from Hydra config
        params = Config.from_hydra(hydra_cfg.model)
    """

    def __init__(self, *args, **kwargs):
        r"""Initialize from dict, positional args, or keyword args."""
        super().__init__(*args, **kwargs)
        # Convert any nested dicts to Config instances
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)

    def __getattr__(self, key: str) -> Any:
        r"""Allow attribute-style access to dict keys."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        r"""Allow attribute-style setting of dict keys, wrapping dicts as Config."""
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        self[key] = value

    def __delattr__(self, key: str):
        r"""Allow attribute-style deletion of dict keys."""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    @classmethod
    def from_hydra(cls, hydra_cfg: Any) -> "Config":
        r"""
        Convert a Hydra/OmegaConf config object to a Config instance.

        Parameters
        ----------
        hydra_cfg : Any
            A Hydra DictConfig, dataclass, or dict-like object.

        Returns
        -------
        Config
            A Config instance with values from ``hydra_cfg``.
        """
        if hydra_cfg is None:
            return cls()

        # OmegaConf DictConfig
        if hasattr(hydra_cfg, "to_container"):
            return cls(hydra_cfg.to_container(resolve=True))

        # Dataclass
        if hasattr(hydra_cfg, "__dataclass_fields__"):
            return cls(_dataclass_to_dict(hydra_cfg))

        # Already a dict
        if isinstance(hydra_cfg, dict):
            return cls(hydra_cfg)

        # Object with __dict__
        if hasattr(hydra_cfg, "__dict__"):
            return cls(
                {k: v for k, v in hydra_cfg.__dict__.items() if not k.startswith("_")}
            )

        raise TypeError(f"Cannot convert {type(hydra_cfg)} to Config")


def _dataclass_to_dict(obj: Any) -> dict:
    r"""Recursively convert a dataclass to a dict."""
    result = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        if is_dataclass(value) and not isinstance(value, type):
            result[field.name] = _dataclass_to_dict(value)
        else:
            result[field.name] = value
    return result


# ============================================================================
# Default model parameters for Domino
# ============================================================================

DEFAULT_MODEL_PARAMS = Config(
    {
        "model_type": "combined",
        "activation": "gelu",
        "interp_res": [128, 64, 64],
        "use_sdf_in_basis_func": True,
        "positional_encoding": False,
        "surface_neighbors": True,
        "num_neighbors_surface": 7,
        "num_neighbors_volume": 10,
        "use_surface_normals": True,
        "use_surface_area": True,
        "encode_parameters": False,
        "combine_volume_surface": False,
        "geometry_encoding_type": "both",
        "solution_calculation_mode": "two-loop",
        "geometry_rep": {
            "base_filters": 8,
            "geo_conv": {
                "base_neurons": 32,
                "base_neurons_in": 1,
                "base_neurons_out": 1,
                "surface_hops": 1,
                "volume_hops": 1,
                "volume_radii": [0.1, 0.5, 1.0, 2.5],
                "volume_neighbors_in_radius": [32, 64, 128, 256],
                "surface_radii": [0.01, 0.05, 1.0],
                "surface_neighbors_in_radius": [8, 16, 128],
                "activation": "gelu",
                "fourier_features": False,
                "num_modes": 5,
            },
            "geo_processor": {
                "base_filters": 8,
                "activation": "gelu",
                "processor_type": "unet",
                "self_attention": False,
                "cross_attention": False,
                "volume_sdf_scaling_factor": [0.04],
                "surface_sdf_scaling_factor": [0.01, 0.02, 0.04],
            },
        },
        "geometry_local": {
            "base_layer": 512,
            "volume_neighbors_in_radius": [64, 128],
            "surface_neighbors_in_radius": [32, 128],
            "volume_radii": [0.1, 0.25],
            "surface_radii": [0.05, 0.25],
        },
        "nn_basis_functions": {
            "base_layer": 512,
            "fourier_features": True,
            "num_modes": 5,
            "activation": "gelu",
        },
        "local_point_conv": {
            "activation": "gelu",
        },
        "aggregation_model": {
            "base_layer": 512,
            "activation": "gelu",
        },
        "position_encoder": {
            "base_neurons": 512,
            "activation": "gelu",
            "fourier_features": True,
            "num_modes": 5,
        },
        "parameter_model": {
            "base_layer": 512,
            "fourier_features": False,
            "num_modes": 5,
            "activation": "gelu",
        },
    }
)

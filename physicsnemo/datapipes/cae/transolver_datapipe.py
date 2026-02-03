# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""
This code provides the datapipe for reading the processed npy files,
generating multi-res grids, calculating signed distance fields,
sampling random points in the volume and on surface,
normalizing fields and returning the output tensors as a dictionary.

This datapipe also non-dimensionalizes the fields, so the order in which the variables should
be fixed: velocity, pressure, turbulent viscosity for volume variables and
pressure, wall-shear-stress for surface variables. The different parameters such as
variable names, domain resolution, sampling size etc. are configurable in config.yaml.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from physicsnemo.datapipes.cae.cae_dataset import (
    CAEDataset,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.domino.utils import (
    normalize,
    standardize,
    unnormalize,
    unstandardize,
)
from physicsnemo.nn.functional import signed_distance_field


@dataclass
class TransolverDataConfig:
    """
    Configuration for Transolver data processing pipeline.

    Attributes:

    Attributes:
        data_path: Path to the dataset to load.
        model_type: Type of the model ("surface" or "volume").
        resolution: Resolution of the sampled data, per batch.
        include_normals: Whether to include surface normals in embeddings.
        include_sdf: Whether to include signed distance fields in embeddings.
        translational_invariance: Enable translational adjustment using center of mass.
        reference_origin: Origin for translational invariance, defaults to the center of mass.
        broadcast_global_features: Whether to apply global features across all points.
        volume_sample_from_disk: Whether to sample points from the disk for volume data.
        return_mesh_features: Whether to return the mesh areas and normals for the surface data.
            Used to compute force coefficients. Transformations are applied to the mesh coordinates.
    """

    data_path: Path | None
    model_type: Literal["surface", "volume", "combined"] = "surface"
    resolution: int = 200_000

    # Control what features are added to the inputs to the model:
    include_normals: bool = True
    include_sdf: bool = True

    # Control the geometry configuration:
    include_geometry: bool = False
    geometry_sampling: int = 300_000

    # For controlling the normalization of target values:
    scaling_type: Optional[Literal["min_max_scaling", "mean_std_scaling"]] = None
    surface_factors: Optional[torch.Tensor] = None
    volume_factors: Optional[torch.Tensor] = None

    ############################################################
    # Translation invariance configuration:
    ############################################################

    translational_invariance: bool = False
    # If none, uses the center of mass from the STLs:
    reference_origin: torch.Tensor | None = None

    ############################################################
    # Scale Invariance:
    ############################################################
    scale_invariance: bool = False
    # Must be set if scale invariance is enabled.
    # Should be castable to torch tensor
    reference_scale: list[float] | None = None

    broadcast_global_features: bool = True

    volume_sample_from_disk: bool = True

    return_mesh_features: bool = False

    def __post_init__(self):
        if self.data_path is not None:
            # Ensure data_path is a Path object:
            if isinstance(self.data_path, str):
                self.data_path = Path(self.data_path)
            self.data_path = self.data_path.expanduser()

            if not self.data_path.exists():
                raise ValueError(f"Path {self.data_path} does not exist")

            if not self.data_path.is_dir():
                raise ValueError(f"Path {self.data_path} is not a directory")

        if self.scaling_type is not None:
            if self.scaling_type not in [
                # "min_max_scaling",
                "mean_std_scaling",
            ]:
                raise ValueError(
                    f"scaling_type should be one of ['min_max_scaling', 'mean_std_scaling'], got {self.scaling_type}"
                )

        if self.scale_invariance:
            if self.reference_scale is None:
                raise ValueError(
                    "reference_scale must be set if scale invariance is enabled"
                )

            self.reference_scale = list(self.reference_scale)
            if len(self.reference_scale) != 3:
                raise ValueError("reference_scale must be a list of 3 floats")
            self.reference_scale = (
                torch.tensor(self.reference_scale).to(torch.float32).reshape(1, 3)
            )


class TransolverDataPipe(Dataset):
    """
    Base Datapipe for Transolver

    Leverages a dataset for the actual reading of the data, and this
    object is responsible for preprocessing the data.

    """

    def __init__(
        self,
        input_path,
        model_type: Literal["surface", "volume"],
        pin_memory: bool = False,
        **data_config_overrides,
    ):
        # Perform config packaging and validation
        self.config = TransolverDataConfig(
            data_path=input_path, model_type=model_type, **data_config_overrides
        )

        # Set up the distributed manager:
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()

        self.dataset = None

    def preprocess_surface_data(
        self,
        data_dict,
        center_of_mass: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ):
        positions = data_dict["surface_mesh_centers"]

        if self.config.resolution is not None:
            idx = torch.multinomial(
                torch.ones(data_dict["surface_mesh_centers"].shape[0]),
                self.config.resolution,
            )
        else:
            idx = None

        if idx is not None:
            positions = positions[idx]

        # This is a center of mass computation for the stl surface,
        # using the size of each mesh point as weight.
        if self.config.translational_invariance:
            positions -= center_of_mass

        if self.config.scale_invariance:
            positions = positions / scale_factor

        # Build the embeddings:
        embeddings_inputs = [positions]

        if self.config.include_normals:
            normals = data_dict["surface_normals"]
            if idx is not None:
                normals = normals[idx]
            normals = normals / torch.norm(normals, dim=-1, keepdim=True)
            embeddings_inputs.append(normals)

        embeddings = torch.cat(embeddings_inputs, dim=-1)

        fields = data_dict["surface_fields"]
        if idx is not None:
            fields = fields[idx]

        if self.config.scaling_type is not None:
            fields = self.scale_model_targets(fields, self.config.surface_factors)

        if "air_density" in data_dict and "stream_velocity" in data_dict:
            # Build fx:
            fx_inputs = [
                data_dict["air_density"],
                data_dict["stream_velocity"],
            ]
            fx = torch.stack(fx_inputs, dim=-1)

            if self.config.broadcast_global_features:
                fx = fx.broadcast_to(embeddings.shape[0], -1)
            else:
                fx = fx.unsqueeze(0)

            return {
                "embeddings": embeddings,
                "fx": fx,
                "fields": fields,
            }

        else:
            return {
                "embeddings": embeddings,
                "fields": fields,
            }

    def preprocess_volume_data(
        self,
        data_dict,
        center_of_mass: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ):
        positions = data_dict["volume_mesh_centers"]

        if self.config.resolution is not None:
            idx = poisson_sample_indices_fixed(
                positions.shape[0], self.config.resolution, device=positions.device
            )
        else:
            idx = None

        if idx is not None:
            positions = positions[idx]

        # We need the CoM for some operations, regardless of translation invariance:
        if center_of_mass is None:
            center_of_mass = torch.mean(data_dict["stl_centers"], dim=0).unsqueeze(0)

        if self.config.translational_invariance:
            positions -= center_of_mass

        if self.config.scale_invariance:
            positions = positions / scale_factor

        # Build the embeddings:
        embeddings_inputs = [positions]

        if self.config.include_sdf:
            coords = data_dict["stl_coordinates"]
            # Remove CoM, optionally:
            if self.config.translational_invariance:
                coords = coords - center_of_mass

            # Set scale, optionally:
            if self.config.scale_invariance:
                coords = coords / scale_factor

            sdf, closest_points = signed_distance_field(
                coords,
                data_dict["stl_faces"].flatten().to(torch.int32),
                positions,
                use_sign_winding_number=True,
            )

            embeddings_inputs.append(sdf.reshape(-1, 1))
        else:
            closest_points = center_of_mass

            # Make sure we have a scale-invariant component to subtract
            # from scale-invariant positions, below:
            if self.config.scale_invariance:
                closest_points = closest_points / scale_factor

        if self.config.include_normals:
            normals = positions - closest_points

            # Be sure to normalize:

            # Sometimes, if the points are very close or on the mesh, the
            # sdf is 0.0, and the norm goes to 0.0

            distance_to_closest_point = torch.norm(positions - closest_points, dim=-1)
            null_points = distance_to_closest_point < 1e-6

            # In these cases, we update the vector to be from the center of mass
            normals[null_points] = positions[null_points] - center_of_mass

            norm = torch.norm(normals, dim=-1, keepdim=True) + 1e-6
            normals = normals / norm

            embeddings_inputs.append(normals)

        embeddings = torch.cat(embeddings_inputs, dim=-1)

        fields = data_dict["volume_fields"]
        if idx is not None:
            fields = fields[idx]

        if self.config.scaling_type is not None:
            fields = self.scale_model_targets(fields, self.config.volume_factors)

        if "air_density" in data_dict and "stream_velocity" in data_dict:
            # Build fx:
            fx_inputs = [
                data_dict["air_density"],
                data_dict["stream_velocity"],
            ]
            fx = torch.stack(fx_inputs, dim=-1)

            if self.config.broadcast_global_features:
                fx = fx.broadcast_to(embeddings.shape[0], -1)
            else:
                fx = fx.unsqueeze(0)

            return {
                "embeddings": embeddings,
                "fx": fx,
                "fields": fields,
            }
        else:
            return {
                "embeddings": embeddings,
                "fields": fields,
            }

    def process_geometry(
        self,
        data_dict,
        center_of_mass: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ):
        """
        Process the geometry data.
        """
        geometry_coordinates = data_dict["stl_coordinates"]
        if self.config.geometry_sampling is not None:
            # idx = torch.multinomial(
            #     torch.ones(data_dict["stl_coordinates"].shape[0]),
            #     self.config.geometry_sampling,
            # )
            idx = poisson_sample_indices_fixed(
                data_dict["stl_coordinates"].shape[0],
                self.config.geometry_sampling,
                device=data_dict["stl_coordinates"].device,
            )
            geometry_coordinates = geometry_coordinates[idx]

        if self.config.translational_invariance:
            geometry_coordinates -= center_of_mass

        if self.config.scale_invariance:
            geometry_coordinates = geometry_coordinates / scale_factor

        return geometry_coordinates

    @torch.no_grad()
    def process_data(self, data_dict):
        """
        Preprocess the data.  We have slight differences between surface and volume data processing,
        mostly revolving around the keys that represent the inputs.

        - For surface data, we use the mesh coordinates and normals as the embeddings.
            - Normals are always normalized to 1.0, and are a relative direction.
            - coordinates can be shifted to the center of mass, and then the whole
              coordinate system can be aligned to the preferred direction.
            - SDF is identically 0 for surface data.
            - Optionally, if the scale invariance is enabled, the coordinates
              are scaled by the (maybe-rotated) scale factor.

        - For Volume data: we still use the volume coordinates
            - normals are approximated as the direction between the volume point
              and closest mesh point.  Normalized to 1.0.
            - SDF is not zero for volume data.


        To make the calculations consistent and logical to follow:
        - First, get the coordinates (volume_mesh_centers or surface_mesh_centers, usually)
          which is a configuration.
        - Second, get the STL information.  We need the "stl_vertices" and "stl_indices"
          to compute an SDF.  We downsample "stl_coordinates" to potentially encode
          a geometry tensor, which is optional.

        Then, start imposing optional symmetries:
        - Impose translation invariance.  For every "position-like" tensor, subtract
          off the reference_origin if translation invariance is enabled.
        - Second, impose scale invariance: for every position-like tensor, multiply
          by the reference scale.
        - Finally, apply rotation invariance.  Normals are rotated, points are rotated.
          Roation requires not just a reference vector (in the config) but a
          vector unique to this example to come from the data - we have to rotate to it.

        After that, the rest is simple:
          - Spatial Encodings are the point locations + normal vectors (optional) + sdf (optional)
            - If the normals aren't provided, we derive them from the center of mass (without SDF) or SDF point (with SDF)
          - Geometry encoding (if using) is the STL coordinates, downsampled.
          - parameter encodings are straight forward vectors / reference values.

        The downstream applications can take the embeddings and the features as needed.

        """

        # Validate that all required keys are present in data_dict
        required_keys = [
            "stl_centers",
        ]

        if self.config.model_type == "volume" or self.config.model_type == "combined":
            # We need these for the SDF calculation:
            required_keys.extend(
                [
                    "stl_coordinates",
                    "stl_faces",
                ]
            )
        elif (
            self.config.model_type == "surface" or self.config.model_type == "combined"
        ):
            required_keys.extend(
                [
                    "surface_normals",
                ]
            )

        if self.config.translational_invariance:
            if self.config.reference_origin is not None:
                center_of_mass = self.config.reference_origin
            else:
                center_of_mass = torch.mean(data_dict["stl_centers"], dim=0)
            center_of_mass = center_of_mass.unsqueeze(0)  # (1, 3)
        else:
            center_of_mass = None

        if self.config.model_type == "surface" or self.config.model_type == "combined":
            required_keys.extend(
                [
                    "surface_fields",
                    "surface_mesh_centers",
                ]
            )
        elif self.config.model_type == "volume" or self.config.model_type == "combined":
            required_keys.extend(
                [
                    "volume_fields",
                    "volume_mesh_centers",
                ]
            )

        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in data_dict: {missing_keys}. "
                f"Required keys are: {required_keys}"
            )

        scale_factor = (
            self.config.reference_scale if self.config.scale_invariance else None
        )

        if self.config.model_type == "surface":
            outputs = self.preprocess_surface_data(
                data_dict, center_of_mass, scale_factor
            )
        elif self.config.model_type == "volume":
            outputs = self.preprocess_volume_data(
                data_dict, center_of_mass, scale_factor
            )
        elif self.config.model_type == "combined":
            outputs_surf = self.preprocess_surface_data(
                data_dict, center_of_mass, scale_factor
            )

            outputs_vol = self.preprocess_volume_data(
                data_dict, center_of_mass, scale_factor
            )

            outputs = {}
            outputs["embeddings"] = [
                outputs_surf["embeddings"],
                outputs_vol["embeddings"],
            ]
            # This should be the same in either:
            outputs["fx"] = outputs_surf["fx"]
            outputs["fields"] = [outputs_surf["fields"], outputs_vol["fields"]]

        if self.config.include_geometry:
            outputs["geometry"] = self.process_geometry(
                data_dict, center_of_mass, scale_factor
            )

        if self.config.return_mesh_features:
            outputs["surface_areas"] = data_dict["surface_areas"]
            outputs["surface_normals"] = data_dict["surface_normals"]

        if "air_density" in data_dict:
            outputs["air_density"] = data_dict["air_density"]
        if "stream_velocity" in data_dict:
            outputs["stream_velocity"] = data_dict["stream_velocity"]

        return outputs

    def scale_model_targets(
        self, fields: torch.Tensor, factors: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale the model targets based on the configured scaling factors.
        """
        if self.config.scaling_type == "mean_std_scaling":
            field_mean = factors["mean"]
            field_std = factors["std"]
            return standardize(fields, field_mean, field_std)
        elif self.config.scaling_type == "min_max_scaling":
            field_min = factors["min"]
            field_max = factors["max"]
            return normalize(fields, field_max, field_min)

    def unscale_model_targets(
        self,
        fields: torch.Tensor | None = None,
        air_density: torch.Tensor | None = None,
        stream_velocity: torch.Tensor | None = None,
        factor_type: Literal["surface", "volume", "auto"] = "auto",
    ):
        """
        Unscale the model outputs based on the configured scaling factors.

        The unscaling is included here to make it a consistent interface regardless
        of the scaling factors and type used.

        """

        match factor_type:
            case "surface":
                factors = self.config.surface_factors
            case "volume":
                factors = self.config.volume_factors
            case "auto":
                if self.config.model_type == "surface":
                    factors = self.config.surface_factors
                elif self.config.model_type == "volume":
                    factors = self.config.volume_factors
                else:
                    raise ValueError(f"Invalid model type {self.config.model_type}")

        if self.config.scaling_type == "mean_std_scaling":
            field_mean = factors["mean"]
            field_std = factors["std"]
            fields = unstandardize(fields, field_mean, field_std)
        elif self.config.scaling_type == "min_max_scaling":
            field_min = factors["min"]
            field_max = factors["max"]
            fields = unnormalize(fields, field_max, field_min)

        # if air_density is not None and stream_velocity is not None:
        #     fields = fields * air_density * stream_velocity**2

        return fields

    def set_dataset(self, dataset: Iterable) -> None:
        """
        Pass a dataset to the datapipe to enable iterating over both in one pass.
        """
        self.dataset = dataset

        if self.config.scale_invariance:
            self.config.reference_scale = self.config.reference_scale.to(
                self.dataset.output_device
            )

        if self.config.model_type == "volume" and self.config.volume_sample_from_disk:
            # We deliberately double the data to read compared to the sampling size:
            self.dataset.set_volume_sampling_size(25 * self.config.resolution)

    def __len__(self):
        if self.dataset is not None:
            return len(self.dataset)
        else:
            return 0

    def __getitem__(self, idx):
        """
        Function for fetching and processing a single file's data.

        Domino, in general, expects one example per file and the files
        are relatively large due to the mesh size.

        Requires the user to have set a dataset via `set_dataset`.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not present")

        # Get the data from the dataset.
        # Under the hood, this may be fetching preloaded data.
        data_dict = self.dataset[idx]

        return self.__call__(data_dict)

    def __call__(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process the incoming data dictionary.
        - Processes the data
        - moves it to GPU
        - adds a batch dimension

        Args:
            data_dict: Dictionary containing the data to process as torch.Tensors.

        Returns:
            Dictionary containing the processed data as torch.Tensors.

        """
        outputs = self.process_data(data_dict)
        for key in outputs.keys():
            if isinstance(outputs[key], list):
                outputs[key] = [item.unsqueeze(0) for item in outputs[key]]
            else:
                outputs[key] = outputs[key].unsqueeze(0)

        return outputs

    def __iter__(self):
        if self.dataset is None:
            raise ValueError(
                "Dataset is not present, can not use the datapipe as an iterator."
            )

        for i, batch in enumerate(self.dataset):
            yield self.__call__(batch)


def create_transolver_dataset(
    cfg: DictConfig,
    phase: Literal["train", "val", "test"],
    surface_factors: dict[str, torch.Tensor] | None = None,
    volume_factors: dict[str, torch.Tensor] | None = None,
    device_mesh: torch.distributed.DeviceMesh | None = None,
    placements: dict[str, torch.distributed.tensor.Placement] | None = None,
):
    model_type = cfg.mode
    if phase == "train":
        input_path = cfg.train.data_path
    elif phase == "val":
        input_path = cfg.val.data_path
    # elif phase == "test":
    # input_path = cfg.eval.test_path
    else:
        raise ValueError(f"Invalid phase {phase}")

    # The dataset path works in two pieces:
    # There is a core "dataset" which is loading data and moving to GPU
    # And there is the preprocess step, here.

    # Optionally, and for backwards compatibility, the preprocess
    # object can accept a dataset which will enable it as an iterator.
    # The iteration function will loop over the dataset, preprocess the
    # output, and return it.

    keys_to_read = cfg.data_keys

    overrides = {}

    dm = DistributedManager()

    if torch.cuda.is_available():
        device = dm.device
        consumer_stream = torch.cuda.default_stream()
    else:
        device = torch.device("cpu")
        consumer_stream = None

    if cfg.get("preload_depth", None) is not None:
        preload_depth = cfg.preload_depth
    else:
        preload_depth = 1

    if cfg.get("pin_memory", None) is not None:
        pin_memory = cfg.pin_memory
    else:
        pin_memory = False

    # These are keys that could be set in the config,
    # but have a sensible default if not.
    optional_cfg_keys = [
        "include_normals",
        "include_sdf",
        "volume_sample_from_disk",
        "broadcast_global_features",
        "include_geometry",
        "geometry_sampling",
        "translational_invariance",
        "reference_origin",
        "scale_invariance",
        "reference_scale",
        "return_mesh_features",
    ]

    for optional_key in optional_cfg_keys:
        if cfg.get(optional_key, None) is not None:
            overrides[optional_key] = cfg[optional_key]

    dataset = CAEDataset(
        data_dir=input_path,
        keys_to_read=keys_to_read,
        keys_to_read_if_available={},
        output_device=device,
        preload_depth=preload_depth,
        pin_memory=pin_memory,
        device_mesh=device_mesh,
        placements=placements,
        consumer_stream=consumer_stream,
    )

    datapipe = TransolverDataPipe(
        input_path,
        resolution=cfg.resolution,
        surface_factors=surface_factors,
        volume_factors=volume_factors,
        model_type=model_type,
        scaling_type="mean_std_scaling",
        **overrides,
    )

    datapipe.set_dataset(dataset)

    return datapipe


def poisson_sample_indices_fixed(N: int, k: int, device=None):
    """
    This function is a nearly uniform sampler of indices for when the
    number of indices to sample is very, very large.  It's useful when
    the number of indices to sample is larger than 2^24 and torch
    multinomial can't work.  Unlike using randperm, there is no
    need to materialize and randomize the entire tensor of indices.

    """
    # Draw exponential gaps off of random initializations:
    gaps = torch.rand(k, device=device).exponential_()

    summed = gaps.sum()

    # Normalize so total cumulative sum == N
    gaps *= N / summed

    # Compute cumulative positions
    idx = torch.cumsum(gaps, dim=0)

    # Shift down so range starts at 0 and ends below N
    idx -= gaps[0] / 2

    # Round to nearest integer index
    idx = torch.clamp(idx.floor().long(), min=0, max=N - 1)

    return idx

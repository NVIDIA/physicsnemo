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
import re
import glob
import time
import json
import logging
import numpy as np
import torch
from torch_geometric.data import Data
from hydra.utils import to_absolute_path
from sim_utils import EclReader, Well, Grid
from multiprocessing import Pool, cpu_count, Manager
from scipy.interpolate import interp1d

# Module-level logger
logger = logging.getLogger(__name__)


class ReservoirGraphBuilder:
    """Builds graph structures from reservoir simulation data.

    This class processes reservoir simulation output files and creates
    PyTorch Geometric graph structures for machine learning tasks.
    """

    ECL_SIMULATORS = ["OPM", "ECLIPSE", "IX"]

    # CMG_SIMULATORS = ["IMEX", "GEM", "STARS"] # TODO: implement
    # NEXUS_SIMULATORS = [""] # TODO: implement
    def __init__(self, cfg):
        self.sim_dir = to_absolute_path(cfg.dataset.sim_dir)
        self.simulator = cfg.dataset.get("simulator", "").upper()
        self.num_samples = cfg.dataset.get(
            "num_samples", None
        )  # Limit number of samples to process
        self.num_preprocess_workers = cfg.preprocessing.get(
            "num_preprocess_workers", 4
        )  # Number of parallel workers for sample processing

        # Get graph configuration
        self.graph_config = cfg.dataset.get("graph", None)
        if self.graph_config is None:
            raise ValueError(
                "'dataset.graph' section is required in configuration. Please provide graph configuration."
            )

        # Set prev_timestep_idx based on prev_timesteps
        self.prev_timestep_idx = (
            self.graph_config.node_features.dynamic.get("prev_timesteps", 0) + 1
        )

        # Create vars config for reading simulation data and graph creation
        # Check which coordinate components are requested as static features
        self.requested_coordinates = [
            coord
            for coord in ["X", "Y", "Z"]
            if coord in self.graph_config.node_features.static
        ]
        self.include_coordinates_as_features = len(self.requested_coordinates) == 3

        # Don't filter out X, Y, Z - let them be handled as regular static variables if requested
        static_vars = list(self.graph_config.node_features.static)

        self.vars = {
            "grid": {
                "static": static_vars,
                "dynamic": list(self.graph_config.node_features.dynamic.variables),
            }
        }

        # Add time_series variables if specified
        if hasattr(self.graph_config.node_features, "time_series"):
            self.vars["time_series"] = list(self.graph_config.node_features.time_series)
        else:
            self.vars["time_series"] = []

        self.output_vars = list(self.graph_config.target_vars.node_features)

        # Get nonlinear scaling configuration
        self.nonlinear_scaling = getattr(self.graph_config, "nonlinear_scaling", [])
        self.global_vars = self.graph_config.global_features

        # Add edge features to static vars if specified
        if hasattr(self.graph_config, "edge_features"):
            edge_vars = self.graph_config.edge_features
            if isinstance(edge_vars, list):
                self.vars["grid"]["static"].extend(list(edge_vars))

        # Add nonlinear scaling if specified
        if hasattr(self.graph_config, "nonlinear_scaling"):
            self.vars["nonlinear_scaling"] = list(self.graph_config.nonlinear_scaling)

        self._validate_config()
        self._parse_dist()
        self._set_output_path()

    def _validate_config(self) -> None:
        if self.simulator not in self.ECL_SIMULATORS:
            raise NotImplementedError(
                f"Unsupported simulator '{self.simulator}'. Supported simulators are: {self.ECL_SIMULATORS}"
            )
        if self.vars is None:
            raise ValueError("'vars' cannot be empty.")
        if self.output_vars is None:
            raise ValueError("'output_vars' must be specified in config.")

        # Validate output_vars
        if not hasattr(self.output_vars, "__iter__"):
            raise ValueError("'output_vars' must be iterable.")

    def _parse_dist(self):
        self._dist_map = {}
        entries = self.vars.get("nonlinear_scaling", [])
        for entry in entries:
            key, method = entry.split(":")
            self._dist_map[key.upper()] = method.upper()

    def _find_primary_input_files(self):
        """Find simulation primary input files based on simulator type.

        Returns
            list: Sorted list of primary input file paths

        Note:
            - ECLIPSE/OPM: *.DATA files
            - IX: *.AFI files
            - CMG, NEXUS: TODO - implement support
        """
        # Determine file extension based on simulator type
        if self.simulator == "IX":
            extension = "*.AFI"
            pattern_regex = r"_(\d+)\.AFI$"
        else:  # ECLIPSE, OPM
            extension = "*.DATA"
            pattern_regex = r"_(\d+)\.DATA$"

        pattern = os.path.join(self.sim_dir, "**", extension)
        return sorted(
            glob.glob(pattern, recursive=True),
            key=lambda fp: int(
                re.search(pattern_regex, os.path.basename(fp), re.IGNORECASE).group(1)
            )
            if re.search(pattern_regex, os.path.basename(fp), re.IGNORECASE)
            else float("inf"),
        )

    def _set_output_path(self):
        out_dir = os.path.join(
            os.path.dirname(self.sim_dir), f"{os.path.basename(self.sim_dir)}.dataset"
        )
        self._output_path_graph = os.path.join(out_dir, "graphs")
        self._output_path_well = os.path.join(out_dir, "well.json")
        # Note: Directory creation is handled by preprocessor to ensure correct job-specific paths

    def get_completion_info(self, grid, well_info) -> list:
        """Extract well completion information from simulation data.

        Parameters
            grid: Grid object containing grid information.
            well_info: Dictionary containing well data from restart files.

        Returns
            list: List of Well objects with completion data for each timestep.
        """
        wells_lst = []
        for i in range(len(well_info["ZWEL"])):
            INTEHEAD = well_info["INTEHEAD"][i]
            ZWEL = well_info["ZWEL"][i]
            IWEL = well_info["IWEL"][i]
            ICON = well_info["ICON"][i]
            SCON = well_info["SCON"][i]

            NWELLS, NCWMAX, NICONZ, NSCONZ = (
                INTEHEAD[16],
                INTEHEAD[17],
                INTEHEAD[32],
                INTEHEAD[33],
            )
            if NWELLS == 0:
                wells_lst.append([])  # no wells operating
                continue

            IWEL = IWEL.reshape((-1, NWELLS), order="F")
            ICON = ICON.reshape((NICONZ, NCWMAX, NWELLS), order="F")
            SCON = SCON.reshape((NSCONZ, NCWMAX, NWELLS), order="F")

            well_names = ["".join(row).strip() for row in ZWEL if "".join(row).strip()]
            wells = {
                name: Well(name=name, type_id=IWEL[6, i], stat=IWEL[10, i])
                for i, name in enumerate(well_names)
            }

            for iwell, name in enumerate(well_names):
                for id, icon in enumerate(ICON[:, :, iwell].T):
                    scon = SCON[:, id, iwell]
                    if icon[0] == 0:
                        break
                    I, J, K = icon[1:4]
                    well = wells[name]
                    well.add_completion(
                        I=I, J=J, K=K, dir=icon[13], stat=icon[5], conx_factor=scon[0]
                    )
                    well.completions[-1].set_ijk(grid.ijk_from_I_J_K(I, J, K))

            wells_lst.append(wells)

        return wells_lst

    def _apply_nonlinear_scaling(self, data, var_name):
        """
        Apply nonlinear scaling to data based on configuration.

        Parameters
        -----------
        data : np.ndarray
            Input data array
        var_name : str
            Variable name to check for scaling configuration

        Returns
        --------
        np.ndarray
            Scaled data array
        """
        # Check if this variable has nonlinear scaling configured
        for scaling_config in self.nonlinear_scaling:
            if ":" in scaling_config:
                var, scaling_type = scaling_config.split(":", 1)
                if var == var_name:
                    if scaling_type.upper() == "LOG10":
                        # Apply log10 scaling: log10(max(data, 1e-10))
                        # Use 1e-10 as minimum to avoid log(0)
                        data_scaled = np.log10(np.maximum(data, 1e-10))
                        return data_scaled
                    elif scaling_type.upper() == "LOG":
                        # Apply natural log scaling: log(max(data, 1e-10))
                        data_scaled = np.log(np.maximum(data, 1e-10))
                        return data_scaled
                    elif scaling_type.upper() == "SQRT":
                        # Apply square root scaling: sqrt(max(data, 0))
                        data_scaled = np.sqrt(np.maximum(data, 0))
                        return data_scaled
                    else:
                        logger.warning(
                            f"Unknown scaling type '{scaling_type}' for variable '{var_name}'. Skipping scaling."
                        )

        # No scaling configured for this variable
        return data

    def build_graph_from_simulation_data(
        self,
        grid,
        wells_data,
        data,
        sample_idx,
        timestep_idx=0,
        case_name=None,
        time_series_data=None,
    ):
        """
        Build a reservoir simulation graph from processed data.

        Parameters
        -----------
        grid : Grid object
            Grid object with all grid information
        wells_data : list
            List of Well objects for this sample
        data : dict
            Combined data dictionary containing both static and dynamic properties
        sample_idx : int
            Index of the sample
        timestep_idx : int
            Current timestep index
        case_name : str, optional
            Name of the case
        time_series_data : dict, optional
            Interpolated time series data {well_name: {var_name: [values]}}

        Returns
        --------
        graph : pyg.data.Data
            Graph with node and edge features
        """

        # Get connections and transmissibility
        conx, tran = grid.get_conx_tran()
        edge_index = conx.T  # (2, E)
        tran = self._apply_nonlinear_scaling(tran, var_name="TRAN")
        edge_features = tran.reshape(-1, 1)  # (E, 1)

        # Create coordinates array for the graph (optional - only if coordinates are not in node features)
        coordinates = (
            None
            if self.include_coordinates_as_features
            else np.column_stack([grid.X, grid.Y, grid.Z])
        )  # (N_active, 3)

        # Extract input variables (current timestep)
        input_tensors = []
        input_var_names = []

        # Add static variables (including X, Y, Z if requested as node features in config)
        for var in self.vars["grid"]["static"]:
            try:
                var_data = np.asarray(data[var], dtype=np.float32)
                var_data = self._apply_nonlinear_scaling(var_data, var)

                input_tensors.append(
                    torch.tensor(var_data, dtype=torch.float32).unsqueeze(1)
                )
                input_var_names.append(var)
            except Exception as e:
                logger.error(
                    f"Failed to process static variable '{var}' - {type(e).__name__}: {e}"
                )
                return None

        # Add dynamic variables (multiple previous timesteps) - including completion
        dynamic_vars = list(self.vars["grid"]["dynamic"])

        for var in dynamic_vars:
            if var in data:
                # Check if we have enough timesteps for the required history
                if timestep_idx + 1 < self.prev_timestep_idx:
                    # Not enough history - skip this timestep (expected for early timesteps)
                    return None

                try:
                    # Extract data from multiple previous timesteps
                    var_data_list = []
                    for t in range(self.prev_timestep_idx):
                        prev_timestep = timestep_idx - t  # Go backwards in time
                        if prev_timestep < 0 or prev_timestep >= len(data[var]):
                            logger.error(
                                f"Variable '{var}' not available at timestep {prev_timestep} - skipping graph"
                            )
                            return None
                        var_data_list.append(data[var][prev_timestep])

                    # Stack the historical data: [prev_timestep_idx, n_active]
                    var_data_stacked = np.stack(var_data_list, axis=0)
                    # Reshape to [n_active, prev_timestep_idx] to match static features
                    var_data_reshaped = (
                        var_data_stacked.T
                    )  # Transpose to [n_active, prev_timestep_idx]

                    # Apply nonlinear scaling if specified in config
                    var_data_reshaped = self._apply_nonlinear_scaling(
                        var_data_reshaped, var
                    )

                    input_tensors.append(
                        torch.tensor(var_data_reshaped, dtype=torch.float32)
                    )
                    # Add individual names for each timestep: current, prev_1, prev_2, ...
                    for t in range(self.prev_timestep_idx):
                        if t == 0:
                            input_var_names.append(f"{var}_current")
                        else:
                            input_var_names.append(f"{var}_prev_{t}")
                except Exception as e:
                    logger.error(
                        f"Failed to process variable '{var}' - {type(e).__name__}: {e}"
                    )
                    return None
            else:
                logger.error(f"Variable '{var}' not available - skipping graph")
                return None

        # Add time series variables if available
        if time_series_data and len(self.vars.get("time_series", [])) > 0:
            time_series_vars = self.vars["time_series"]

            # Get wells for current timestep (use timestep+1 for next state, as done with WCID)
            current_wells = (
                wells_data[timestep_idx + 1]
                if timestep_idx + 1 < len(wells_data)
                else {}
            )

            if current_wells:
                # Create time series arrays for this timestep
                ts_arrays = self._create_time_series_arrays(
                    grid,
                    current_wells,
                    time_series_data,
                    time_series_vars,
                    timestep_idx + 1,
                )

                # Process each time series variable
                # Note: Pressure variables (BHP, THP) will create _INJ and _PRD variants
                for var_name in time_series_vars:
                    var_upper = var_name.upper()
                    is_pressure_var = ("BHP" in var_upper) or ("THP" in var_upper)

                    if is_pressure_var:
                        # Pressure variables create two channels: _INJ and _PRD
                        channel_names = [f"{var_name}_INJ", f"{var_name}_PRD"]
                    else:
                        # Other variables create single channel
                        channel_names = [var_name]

                    # Process each channel (1 for non-pressure vars, 2 for pressure vars)
                    for channel_name in channel_names:
                        if channel_name not in ts_arrays:
                            continue

                        try:
                            # For time series, we can also include history
                            var_data_list = []
                            for t in range(self.prev_timestep_idx):
                                prev_ts_idx = (timestep_idx + 1) - t
                                if prev_ts_idx < 0:
                                    logger.error(
                                        f"Time series '{channel_name}' not available at timestep {prev_ts_idx} - skipping graph"
                                    )
                                    return None

                                # Create array for this historical timestep
                                prev_wells = (
                                    wells_data[prev_ts_idx]
                                    if prev_ts_idx < len(wells_data)
                                    else {}
                                )
                                if not prev_wells:
                                    logger.error(
                                        f"No wells data at timestep {prev_ts_idx} - skipping graph"
                                    )
                                    return None

                                prev_ts_arrays = self._create_time_series_arrays(
                                    grid,
                                    prev_wells,
                                    time_series_data,
                                    time_series_vars,
                                    prev_ts_idx,
                                )

                                if channel_name not in prev_ts_arrays:
                                    var_data_list.append(
                                        np.zeros(grid.nact, dtype=np.float32)
                                    )
                                else:
                                    var_data_list.append(prev_ts_arrays[channel_name])

                            # Stack historical data
                            var_data_stacked = np.stack(var_data_list, axis=0)
                            var_data_reshaped = (
                                var_data_stacked.T
                            )  # [n_active, prev_timestep_idx]

                            # Apply nonlinear scaling if specified (use original var_name for config lookup)
                            var_data_reshaped = self._apply_nonlinear_scaling(
                                var_data_reshaped, var_name
                            )

                            input_tensors.append(
                                torch.tensor(var_data_reshaped, dtype=torch.float32)
                            )
                            # Add individual names for each timestep: current, prev_1, prev_2, ...
                            for t in range(self.prev_timestep_idx):
                                if t == 0:
                                    input_var_names.append(f"{channel_name}_current")
                                else:
                                    input_var_names.append(f"{channel_name}_prev_{t}")

                        except Exception as e:
                            logger.error(
                                f"Failed to process time series variable '{channel_name}' - {type(e).__name__}: {e}"
                            )
                            return None

        # Concatenate all input features (will add temporal features if enabled)
        node_features = torch.cat(input_tensors, dim=1)

        # Extract target variables (next timestep)
        target_timestep = timestep_idx + 1
        target_tensors = []
        target_var_names = []

        for var in self.output_vars:
            if var in data and target_timestep < len(data[var]):
                try:
                    target_data = np.asarray(
                        data[var][target_timestep], dtype=np.float32
                    )
                    target_data = self._apply_nonlinear_scaling(target_data, var)

                    target_tensors.append(
                        torch.tensor(target_data, dtype=torch.float32).unsqueeze(1)
                    )
                    target_var_names.append(var)
                except Exception as e:
                    logger.error(
                        f"Failed to process target variable '{var}' at timestep {target_timestep} for sample {sample_idx} - {type(e).__name__}: {e}"
                    )
                    return None
            else:
                logger.error(
                    f"Target variable '{var}' not available at timestep {target_timestep} for sample {sample_idx}"
                )
                # If target is not available, return None to skip this graph
                return None

        # Concatenate target features
        target = torch.cat(target_tensors, dim=1)

        # Add temporal features to node features if enabled in config
        # This is the standard approach for incorporating time info in autoregressive GNNs
        n_nodes = node_features.shape[0]
        temporal_feature_names = []

        if self.global_vars.get("delta_t", False) or self.global_vars.get(
            "time", False
        ):
            # Validate that we have TIME data available
            if "TIME" not in data or len(data["TIME"]) == 0:
                logger.error(
                    f"No TIME data available in restart file for sample {sample_idx}"
                )
                return None

            if target_timestep >= len(data["TIME"]):
                logger.error(
                    f"Target timestep {target_timestep} >= available timesteps {len(data['TIME'])} for sample {sample_idx}"
                )
                return None

            if timestep_idx >= len(data["TIME"]):
                logger.error(
                    f"Current timestep {timestep_idx} >= available timesteps {len(data['TIME'])} for sample {sample_idx}"
                )
                return None

            try:
                # Calculate actual delta_t from TIME array
                delta_t = data["TIME"][target_timestep] - data["TIME"][timestep_idx]
            except Exception as e:
                logger.error(
                    f"Failed to calculate delta_t for sample {sample_idx} - {type(e).__name__}: {e}"
                )
                return None

            # Add delta_t as node feature (broadcast to all nodes)
            if self.global_vars.get("delta_t", False):
                delta_t_feature = torch.full((n_nodes, 1), delta_t, dtype=torch.float32)
                node_features = torch.cat([node_features, delta_t_feature], dim=1)
                temporal_feature_names.append("delta_t")
                input_var_names.append("delta_t")

            # Add normalized time as node feature (broadcast to all nodes)
            if self.global_vars.get("time", False):
                total_time = data["TIME"][-1] if len(data["TIME"]) > 0 else 1.0
                time_normalized = data["TIME"][timestep_idx] / max(total_time, 1.0)
                time_feature = torch.full(
                    (n_nodes, 1), time_normalized, dtype=torch.float32
                )
                node_features = torch.cat([node_features, time_feature], dim=1)
                temporal_feature_names.append("time")
                input_var_names.append("time")

        # Keep global_features for backward compatibility and metadata (not used by model)
        global_features = None

        try:
            # Build graph data dictionary dynamically
            graph_data = {
                "x": node_features,
                "edge_index": torch.tensor(edge_index, dtype=torch.long),
                "edge_attr": torch.tensor(edge_features, dtype=torch.float32),
                "y": target,
                "grid_info": {
                    "nx": grid.nx,
                    "ny": grid.ny,
                    "nz": grid.nz,
                    "total_cells": grid.nn,
                    "active_cells": grid.nact,
                    "sample_idx": sample_idx,
                    "timestep_idx": timestep_idx,
                    "target_timestep": target_timestep,
                    "input_vars": input_var_names,
                    "target_vars": target_var_names,
                },
                "case_name": case_name or f"sample_{sample_idx:03d}",
                "timestep_id": timestep_idx,
            }

            # Add global_features only if populated (for backward compatibility)
            if global_features is not None:
                graph_data["global_features"] = global_features

            # Add coordinates only if they're not already in node features
            if coordinates is not None:
                graph_data["coordinates"] = torch.tensor(
                    coordinates, dtype=torch.float32
                )
        except Exception as e:
            logger.error(
                f"Failed to create graph data dictionary for sample {sample_idx} - {type(e).__name__}: {e}"
            )
            return None

        try:
            # Create graph
            graph = Data(**graph_data)
            return graph
        except Exception as e:
            logger.error(
                f"Failed to create PyTorch Geometric Data object for sample {sample_idx} - {type(e).__name__}: {e}"
            )
            return None

    def _prepare_data_keys(self):
        """Prepare the keys needed for reading simulation data."""
        init_keys = list(
            dict.fromkeys(
                self.vars["grid"]["static"]
                + ["INTEHEAD", "PORV", "TRANX", "TRANY", "TRANZ", "TRANNNC"]
            )
        )

        # EGRID keys (skip expensive ones if coordinates aren't needed)
        egrid_keys_geometry = ["COORD", "ZCORN", "FILEHEAD"]
        if not self.include_coordinates_as_features:
            egrid_keys_geometry = [
                k for k in egrid_keys_geometry if k not in ("COORD", "ZCORN")
            ]
        egrid_keys_nnc = ["NNC1", "NNC2"]

        rst_well_keys = ["INTEHEAD", "ZWEL", "IWEL", "ICON", "SCON"]

        return init_keys, (egrid_keys_geometry, egrid_keys_nnc), rst_well_keys

    def _prepare_dynamic_variables(self):
        """Prepare dynamic variables and completion requirements."""
        dyn_vars = self.vars.get("grid", {}).get("dynamic", []) or []
        include_well_completion_ids = "WCID" in dyn_vars
        include_well_completion_cf = "WCCF" in dyn_vars
        if include_well_completion_ids:
            dyn_vars.remove("WCID")
        if include_well_completion_cf:
            dyn_vars.remove("WCCF")
        include_well_completions = (
            include_well_completion_ids or include_well_completion_cf
        )

        # Get time series variables
        time_series_vars = self.vars.get("time_series", []) or []
        include_time_series = len(time_series_vars) > 0

        return (
            dyn_vars,
            include_well_completion_ids,
            include_well_completion_cf,
            include_well_completions,
            time_series_vars,
            include_time_series,
        )

    def _process_static_data(self, reader, init_keys, egrid_data, sample_idx_1based):
        """Process static grid data and validate it."""
        init_data = reader.read_init(init_keys)

        # Add coordinates if requested
        if self.include_coordinates_as_features:
            grid = Grid(init_data, egrid_data)
            for key in self.requested_coordinates:
                init_data[key] = getattr(grid, key)
        else:
            grid = Grid(init_data, egrid_data)

        # Filter full-grid keys to active cells only
        for key in Grid.FULL_GRID_KEYS:
            if key in init_data and len(init_data[key]) == grid.nn:
                init_data[key] = init_data[key][grid.actnum_bool]

        # Validate static data
        for key in self.vars["grid"]["static"]:
            if len(init_data[key]) == 0:
                raise ValueError(
                    f"  Error: Failed to read {key} from init/egrid file for sample {sample_idx_1based}"
                )

        return init_data, grid

    def _process_dynamic_data(
        self,
        reader,
        grid,
        dynamic_variables,
        rst_well_keys,
        include_well_completions,
        include_well_completion_cf,
        sample_idx_1based,
    ):
        """Process dynamic data including wells and completion arrays."""
        wells_data = []
        rst_data = {}

        if not dynamic_variables:
            return wells_data, rst_data

        try:
            rst_well_data = reader.read_restart(rst_well_keys)
            wells_data = self.get_completion_info(grid, rst_well_data)
            rst_data = reader.read_restart(dynamic_variables)

            # Handle completion arrays if needed
            if wells_data and include_well_completions:
                try:
                    completion_arrays_inj, completion_arrays_prd = [], []
                    for wells in wells_data:
                        cmpl_inj, cmpl_prd = grid.create_completion_array(
                            wells, include_well_completion_cf
                        )
                        completion_arrays_inj.append(cmpl_inj)
                        completion_arrays_prd.append(cmpl_prd)
                    # use states from next time step
                    rst_data["WCID_INJ"] = completion_arrays_inj[1:]
                    rst_data["WCID_PRD"] = completion_arrays_prd[1:]
                except Exception as e:
                    self._log_error_and_continue(
                        f"Failed to create completion arrays for sample {sample_idx_1based} - {type(e).__name__}: {e}"
                    )
                    return None, None

            # Validate dynamic data
            for key in dynamic_variables:
                if key not in rst_data or not rst_data[key]:
                    logger.warning(
                        f"Failed to read {key} from restart file for sample {sample_idx_1based}"
                    )
                    rst_data[key] = []

        except Exception as e:
            self._log_error_and_continue(
                f"Failed to read restart data for sample {sample_idx_1based} - {type(e).__name__}: {e}"
            )
            return None, None

        return wells_data, rst_data

    def _read_and_interpolate_time_series(
        self,
        reader,
        time_series_vars,
        restart_times,
        sample_idx_1based,
        wells_data=None,
    ):
        """Read summary data and interpolate to match restart timesteps.

        Parameters
            reader: EclReader instance
            time_series_vars: List of time series variable names (e.g., ["WWIR", "WGIR", "WBHP"])
            restart_times: Array of restart file timesteps (in days)
            sample_idx_1based: Sample index for error messages
            wells_data: List of well dictionaries (one per timestep) for checking well status

        Returns
            dict: Interpolated time series data structured as:
                  {well_name: {var_name: [val_t0, val_t1, ..., val_tn]}}
        """
        if not time_series_vars or len(restart_times) == 0:
            return {}

        try:
            # Read summary data for all entities (wells)
            smry_data = reader.read_smry(keys=time_series_vars, entities=None)

            if "TIME" not in smry_data:
                logger.warning(
                    f"No TIME data in summary file for sample {sample_idx_1based}"
                )
                return {}

            smry_times = smry_data["TIME"]

            # Check if restart times are within summary time range (only warn once)
            if restart_times[0] < smry_times[0] or restart_times[-1] > smry_times[-1]:
                # Store the warning info but don't print yet (will be collected and summarized)
                if not hasattr(self, "_time_range_warnings"):
                    self._time_range_warnings = []
                self._time_range_warnings.append(
                    {
                        "sample": sample_idx_1based,
                        "restart_range": (restart_times[0], restart_times[-1]),
                        "summary_range": (smry_times[0], smry_times[-1]),
                    }
                )

            # Interpolate time series data for each well and variable
            interpolated_data = {}

            for entity, entity_data in smry_data.items():
                if entity == "TIME":
                    continue

                if not isinstance(entity_data, dict):
                    continue

                interpolated_data[entity] = {}

                for var_name, var_values in entity_data.items():
                    if var_name not in time_series_vars:
                        continue

                    # Create interpolation function
                    # Use linear interpolation with bounds_error=False to extrapolate if needed
                    interp_func = interp1d(
                        smry_times,
                        var_values,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(
                            var_values[0],
                            var_values[-1],
                        ),  # Use edge values for extrapolation
                    )

                    # Interpolate to restart timesteps
                    interpolated_values = interp_func(restart_times)

                    # Check well status for times before summary start
                    # Use 0.0 for wells that are SHUT or don't exist yet (not drilled)
                    if wells_data is not None and restart_times[0] < smry_times[0]:
                        for t_idx, restart_time in enumerate(restart_times):
                            # Only check times before summary data starts
                            if restart_time >= smry_times[0]:
                                break

                            # Check if well exists and is open at this timestep
                            if t_idx < len(wells_data):
                                if entity not in wells_data[t_idx]:
                                    # Well doesn't exist yet (not drilled/completed), use 0.0
                                    interpolated_values[t_idx] = 0.0
                                else:
                                    well = wells_data[t_idx][entity]
                                    if (
                                        hasattr(well, "status")
                                        and well.status == "SHUT"
                                    ):
                                        # Well exists but is shut, use 0.0 instead of first summary value
                                        interpolated_values[t_idx] = 0.0
                                    # else: well is OPEN, keep interpolated value (first summary value)

                    interpolated_data[entity][var_name] = interpolated_values

            return interpolated_data

        except Exception as e:
            logger.warning(
                f"Failed to read/interpolate time series for sample {sample_idx_1based}: {type(e).__name__}: {e}"
            )
            return {}

    def _create_time_series_arrays(
        self, grid, wells, time_series_data, time_series_vars, timestep_idx
    ):
        """Create time series arrays mapped to grid cells with well completions.

        Similar to completion ID arrays, this creates one array per time series variable,
        where values are assigned to grid cells containing well completions.

        For pressure variables (e.g., WBHP, WTHP - anything with BHP or THP in name),
        creates separate channels for injection and production wells
        (e.g., WBHP_INJ, WBHP_PRD).

        Parameters
            grid: Grid object
            wells: Dictionary of Well objects for current timestep
            time_series_data: Interpolated time series data
                             {well_name: {var_name: [val_t0, val_t1, ...]}}
            time_series_vars: List of time series variable names
            timestep_idx: Current timestep index

        Returns
            dict: {var_name: np.ndarray} where arrays have shape (n_active_cells,)
                  For pressure variables, returns {var_name_INJ: array, var_name_PRD: array}
        """
        if not time_series_data or not wells:
            return {}

        result = {}

        for var_name in time_series_vars:
            # Check if this is a pressure-related variable (bottom-hole pressure or tubing head pressure)
            # These should be split into injection and production channels
            var_upper = var_name.upper()
            is_pressure_var = ("BHP" in var_upper) or ("THP" in var_upper)

            if is_pressure_var:
                # Create separate arrays for injection and production wells
                var_array_inj = np.zeros(grid.nact, dtype=np.float32)
                var_array_prd = np.zeros(grid.nact, dtype=np.float32)
            else:
                # Single array for non-pressure variables
                var_array = np.zeros(grid.nact, dtype=np.float32)

            # Iterate through wells and assign values to completion cells
            for well_name, well in wells.items():
                if well_name not in time_series_data:
                    continue

                if var_name not in time_series_data[well_name]:
                    continue

                # Skip shut wells - assign zeros (default array value)
                if well.status == "SHUT":
                    continue

                # Get interpolated value at this timestep
                var_values = time_series_data[well_name][var_name]
                if timestep_idx >= len(var_values):
                    continue

                value = var_values[timestep_idx]

                # Determine well type from Well object
                # well.type is "INJ" or "PRD" (set by _set_type method)
                is_injector = well.type == "INJ"

                # Assign value to all completion cells for this well
                for comp in well.completions:
                    # Check if IJK attribute exists and is valid
                    if hasattr(comp, "IJK") and comp.IJK is not None:
                        # Convert from 1-based to 0-based indexing, then map to active-only index
                        # (same logic as WCID in grid.create_completion_array)
                        ijk = comp.IJK - 1  # Convert to 0-based
                        if ijk in grid.ijk_to_active:
                            active_idx = grid.ijk_to_active[ijk]
                            if is_pressure_var:
                                # Assign to injection or production array based on well type
                                if is_injector:
                                    var_array_inj[active_idx] = value
                                else:
                                    var_array_prd[active_idx] = value
                            else:
                                # Single array for non-pressure variables
                                var_array[active_idx] = value

            # Store results
            if is_pressure_var:
                result[f"{var_name}_INJ"] = var_array_inj
                result[f"{var_name}_PRD"] = var_array_prd
            else:
                result[var_name] = var_array

        return result

    def _validate_timesteps(self, combined_data, dynamic_variables, sample_idx_1based):
        """Validate timesteps and return valid ones."""
        times = combined_data.get("TIME") or []
        if len(times) < 2:
            self._log_error_and_continue(
                f"Sample {sample_idx_1based} has only {len(times)} timestep(s), need at least 2 for current->target prediction"
            )
            return []

        # Find valid timesteps
        valid_timesteps = []
        max_t = len(times) - 1  # because we use t+1 as target

        for t in range(max_t):
            if self._is_timestep_valid(combined_data, dynamic_variables, t):
                valid_timesteps.append(t)

        if not valid_timesteps:
            self._log_error_and_continue(
                f"No valid timesteps found for sample {sample_idx_1based} (simulation may have died early)"
            )
            return []

        return valid_timesteps

    def _is_timestep_valid(self, combined_data, dynamic_variables, t):
        """Check if a timestep has all required data."""
        # Check inputs (t)
        for var in dynamic_variables:
            if var not in combined_data or len(combined_data[var]) <= t:
                return False

        # Check targets (t+1)
        for var in self.output_vars:
            if var not in combined_data or len(combined_data[var]) <= (t + 1):
                return False

        return True

    def _log_error_and_continue(self, message, context=""):
        """Log error message and return indication to continue."""
        logger.error(f"{context}{message}")
        logger.info("Skipping and continuing with the next one...")
        return True

    def _build_graphs_for_sample(
        self,
        grid,
        wells_data,
        combined_data,
        valid_timesteps,
        sample_idx_1based,
        case_name,
        time_series_data=None,
    ):
        """Build graphs for all valid timesteps of a sample."""
        graphs = []
        total_possible = len(combined_data.get("TIME", [])) - 1

        for t in valid_timesteps:
            try:
                graph = self.build_graph_from_simulation_data(
                    grid,
                    wells_data,
                    combined_data,
                    sample_idx=sample_idx_1based - 1,
                    timestep_idx=t,
                    case_name=case_name,
                    time_series_data=time_series_data,
                )
                if graph is None:
                    # Only log as error if timestep should have had enough history
                    if t + 1 >= self.prev_timestep_idx:
                        self._log_error_and_continue(
                            f"Graph creation returned None for timestep {t} (missing data or simulation died)",
                            "  ",
                        )
                    # Otherwise skip silently (expected for early timesteps)
                    continue

                graphs.append(graph)

            except Exception as e:
                self._log_error_and_continue(
                    f"Graph creation failed for timestep {t} - {type(e).__name__}: {e}",
                    "  ",
                )
                continue

        return graphs

    def _process_single_sample_worker(
        self,
        file_path,
        sample_idx_1based,
        total_samples,
        egrid_keys_geometry,
        egrid_keys_nnc,
        init_keys,
        dynamic_variables,
        rst_well_keys,
        include_well_completions,
        include_well_completion_cf,
        time_series_vars,
        include_time_series,
        output_path_graph,
        progress_counter,
    ):
        """
        Process a single sample, save graphs immediately, and return filenames.

        This method is designed to be called in parallel by multiprocessing workers.
        Returns (saved_filenames, case_name, error_msg) tuple.
        """
        case_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            reader = EclReader(file_path)

            # Read EGRID data (each worker reads independently - simpler than sharing)
            egrid_data_geometry = reader.read_egrid(egrid_keys_geometry)
            egrid_data_nnc = reader.read_egrid(egrid_keys_nnc)
            egrid_data = {**egrid_data_geometry, **egrid_data_nnc}

            # Process static grid data
            init_data, grid = self._process_static_data(
                reader, init_keys, egrid_data, sample_idx_1based
            )

            # Process dynamic data
            wells_data, restart_data = self._process_dynamic_data(
                reader,
                grid,
                dynamic_variables,
                rst_well_keys,
                include_well_completions,
                include_well_completion_cf,
                sample_idx_1based,
            )

            if wells_data is None:  # Error occurred
                return None, case_name, "Error in processing dynamic data"

            # Combine static + dynamic data
            combined_data = {**init_data, **restart_data}

            # Read and interpolate time series data if needed
            time_series_data = None
            if include_time_series and time_series_vars:
                restart_times = np.array(combined_data.get("TIME", []))
                if len(restart_times) > 0:
                    time_series_data = self._read_and_interpolate_time_series(
                        reader,
                        time_series_vars,
                        restart_times,
                        sample_idx_1based,
                        wells_data,
                    )

            # Validate timesteps
            valid_timesteps = self._validate_timesteps(
                combined_data, dynamic_variables, sample_idx_1based
            )

            if not valid_timesteps:
                return None, case_name, "No valid timesteps found"

            # Build graphs for all valid timesteps
            sample_graphs = self._build_graphs_for_sample(
                grid,
                wells_data,
                combined_data,
                valid_timesteps,
                sample_idx_1based,
                case_name,
                time_series_data=time_series_data,
            )

            # Save graphs immediately (memory efficient)
            saved_filenames = []
            for graph in sample_graphs:
                timestep_id = getattr(graph, "timestep_id", 0)
                filename = f"{case_name}_{timestep_id:03d}.pt"
                graph_path = os.path.join(output_path_graph, filename)
                torch.save(graph, graph_path)
                saved_filenames.append(filename)

            # Increment progress counter (Manager.Value is automatically thread-safe)
            progress_counter.value += 1

            return saved_filenames, case_name, None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            # Increment progress counter even on error
            progress_counter.value += 1
            return None, case_name, error_msg

    def _parse_results_from_samples(self) -> dict:
        """
        Parse simulation results and create graphs from all samples.

        This is the main orchestration method that:
        1. Finds and validates input files
        2. Processes each sample's static and dynamic data
        3. Validates timesteps and builds graphs
        4. Returns all successfully created graphs
        """
        # === INITIALIZATION ===
        sim_input_files = self._find_primary_input_files()
        if not sim_input_files:
            file_type = ".AFI" if self.simulator == "IX" else ".DATA"
            raise RuntimeError(
                f"No {file_type} files found in {self.sim_dir}. Check the path and file naming."
            )

        # Prepare data keys and variables for reading simulation files
        init_keys, egrid_keys, rst_well_keys = self._prepare_data_keys()
        egrid_keys_geometry, egrid_keys_nnc = egrid_keys

        (
            dynamic_variables,
            include_well_completion_ids,
            include_well_completion_cf,
            include_well_completions,
            time_series_vars,
            include_time_series,
        ) = self._prepare_dynamic_variables()

        all_graph_files = []
        failed_sample_count = 0

        # Limit samples if specified
        if self.num_samples is not None:
            sim_input_files = sim_input_files[: self.num_samples]

        total_samples = len(sim_input_files)

        # Determine number of workers
        n_workers = min(self.num_preprocess_workers, cpu_count(), total_samples)

        logger.info(
            f"Processing {total_samples} simulation results using {n_workers} parallel workers..."
        )

        start_time = time.time()

        with Manager() as manager:
            # Create shared progress counter
            progress_counter = manager.Value("i", 0)

            # Prepare arguments for parallel processing
            worker_args = [
                (
                    file_path,
                    sample_idx_1based,
                    total_samples,
                    egrid_keys_geometry,
                    egrid_keys_nnc,
                    init_keys,
                    dynamic_variables,
                    rst_well_keys,
                    include_well_completions,
                    include_well_completion_cf,
                    time_series_vars,
                    include_time_series,
                    self._output_path_graph,
                    progress_counter,
                )
                for sample_idx_1based, file_path in enumerate(sim_input_files, start=1)
            ]

            with Pool(processes=n_workers) as pool:
                # Start async processing
                async_result = pool.starmap_async(
                    self._process_single_sample_worker, worker_args
                )

                # Print progress every 30 seconds
                while not async_result.ready():
                    async_result.wait(timeout=30.0)
                    if not async_result.ready():
                        completed = progress_counter.value
                        elapsed = time.time() - start_time
                        logger.info(
                            f"... {completed}/{total_samples} samples completed (elapsed: {elapsed:.0f}s) ..."
                        )

                results = async_result.get()

        elapsed = time.time() - start_time
        logger.info(
            f"Completed in {elapsed:.1f}s ({total_samples / elapsed:.1f} samples/s)"
        )

        # Collect results and handle errors
        logger.info("Collecting results...")
        for sample_idx, (saved_filenames, case_name, error_msg) in enumerate(
            results, start=1
        ):
            if saved_filenames is None:
                failed_sample_count += 1
                self._log_error_and_continue(
                    f"Processing sample {sample_idx} ({case_name}): {error_msg}"
                )
                if failed_sample_count > 0.2 * total_samples:
                    raise RuntimeError("Failed to process too many samples.")
            else:
                all_graph_files.extend(saved_filenames)

        # === FINAL SUMMARY ===
        total_samples = len(sim_input_files)
        avg_graphs_per_sample = (
            (len(all_graph_files) / total_samples) if total_samples else 0.0
        )
        logger.info("Processing Summary:")
        logger.info(f"  Total samples processed: {total_samples}")
        logger.info(f"  Total graphs created: {len(all_graph_files)}")
        logger.info(f"  Average graphs per sample: {avg_graphs_per_sample:.1f}")
        logger.info(f"  Graphs saved to: {self._output_path_graph}")

        return all_graph_files

    def _completion_to_dict(self, completion):
        """
        Convert a Completion object to a JSON-serializable dictionary.

        Parameters
            completion: Completion object

        Returns
            dict: Dictionary representation of the completion
        """
        comp_dict = {
            "I": completion.I,
            "J": completion.J,
            "K": completion.K,
            "dir": completion.dir,
            "status": completion.status,
            "connection_factor": completion.connection_factor,
        }

        # Add optional attributes if they exist
        if hasattr(completion, "IJK"):
            comp_dict["IJK"] = (
                int(completion.IJK) if completion.IJK is not None else None
            )
        if hasattr(completion, "flow_rate"):
            comp_dict["flow_rate"] = float(completion.flow_rate)

        return comp_dict

    def _well_to_dict(self, well):
        """
        Convert a Well object to a JSON-serializable dictionary.

        Parameters
            well: Well object

        Returns
            dict: Dictionary representation of the well
        """
        return {
            "name": well.name,
            "type": well.type,
            "status": well.status,
            "num_active_completions": well.num_active_completions,
            "completions": [
                self._completion_to_dict(comp) for comp in well.completions
            ],
        }

    def _save_well_list_json(self, wells_data):
        """
        Save wells_data (list of lists of dicts of Well objects) to a JSON file.

        Parameters
            wells_data: List of lists of dictionaries of Well objects
        """
        # Convert Wells objects to JSON-serializable format
        json_data = []
        for timestep_wells in wells_data:
            if isinstance(timestep_wells, dict):
                # Dictionary of well_name: Well object
                timestep_dict = {
                    well_name: self._well_to_dict(well)
                    for well_name, well in timestep_wells.items()
                }
            elif isinstance(timestep_wells, list):
                # List is empty or contains Wells
                if len(timestep_wells) == 0:
                    timestep_dict = {}
                else:
                    timestep_dict = [
                        self._well_to_dict(well) for well in timestep_wells
                    ]
            else:
                timestep_dict = {}

            json_data.append(timestep_dict)

        with open(self._output_path_well, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Well data saved to {self._output_path_well}")

    def execute(self):
        # Process samples and save graphs (returns filenames)
        generated_files = self._parse_results_from_samples()

        logger.info(f"Processed {len(generated_files)} graphs successfully!")

        return generated_files

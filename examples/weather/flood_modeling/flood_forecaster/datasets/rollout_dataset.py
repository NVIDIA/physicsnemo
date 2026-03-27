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

r"""
Rollout test dataset for flood prediction.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FloodRolloutTestDatasetNew(Dataset):
    r"""
    Dataset for rollout evaluation with channel order [WD, VX, VY].
    """

    def __init__(
            self,
            rollout_data_root,
            n_history,
            rollout_length,
            xy_file=None,
            query_res=None,
            static_files=None,
            dynamic_patterns=None,
            boundary_patterns=None,
            raise_on_smaller=True,
            skip_before_timestep=0
    ):
        r"""
        Initialize rollout test dataset.

        Parameters
        ----------
        rollout_data_root : str or Path
            Root directory containing rollout test data.
        n_history : int
            Number of history timesteps.
        rollout_length : int
            Length of rollout to evaluate.
        xy_file : str, optional
            Filename for XY coordinates.
        query_res : List[int], optional, default=[64, 64]
            Query resolution.
        static_files : List[str], optional
            List of static feature filenames.
        dynamic_patterns : Dict[str, str], optional
            Dict mapping variable names to filename patterns.
        boundary_patterns : Dict[str, str], optional
            Dict mapping boundary names to filename patterns.
        raise_on_smaller : bool, optional, default=True
            Whether to raise error if data is smaller than expected.
        skip_before_timestep : int, optional, default=0
            Number of timesteps to skip at the beginning.

        Raises
        ------
        FileNotFoundError
            If data root or required files are not found.
        ValueError
            If no valid run IDs are found.
        """
        super().__init__()
        self.data_root = Path(rollout_data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Rollout data root not found: {self.data_root}")
        self.n_history = n_history
        self.rollout_length = rollout_length
        self.xy_file = xy_file
        self.query_res = query_res if query_res else [64, 64]
        self.static_files = static_files if static_files else []
        self.dynamic_patterns = dynamic_patterns if dynamic_patterns else {}
        self.boundary_patterns = boundary_patterns if boundary_patterns else {}
        self.raise_on_smaller = raise_on_smaller
        self.skip_before_timestep = skip_before_timestep

        # Read run IDs from test.txt
        test_txt = self.data_root / "test.txt"
        if not test_txt.exists():
            raise FileNotFoundError(f"Expected test.txt at {test_txt}, not found!")
        try:
            with open(test_txt, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except IOError as e:
            raise IOError(f"Failed to read test.txt from {test_txt}: {e}") from e
        if len(lines) == 1 and "," in lines[0]:
            self.run_ids = lines[0].split(",")
        else:
            self.run_ids = lines
        self.run_ids = [rid.strip() for rid in self.run_ids if rid.strip()]
        if not self.run_ids:
            raise ValueError("No valid run IDs found in test.txt")

        self.xy_coords = None
        self.static_data = None
        self.cell_area = None  # For volume conservation
        self.dynamic_data = {}
        self.boundary_data = {}

        self.reference_cell_count = self._load_xy_file()
        self._load_static()
        self._load_all_runs()

        # Filter out runs that lack enough time steps
        self.valid_run_ids = []
        for run_id in self.run_ids:
            missing_vars = [var for var in ["WD", "VX", "VY"] if var not in self.dynamic_data[run_id]]
            missing_bc = [var for var in ["inflow"] if var not in self.boundary_data[run_id]]
            if missing_vars or missing_bc:
                warnings.warn(f"Run ID {run_id} missing variables: {missing_vars}, bc: {missing_bc}. Skipping.")
                continue
            T = self.dynamic_data[run_id]["WD"].shape[0]
            if T >= self.skip_before_timestep + self.n_history + self.rollout_length:
                self.valid_run_ids.append(run_id)

        if not self.valid_run_ids:
            raise ValueError("No hydrographs have enough time steps for rollout evaluation.")

        # Store the geometry/static from the first valid sample
        # Safe to call __getitem__ here because all data has been loaded in _load_all_runs()
        # and valid_run_ids has been populated, ensuring the dataset is fully initialized
        if len(self.valid_run_ids) > 0:
            sample0 = self.__getitem__(0)
            self.geometry = sample0["geometry"]
            self.static = sample0["static"]
        else:
            # This should never happen due to the check above, but defensive programming
            raise RuntimeError("Cannot initialize geometry/static: no valid runs available")

    def _load_xy_file(self):
        r"""Load and normalize XY coordinates."""
        if not self.xy_file:
            raise ValueError("xy_file was not provided for rollout dataset! Please specify in config.")
        xy_path = self.data_root / self.xy_file
        if not xy_path.exists():
            raise FileNotFoundError(f"Rollout XY file not found: {xy_path}")

        # Load raw coordinates
        xy_arr = np.loadtxt(str(xy_path), delimiter="\t", dtype=np.float32)
        if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
            raise ValueError(f"{self.xy_file} must be shape (num_cells,2). Got {xy_arr.shape}.")

        # Unit-box normalization
        min_xy = xy_arr.min(axis=0)
        max_xy = xy_arr.max(axis=0)
        range_xy = max_xy - min_xy
        range_xy[range_xy == 0] = 1.0
        xy_arr = (xy_arr - min_xy) / range_xy

        self.xy_coords = torch.tensor(xy_arr, device='cpu')
        return self.xy_coords.shape[0]

    def _load_static(self):
        r"""Load static feature files, including cell area."""
        if not self.static_files:
            self.static_data = torch.zeros((self.reference_cell_count, 0), device='cpu')
            return
        static_list = []
        for fname in self.static_files:
            fpath = self.data_root / fname
            if not fpath.exists():
                warnings.warn(f"Static file not found in rollout folder: {fpath}, skipping.")
                continue
            arr = np.loadtxt(str(fpath), delimiter="\t", dtype=np.float32)

            if arr.ndim == 1:
                arr = arr[:, None]
            n_file = arr.shape[0]
            if n_file < self.reference_cell_count:
                msg = f"Static {fname} has {n_file} < {self.reference_cell_count}"
                if self.raise_on_smaller:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg + " -> skipping.")
                    continue
            elif n_file > self.reference_cell_count:
                arr = arr[:self.reference_cell_count, :]
            
            # Capture cell area AFTER trimming to match reference cell count
            if "M40_CA.txt" in str(fname):
                self.cell_area = torch.from_numpy(arr.flatten()).float()
            
            static_list.append(arr)

        if self.cell_area is None:
            warnings.warn(
                "Cell Area file ('M40_CA.txt') not found in static_files. Volume conservation cannot be calculated.")

        if not static_list:
            self.static_data = torch.zeros((self.reference_cell_count, 0), device='cpu')
            return

        combined_arr = np.concatenate(static_list, axis=1)
        self.static_data = torch.tensor(combined_arr, device='cpu')

    def _load_all_runs(self):
        r"""Load dynamic and boundary data for all runs."""
        for run_id in tqdm(self.run_ids, desc="Loading runs for rollout evaluation"):
            self.dynamic_data[run_id] = {}
            self.boundary_data[run_id] = {}
            # dynamic
            for dkey, pattern in self.dynamic_patterns.items():
                fname = pattern.format(run_id)
                fpath = self.data_root / fname
                if not fpath.exists():
                    warnings.warn(f"Dynamic file not found: {fpath}, skipping {dkey}.")
                    continue
                arr = np.loadtxt(str(fpath), delimiter="\t", dtype=np.float32)
                N_file = arr.shape[1]
                if N_file < self.reference_cell_count:
                    msg = f"{fname} has {N_file} < {self.reference_cell_count}"
                    if self.raise_on_smaller:
                        raise ValueError(msg)
                    else:
                        warnings.warn(msg + " -> skipping.")
                        continue
                elif N_file > self.reference_cell_count:
                    arr = arr[:, :self.reference_cell_count]
                self.dynamic_data[run_id][dkey] = torch.tensor(arr, device='cpu')
            # boundary
            for bc_key, bc_pattern in self.boundary_patterns.items():
                fname = bc_pattern.format(run_id)
                fpath = self.data_root / fname
                if not fpath.exists():
                    warnings.warn(f"Boundary file not found: {fpath}, skipping {bc_key}.")
                    continue
                bc_arr = np.loadtxt(str(fpath), delimiter="\t", dtype=np.float32)
                if bc_arr.ndim == 1:
                    bc_arr = bc_arr[:, None]
                if bc_arr.shape[1] == 2:
                    bc_arr = bc_arr[:, 1].reshape(-1, 1)
                bc_tensor = torch.tensor(bc_arr, device='cpu')
                bc_tensor = bc_tensor.expand(-1, self.reference_cell_count)
                bc_tensor = bc_tensor.unsqueeze(-1)
                self.boundary_data[run_id][bc_key] = bc_tensor

    def __len__(self):
        return len(self.valid_run_ids)

    def __getitem__(self, idx):
        r"""Get a single rollout sample."""
        run_id = self.valid_run_ids[idx]
        dynamic_vars = ["WD", "VX", "VY"]
        dynamic = torch.stack([self.dynamic_data[run_id][var] for var in dynamic_vars], dim=-1)
        boundary_keys = sorted(list(self.boundary_data[run_id].keys()))
        if boundary_keys:
            boundary = torch.cat([self.boundary_data[run_id][var] for var in boundary_keys], dim=-1)
        else:
            # Ensure device consistency: xy_coords is on CPU, match dynamic tensor device
            boundary_device = dynamic.device if isinstance(dynamic, torch.Tensor) else self.xy_coords.device
            boundary = torch.zeros((dynamic.shape[0], self.reference_cell_count, 1), device=boundary_device)

        sample = {
            "run_id": run_id,
            "dynamic": dynamic,
            "boundary": boundary,
            "geometry": self.xy_coords,
            "static": self.static_data,
        }
        if self.cell_area is not None:
            sample["cell_area"] = self.cell_area
        return sample


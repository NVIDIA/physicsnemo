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
Training dataset for flood prediction with query points.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FloodDatasetWithQueryPoints(Dataset):
    r"""
    Dataset for training/one-step testing with channel order [WD, VX, VY].
    
    Ensures dynamic history channels are [WD=0, VX=1, VY=2].

    Parameters
    ----------
    data_root : str or Path
        Root directory containing data files.
    n_history : int
        Number of history timesteps.
    query_res : List[int], optional, default=[64, 64]
        Query resolution.
    xy_file : str, optional
        Filename for XY coordinates.
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
    noise_type : str, optional, default="none"
        Type of noise to apply ("none", "only_last", "correlated", "uncorrelated", "random_walk").
    noise_std : List[float], optional
        List of 3 floats for noise std for [WD, VX, VY].

    Raises
    ------
    FileNotFoundError
        If data root or required files are not found.
    ValueError
        If noise_std length is not 3 or if no valid run IDs are found.
    """

    def __init__(
            self,
            data_root,
            n_history,
            query_res=None,
            xy_file=None,
            static_files=None,
            dynamic_patterns=None,
            boundary_patterns=None,
            raise_on_smaller=True,
            skip_before_timestep=0,
            noise_type="none",
            noise_std=None,
    ):
        r"""
        Initialize flood dataset with query points.
        """
        super().__init__()
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        self.n_history = n_history
        self.query_res = query_res if query_res else [64, 64]
        self.xy_file = xy_file
        self.static_files = static_files if static_files else []
        self.dynamic_patterns = dynamic_patterns if dynamic_patterns else {}
        self.boundary_patterns = boundary_patterns if boundary_patterns else {}
        self.raise_on_smaller = raise_on_smaller
        self.skip_before_timestep = skip_before_timestep

        # NOISE PARAMS
        if noise_std is None or (isinstance(noise_std, (list, tuple)) and len(noise_std) == 0):
            self.noise_type = "none"
            self.noise_std = [0.0, 0.0, 0.0]
        else:
            self.noise_type = noise_type.lower() if noise_type else "none"
            self.noise_std = noise_std
            if len(self.noise_std) != 3:
                raise ValueError("noise_std must be a list of exactly 3 floats for WD, VX, VY.")

        # Read run IDs from train.txt
        train_txt = self.data_root / "train.txt"
        if not train_txt.exists():
            raise FileNotFoundError(f"Expected train.txt at {train_txt}, not found!")
        try:
            with open(train_txt, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except IOError as e:
            raise IOError(f"Failed to read train.txt from {train_txt}: {e}") from e
        if len(lines) == 1 and "," in lines[0]:
            self.run_ids = lines[0].split(",")
        else:
            self.run_ids = lines
        self.run_ids = [rid.strip() for rid in self.run_ids if rid.strip()]
        if not self.run_ids:
            raise ValueError("No valid run IDs found in train.txt")

        # Internals
        self.xy_coords = None
        self.static_data = None
        self.dynamic_data = {}
        self.boundary_data = {}
        self.sample_index = []

        # Load data
        self.reference_cell_count = self._load_xy_file()
        self._load_static()
        self._load_all_runs()
        self._build_sample_indices()

    def _load_xy_file(self):
        r"""Load and normalize XY coordinates."""
        if not self.xy_file:
            raise ValueError("xy_file was not provided! Please specify in config.")
        xy_path = self.data_root / self.xy_file
        if not xy_path.exists():
            raise FileNotFoundError(f"Reference XY file not found: {xy_path}")

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
        r"""Load static feature files."""
        if not self.static_files:
            self.static_data = torch.zeros((self.reference_cell_count, 0), device='cpu')
            return
        static_list = []
        for fname in self.static_files:
            fpath = self.data_root / fname
            if not fpath.exists():
                warnings.warn(f"Static file not found: {fpath}, skipping.")
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
            static_list.append(arr)
        if not static_list:
            self.static_data = torch.zeros((self.reference_cell_count, 0), device='cpu')
            return
        combined_arr = np.concatenate(static_list, axis=1)
        self.static_data = torch.tensor(combined_arr, device='cpu')

    def _load_all_runs(self):
        r"""Load dynamic and boundary data for all runs."""
        for run_id in tqdm(self.run_ids, desc="Loading runs for training"):
            self.dynamic_data[run_id] = {}
            self.boundary_data[run_id] = {}
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

    def _build_sample_indices(self):
        r"""Build list of (run_id, timestep) indices for valid samples."""
        for run_id in self.run_ids:
            dyn_dict = self.dynamic_data[run_id]
            bc_dict = self.boundary_data[run_id]
            if not dyn_dict or not bc_dict:
                continue
            required_dyn = ["WD", "VX", "VY"]
            required_bc = ["inflow"]
            if not all(k in dyn_dict for k in required_dyn) or not all(k in bc_dict for k in required_bc):
                warnings.warn(f"Run ID {run_id} missing required dynamic/boundary variables. Skipping.")
                continue
            ref_tensor = dyn_dict["WD"]
            T = ref_tensor.shape[0]
            start_t = max(self.n_history, self.skip_before_timestep)
            for t in range(start_t, T):
                self.sample_index.append((run_id, t))

    def __len__(self):
        return len(self.sample_index)

    def _apply_noise(self, dynamic_hist: torch.Tensor):
        r"""
        Apply noise to dynamic history tensor.

        Parameters
        ----------
        dynamic_hist : torch.Tensor
            Dynamic history tensor of shape :math:`(H, n_{cells}, 3)` with channels
            [WD=0, VX=1, VY=2] where :math:`H` is history length and :math:`n_{cells}`
            is the number of cells.

        Returns
        -------
        torch.Tensor
            Noisy dynamic history tensor of same shape as input.
        """
        if self.noise_type == "none" or all(s <= 0.0 for s in self.noise_std):
            return dynamic_hist

        n_history, num_cells, d = dynamic_hist.shape
        device = dynamic_hist.device
        H = n_history

        def make_noise_for_all_steps(n_steps: int, n_cells: int):
            base = torch.randn((n_steps, n_cells, d), device=device)
            std_tensor = torch.tensor(self.noise_std, device=device).view(1, 1, d)
            return base * std_tensor

        if self.noise_type == "only_last":
            step_noise = make_noise_for_all_steps(1, num_cells)[0]
            dynamic_hist[-1] += step_noise

        elif self.noise_type == "correlated":
            single_noise = make_noise_for_all_steps(1, num_cells)[0]
            for t in range(n_history):
                dynamic_hist[t] += single_noise

        elif self.noise_type == "uncorrelated":
            noise_ = make_noise_for_all_steps(n_history, num_cells)
            dynamic_hist += noise_

        elif self.noise_type == "random_walk":
            step_sigma = [s / math.sqrt(H) for s in self.noise_std]
            step_sigma_t = torch.tensor(step_sigma, device=device).view(1, 1, 3)
            offset = torch.zeros((num_cells, 3), device=device)
            for t in range(n_history):
                # Broadcast step_sigma_t across channels: (1, 1, 3) -> (3,)
                step_n = torch.randn((num_cells, 3), device=device) * step_sigma_t.squeeze()
                offset += step_n
                dynamic_hist[t] += offset

        else:
            warnings.warn(f"Unknown noise_type={self.noise_type}, skipping noise.")
        return dynamic_hist

    def __getitem__(self, idx):
        r"""Get a single sample."""
        run_id, target_t = self.sample_index[idx]
        in_geom = self.xy_coords
        static_feats = self.static_data
        dyn_dict = self.dynamic_data[run_id]
        bc_dict = self.boundary_data[run_id]
        t0 = target_t - self.n_history
        num_cells = in_geom.shape[0]

        # Build dynamic history with order [WD, VX, VY]
        wanted_order = ["WD", "VX", "VY"]
        hist_list = []
        for dkey in wanted_order:
            arr_slice = dyn_dict[dkey][t0:target_t, :]
            hist_list.append(arr_slice.unsqueeze(-1))
        dynamic_hist = torch.cat(hist_list, dim=-1)

        # Noise injection
        dynamic_hist = self._apply_noise(dynamic_hist)

        # Boundary condition history
        boundary_vars = sorted(bc_dict.keys())
        bc_list = []
        for bck in boundary_vars:
            bc_slice = bc_dict[bck][t0:target_t, :]
            bc_list.append(bc_slice)
        if bc_list:
            bc_hist = torch.cat(bc_list, dim=-1)
        else:
            bc_hist = torch.zeros((self.n_history, num_cells, 1))

        # Build target [WD, VX, VY]
        def safe_get(k):
            if k not in dyn_dict:
                return torch.zeros((num_cells,))
            return dyn_dict[k][target_t, :]

        wd = safe_get("WD").unsqueeze(-1)
        vx = safe_get("VX").unsqueeze(-1)
        vy = safe_get("VY").unsqueeze(-1)
        target_all = torch.cat([wd, vx, vy], dim=-1)

        return {
            "geometry": in_geom,
            "static": static_feats,
            "boundary": bc_hist,
            "dynamic": dynamic_hist,
            "target": target_all,
            "run_id": run_id,
            "time_index": target_t,
        }


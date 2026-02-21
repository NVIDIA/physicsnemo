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

import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any, Literal, Sequence

import pyvista as pv
import torch
from tensordict import TensorDict, tensorclass
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.projections import project
from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.airfrans.dataset")


class CachedPreprocessingDataset(Dataset, ABC):
    """Dataset that lazily preprocesses samples and caches results to disk/RAM.

    Subclasses implement the ``preprocess`` static method to define how raw
    samples are transformed. On first access the result is computed and
    (optionally) saved to *cache_dir* as a ``.pt`` file keyed by the sample
    directory name. Subsequent accesses load the cached result directly.

    Args:
        sample_paths: Paths to individual samples in the dataset.
        cache_dir: Directory for disk caching. ``None`` disables disk caching.
        use_ram_caching: If True, wraps ``__getitem__`` with
            ``functools.cache`` for in-memory caching (increases memory usage).

    Raises:
        FileNotFoundError: If any *sample_paths* entry does not exist on disk.
    """

    def __init__(
        self,
        sample_paths: Sequence[Path | str],
        cache_dir: Path | str | None = None,
        use_ram_caching: bool = False,
    ):
        self.sample_paths = [Path(path) for path in sample_paths]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_ram_caching = use_ram_caching

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        nonexistent_sample_paths = [
            path for path in self.sample_paths if not path.exists()
        ]
        if nonexistent_sample_paths:
            raise FileNotFoundError(
                "The following sample paths were given, but do not exist:\n"
                f"{nonexistent_sample_paths}"
            )

        if self.use_ram_caching:
            self.__getitem__ = cache(self.__getitem__)

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx) -> Any:
        sample_path = self.sample_paths[idx]

        if self.cache_dir is not None:
            cache_path = (self.cache_dir / sample_path.name).with_suffix(".pt")
            if cache_path.exists():
                return torch.load(cache_path, weights_only=False)

        sample = self.preprocess(sample_path=sample_path)

        if self.cache_dir is not None:
            torch.save(sample, cache_path)

        return sample

    @staticmethod
    @abstractmethod
    def preprocess(sample_path: Path) -> Any:
        """Transform a raw sample at *sample_path* into the desired format."""


# --- AirFRANS constants ---
RHO = 1  # kg/m^3
# NOTE: this RHO is correct; in some places, the AirFRANS authors incorrectly
# report their density as 1.204, but if you actually dig into the OpenFOAM case
# files, you can see that the density is actually 1. You can also confirm this
# from the data itself - observe that RHO=1 yields constant far-field total
# pressure (which is physically correct), but RHO=1.204 does not (which is
# physically incorrect).
NU = 1.56e-5  # m^2/s


@tensorclass
class AirFRANSSample:
    interior_mesh: Mesh  # Point cloud of the volume mesh
    boundary_meshes: TensorDict  # BC name -> Mesh; TensorDict so .to(device) transfers them
    reference_lengths: TensorDict  # dict of reference length names to scalar tensors
    global_scalars: TensorDict  # dict of global scalar names to scalar tensors
    global_vectors: TensorDict  # dict of global vector names to vector tensors

    @property
    def model_input_kwargs(self) -> dict:
        """Kwargs for :meth:`GLOBE.forward`, minus control-flow args like ``chunk_size``."""
        return {
            "prediction_points": self.interior_mesh.points,
            "boundary_meshes": self.boundary_meshes,
            "reference_lengths": self.reference_lengths,
            "global_scalars": self.global_scalars,
            "global_vectors": self.global_vectors,
        }



class AirFRANSDataSet(CachedPreprocessingDataset):
    @classmethod
    def get_split_paths(
        cls,
        data_dir: Path,
        task: Literal["full", "scarce", "reynolds", "aoa"],
        split: Literal["train", "test"],
    ) -> list[Path]:
        """Read ``manifest.json`` and return sample paths for a task/split.

        For the ``"scarce"`` task, the test split uses the ``"full"`` test set
        (``"scarce"`` only defines a reduced training set).

        Args:
            data_dir: Root directory containing ``manifest.json`` and sample
                subdirectories.
            task: AirFRANS task name (``"full"``, ``"scarce"``, ``"reynolds"``,
                ``"aoa"``).
            split: ``"train"`` or ``"test"``.

        Returns:
            List of absolute paths to individual sample directories.
        """
        manifest = json.loads((data_dir / "manifest.json").read_text())
        effective_task = "full" if (task == "scarce" and split == "test") else task
        return [data_dir / f for f in manifest[f"{effective_task}_{split}"]]

    @classmethod
    def make_dataloader(
        cls,
        sample_paths: Sequence[Path],
        cache_dir: Path,
        *,
        world_size: int = 1,
        rank: int = 0,
        num_workers: int = 8,
    ) -> DataLoader:
        """Create a distributed DataLoader for this dataset.

        Each item is a single sample (``batch_size=None``) with identity
        collation, suitable for variable-size mesh data that cannot be
        stacked into uniform batches.

        Args:
            sample_paths: Paths to individual sample directories.
            cache_dir: Directory for disk caching of preprocessed samples.
            world_size: Total number of distributed ranks.
            rank: This process's distributed rank.
            num_workers: Number of DataLoader worker processes.

        Returns:
            Configured DataLoader with distributed sampling.
        """
        dataset = cls(sample_paths=sample_paths, cache_dir=cache_dir)
        return DataLoader(
            dataset,
            sampler=DistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
            ),
            batch_size=None,
            collate_fn=lambda x: x,
            num_workers=num_workers,
            prefetch_factor=32 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    @staticmethod
    def preprocess(
        sample_path: Path,
        patch_out_nonphysical_values: bool = True,
    ) -> "AirFRANSSample":
        pv_meshes = AirFRANSDataSet.load_pyvista_meshes(sample_path)

        # Project 3D boundary meshes to 2D (drop z axis, keep x and y).
        # transform_cell_data=True projects vector cell data (e.g., velocity
        # "U") from 3D to 2D alongside the geometry.
        freestream = project(
            from_pyvista(pv_meshes["freestream"], manifold_dim=1),
            keep_dims=[0, 1],
            transform_cell_data=True,
        )
        airfoil = project(
            from_pyvista(pv_meshes["airfoil"], manifold_dim=1),
            keep_dims=[0, 1],
        )
        internal = pv_meshes["internal"]

        def get(
            fieldname: str, preference: Literal["point", "cell"] = "point"
        ) -> torch.Tensor:
            if fieldname == "points":
                array = internal.points
            else:
                if preference == "point":
                    array = internal.point_data[fieldname]
                elif preference == "cell":
                    array = internal.cell_data[fieldname]
                else:
                    raise ValueError(f"Invalid preference: {preference}")
            tensor = torch.tensor(array, dtype=torch.float32)
            indices = tuple([slice(None)] + [slice(None, 2)] * (tensor.ndim - 1))
            return tensor[indices]

        # Compute freestream scaling from the (already-projected) 2D velocity
        U_inf = freestream.cell_data["U"].mean(dim=0)
        U_inf_magnitude = torch.norm(U_inf)
        q_inf = 0.5 * RHO * U_inf_magnitude**2

        # Targets from internal volume mesh
        U = get("U")
        U_over_U_inf = U / U_inf_magnitude
        q = q_inf * torch.sum(U_over_U_inf**2, dim=-1)
        p = get("p")
        C_p = p / q_inf
        C_pt = (p + q) / q_inf
        nut = get("nut")
        chord = 1.0

        ### [Assemble the AirFRANSSample]
        airfoil_for_model = Mesh(points=airfoil.points, cells=airfoil.cells)
        prediction_points = get("points")
        output_dict = TensorDict(
            {
                "U/|U_inf|": U_over_U_inf,
                "ΔU/|U_inf|": (U - U_inf[None, :]) / U_inf_magnitude,
                "C_p": C_p,
                "C_pt": C_pt,
                "ln(1+nut/nu)": torch.log1p(nut / NU),
            },
            batch_size=torch.Size([len(prediction_points)]),
        )

        ### [Adds the pressure gradient vector field]
        # Compute gradient on cells (more stable than on points)
        internal.cell_data["C_p"] = internal.cell_data["p"] / q_inf.item()
        mesh_with_grad = internal.compute_derivative(
            scalars="C_p", gradient=True, preference="cell"
        ).cell_data_to_point_data()
        grad_C_p = (
            torch.tensor(
                mesh_with_grad.point_data["gradient"][:, :2], dtype=torch.float32
            )
            * chord  # Nondimensionalizes
        )

        ### Remove non-physical pressure gradients, which come from extremely fine wall-normal resolution and floating-point error.
        grad_C_p_mag = torch.norm(grad_C_p, dim=-1)
        grad_C_p[grad_C_p_mag > 20] = torch.nan
        output_dict["∇C_p*chord"] = grad_C_p

        ### [Adds local forces]
        airfoil_point_normals = AirFRANSDataSet.compute_airfoil_point_normals(
            internal=internal, airfoil=pv_meshes["airfoil"]
        )
        velocity_gradient_tensor = torch.tensor(
            internal.compute_derivative(scalars="U", gradient="jacobian")
            .point_data["jacobian"]
            .reshape(-1, 3, 3)[:, :2, :2],
            dtype=torch.float32,
        )
        strain_rate_tensor = 0.5 * (
            velocity_gradient_tensor + velocity_gradient_tensor.transpose(1, 2)
        )
        wall_shear_stress_tensor = 2 * NU * strain_rate_tensor
        wall_shear_force = torch.einsum(
            "pij,pj->pi", wall_shear_stress_tensor, airfoil_point_normals
        )
        pressure_force = -1 * p[:, None] * airfoil_point_normals
        net_force = wall_shear_force + pressure_force
        output_dict["C_F,shear"] = wall_shear_force / q_inf
        output_dict["C_F,pressure"] = pressure_force / q_inf
        output_dict["C_F"] = net_force / q_inf

        if patch_out_nonphysical_values:
            non_physical_C_pt = C_pt > 1.02
            if non_physical_C_pt.sum() / len(C_pt) > 0.0001:
                logger.warning(
                    f"In {sample_path.name}, {non_physical_C_pt.sum() / len(C_pt):.2%} of points had non-physical total pressures and were patched out."
                )
            output_dict[non_physical_C_pt] = torch.nan

        ### Triangulate internal mesh to preserve cell connectivity for visualization
        pv_trimesh = internal.triangulate()
        interior_cells = torch.tensor(pv_trimesh.cells_dict[5], dtype=torch.long)

        return AirFRANSSample(
            interior_mesh=Mesh(
                points=prediction_points,
                cells=interior_cells,
                point_data=output_dict,
                global_data=TensorDict({
                    "U_inf": U_inf,
                    "q_inf": q_inf,
                }),
            ),
            boundary_meshes=TensorDict(
                {"no_slip": airfoil_for_model}, batch_size=[]
            ),
            reference_lengths=TensorDict(
                {
                    "chord": torch.as_tensor(chord),
                    "delta_FS": torch.as_tensor((NU / U_inf_magnitude * chord) ** 0.5),
                },
                batch_size=torch.Size([]),
            ),
            global_scalars=TensorDict(),
            global_vectors=TensorDict(
                {"U_inf / U_inf_magnitude": U_inf / U_inf_magnitude},
                batch_size=torch.Size([2]),
            ),
            batch_size=torch.Size([]),
        )

    @staticmethod
    def compute_airfoil_point_normals(
        internal: pv.UnstructuredGrid | pv.PolyData, airfoil: pv.PolyData
    ) -> torch.Tensor:
        point_is_on_airfoil = internal.point_data["implicit_distance"] == 0
        airfoil_point_normals = torch.tensor(
            internal.sample(
                target=airfoil,
                snap_to_closest_point=True,
            )["Normals"][:, :2]
            * -1,  # Oriented so that they point outwards (i.e., into the fluid domain)
            dtype=torch.float32,
        )
        airfoil_point_normals[~point_is_on_airfoil] = torch.nan
        return airfoil_point_normals

    @staticmethod
    def load_pyvista_meshes(sample_path: Path) -> dict[str, pv.PolyData]:
        """Load PyVista meshes for a given AirFRANS sample.

        Loads the three required mesh files for an AirFRANS sample: freestream boundary,
        airfoil boundary, and internal volume mesh. The meshes are expected to follow
        the standard AirFRANS naming convention.

        Args:
            sample_path: Path to the sample directory containing the mesh files.
                Expected files are:
                - {sample_name}_freestream.vtp: Freestream boundary mesh
                - {sample_name}_aerofoil.vtp: Airfoil boundary mesh
                - {sample_name}_internal.vtu: Internal volume mesh

        Returns:
            Dictionary mapping mesh names to loaded PyVista mesh objects:
            - "freestream": Freestream boundary mesh (PolyData)
            - "airfoil": Airfoil boundary mesh (PolyData)
            - "internal": Internal volume mesh (UnstructuredGrid)

        Raises:
            FileNotFoundError: If any of the required mesh files are missing.

        Note:
            The sample directory name is used as the base name for constructing
            the expected mesh file names.
        """
        sample_dir = Path(sample_path)
        base = sample_dir.name

        mesh_paths = {
            "freestream": sample_dir / f"{base}_freestream.vtp",
            "airfoil": sample_dir / f"{base}_aerofoil.vtp",
            "internal": sample_dir / f"{base}_internal.vtu",
        }

        for f in mesh_paths.values():
            if not f.exists():
                raise FileNotFoundError(f"Missing required file: {f}")

        return {k: pv.read(v) for k, v in mesh_paths.items()}

    @staticmethod
    def postprocess(
        pred_mesh: Mesh,
        true_mesh: Mesh,
        *,
        fields: Sequence[str] | None = None,
        show: bool = True,
        show_error: bool = True,
    ) -> Mesh:
        """Visualize and compare predicted vs. true fields on a combined Mesh.

        Builds a combined Mesh whose ``point_data`` contains nested ``"true"``,
        ``"pred"``, and ``"error"`` TensorDicts, then renders a subplot grid
        using :meth:`Mesh.draw`.

        Args:
            pred_mesh: Point-cloud Mesh with predicted field values in
                ``point_data``.
            true_mesh: Mesh with ground-truth field values in ``point_data``.
                Should have cell connectivity (from preprocessing) for
                filled-polygon rendering.
            fields: Which field names to compare. If ``None``, uses the sorted
                intersection of pred and true ``point_data`` keys.
            show: Whether to display the plot via ``plt.show()``.
            show_error: Whether to include an error row in the subplot grid.

        Returns:
            Combined Mesh with ``point_data["true"]``, ``point_data["pred"]``,
            and ``point_data["error"]`` containing the selected fields.

        Raises:
            ValueError: If pred_mesh and true_mesh have different numbers of
                points.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if pred_mesh.n_points != true_mesh.n_points:
            raise ValueError(
                f"Point count mismatch: {pred_mesh.n_points=} != {true_mesh.n_points=}"
            )

        ### Determine fields to compare
        if fields is None:
            fields = sorted(
                set(pred_mesh.point_data.keys()) & set(true_mesh.point_data.keys())
            )

        ### Build combined Mesh with nested point_data
        pred_selected = pred_mesh.point_data.select(*fields)
        true_selected = true_mesh.point_data.select(*fields)
        error_data = pred_selected.apply(lambda p, t: p - t, true_selected)

        combined = Mesh(
            points=true_mesh.points,
            cells=true_mesh.cells,
            point_data=TensorDict(
                {
                    "true": true_selected,
                    "pred": pred_selected,
                    "error": error_data,
                },
                batch_size=[true_mesh.n_points],
            ),
        )

        ### Create subplot grid
        kind_labels = ["Truth", "Prediction"]
        kind_keys = ["true", "pred"]
        if show_error:
            kind_labels.append("Error")
            kind_keys.append("error")
        n_rows, n_cols = len(kind_labels), len(fields)

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4 * n_cols, 3.4 * n_rows),
            squeeze=False,
        )

        for col, field_name in enumerate(fields):
            ### Compute shared vmin/vmax across truth and prediction
            true_vals = true_selected[field_name]
            pred_vals = pred_selected[field_name]
            is_vector = true_vals.ndim > 1 and true_vals.shape[-1] > 1
            true_scalars = true_vals.norm(dim=-1) if is_vector else true_vals.reshape(-1)
            pred_scalars = pred_vals.norm(dim=-1) if is_vector else pred_vals.reshape(-1)

            all_finite = torch.cat([
                true_scalars[torch.isfinite(true_scalars)],
                pred_scalars[torch.isfinite(pred_scalars)],
            ])
            shared_vmin = all_finite.min().item() if len(all_finite) > 0 else 0.0
            shared_vmax = all_finite.max().item() if len(all_finite) > 0 else 1.0
            if shared_vmin == shared_vmax:
                shared_vmin -= 1e-6
                shared_vmax += 1e-6

            for row, (label, key) in enumerate(zip(kind_labels, kind_keys)):
                ax = axes[row, col]

                if key == "error":
                    err_vals = error_data[field_name]
                    if is_vector:
                        finite_err = err_vals.norm(dim=-1)
                        finite_err = finite_err[torch.isfinite(finite_err)]
                        emax = finite_err.max().item() if len(finite_err) > 0 else 1.0
                        cmap, vmin, vmax = "Reds", 0.0, emax
                    else:
                        finite_err = err_vals.reshape(-1)
                        finite_err = finite_err[torch.isfinite(finite_err)]
                        emax = finite_err.abs().max().item() if len(finite_err) > 0 else 1.0
                        cmap, vmin, vmax = "RdBu_r", -emax, emax
                else:
                    cmap, vmin, vmax = "turbo", shared_vmin, shared_vmax

                combined.draw(
                    point_scalars=(key, field_name),
                    ax=ax,
                    show=False,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    show_edges=False,
                )

                sm = mpl.cm.ScalarMappable(  # ty: ignore[unresolved-attribute]
                    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),  # ty: ignore[unresolved-attribute]
                    cmap=plt.get_cmap(cmap),
                )
                fig.colorbar(
                    sm,
                    ax=ax,
                    orientation="horizontal",
                    shrink=0.8,
                    fraction=0.03,
                    aspect=50,
                    pad=0.01,
                )

                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(
                    axis="both",
                    which="both",
                    length=0,
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False,
                )
                if row == 0:
                    ax.set_title(field_name, fontsize=12, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(label, fontsize=12, fontweight="bold")

        plt.tight_layout(h_pad=0.1, w_pad=0)
        if show:
            plt.show()

        return combined

    @staticmethod
    def visualize_output_distributions(
        output_dict: TensorDict,
        show: bool = True,
    ) -> None:
        """Visualize distributions of output quantities with histograms.

        Creates a subplot grid showing the distribution of each output quantity,
        with special handling for vector fields (showing magnitude distributions).

        Args:
            output_dict: Dictionary of output tensors from preprocessing
            show: Whether to display the plot with plt.show()
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import polars as pl

        ### Helper function to get plottable values
        def get_plot_values(values: torch.Tensor) -> tuple[np.ndarray, str]:
            """Convert tensor to plottable array and determine label."""
            if values.ndim > 1 and values.shape[-1] > 1:
                # Vector quantity - return magnitude
                return torch.linalg.norm(
                    values, dim=-1
                ).detach().cpu().numpy(), " (magnitude)"
            else:
                return values.detach().cpu().numpy().flatten(), ""

        ### Create subplot grid
        n_outputs = len(output_dict.keys())
        n_cols = 3
        n_rows = (n_outputs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_outputs > 1 else [axes]

        ### Plot distributions
        for idx, (key, values) in enumerate(output_dict.items()):
            ax = axes[idx]
            plot_values, suffix = get_plot_values(values)

            # Histogram with mean line
            ax.hist(plot_values, bins=50, alpha=0.7, edgecolor="black")
            mean = np.nanmean(plot_values)
            ax.axvline(
                mean,
                color="red",
                linestyle="--",
                label=f"{mean = :.2f}",
                alpha=0.7,
            )

            # Formatting
            ax.set_title(f"{key}{suffix} distribution")
            ax.set_xlabel("Value" if not suffix else "Magnitude")
            ax.set_yscale("log")
            ax.set_ylabel("Count (log scale)")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        if show:
            plt.show()

        ### Print summary statistics using Polars
        logger.info("\n### Summary Statistics ###")
        stats_data = {
            f"{key}{get_plot_values(values)[1]}": get_plot_values(values)[0]
            for key, values in output_dict.items()
        }
        df = pl.DataFrame(stats_data)
        # Replace NaN values with nulls so Polars handles them properly
        df = df.fill_nan(None)
        logger.info(f"\n{df.describe()}")


def compute_max_mesh_sizes(
    dataloader: DataLoader,
    device: torch.device,
    *,
    face_downsampling_ratio: float = 1.0,
    rank: int = 0,
) -> dict[str, dict[str, int]]:
    """Compute the maximum n_points and n_cells per boundary-condition type.

    Scans all samples in *dataloader*, tracking the largest boundary mesh
    dimensions for each BC type. Uses distributed all-reduce to find the
    global maximum across all ranks. The results are used to pad meshes to
    uniform sizes for ``torch.compile`` with static shapes.

    Args:
        dataloader: DataLoader yielding ``AirFRANSSample`` objects.
        device: Device for the all-reduce tensors.
        face_downsampling_ratio: Scale factor applied to cell counts. Use
            a value < 1.0 for training (downsampled meshes) and 1.0 for
            validation (full meshes).
        rank: Distributed rank (progress bar shown only on rank 0).

    Returns:
        Mapping ``{bc_type: {"n_points": int, "n_cells": int}}``.
    """
    max_sizes: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_points": 0, "n_cells": 0}
    )

    for sample in tqdm(
        dataloader,
        desc=f"Computing max mesh sizes (rank {rank})",
        disable=rank != 0,
    ):
        for bc_type, mesh in sample.boundary_meshes.items():
            max_sizes[bc_type]["n_points"] = max(
                max_sizes[bc_type]["n_points"], mesh.n_points
            )
            n_cells = (
                int(mesh.n_cells * face_downsampling_ratio)
                if face_downsampling_ratio != 1.0
                else mesh.n_cells
            )
            max_sizes[bc_type]["n_cells"] = max(
                max_sizes[bc_type]["n_cells"],
                n_cells,
            )

    for bc_type in max_sizes:
        size_tensor = torch.tensor(
            [max_sizes[bc_type]["n_points"], max_sizes[bc_type]["n_cells"]],
            device=device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(size_tensor, op=torch.distributed.ReduceOp.MAX)
        max_sizes[bc_type]["n_points"] = int(size_tensor[0])
        max_sizes[bc_type]["n_cells"] = int(size_tensor[1])

    if rank == 0:
        logger.info(f"Max mesh sizes: {dict(max_sizes)}")

    return dict(max_sizes)


if __name__ == "__main__":
    import os

    if not (_data_env := os.environ.get("AIRFRANS_DATA_DIR")):
        raise ValueError("AIRFRANS_DATA_DIR environment variable is not set.")
    data_dir = Path(_data_env)
    sample_paths = list(data_dir.iterdir())

    # Preprocess a sample
    sample = AirFRANSDataSet.preprocess(sample_paths[0])

    logger.info(f"Sample path: {sample_paths[0]}")
    logger.info(f"Interior mesh points: {sample.interior_mesh.points.shape}")
    logger.info(f"Output keys: {list(sample.interior_mesh.point_data.keys())}")
    logger.info(f"Boundary meshes: {list(sample.boundary_meshes.keys())}")

    # Visualize the output distributions
    output_dict = sample.interior_mesh.point_data
    AirFRANSDataSet.visualize_output_distributions(output_dict, show=True)

    AirFRANSDataSet.postprocess(
        pred_mesh=sample.interior_mesh,
        true_mesh=sample.interior_mesh,
    )

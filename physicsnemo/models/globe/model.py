import operator
from collections import defaultdict
from functools import reduce
from typing import Literal, Sequence

import pyvista as pv
import torch
import torch.nn as nn
from tensordict import TensorDict

from physicsnemo.models.globe.boundary_mesh import BoundaryMesh
from physicsnemo.models.globe.field_kernel import MultiscaleKernel
from physicsnemo.models.globe.utilities.tensordict_utils import (
    combine_tensordicts,
    concatenated_length,
    split_by_leaf_rank,
)


class GLOBE(nn.Module):
    """Green's-function-Like Operator for Boundary Element PDEs.

    GLOBE is a neural surrogate architecture for boundary-driven elliptic PDEs that
    combines learnable Green's-function-like kernels with equivariant ML. The model
    represents solutions as superpositions of kernel evaluations from boundary faces
    to target points, with communication hyperlayers enabling boundary-to-boundary
    information propagation before final interior evaluation.

    The architecture is designed to satisfy fundamental physical requirements:
    - Translation-, rotation-, and parity-equivariant through relative positions and
      local basis reprojection
    - Discretization-invariant via area-weighted boundary integrals
    - Units-invariant through rigorous nondimensionalization
    - Global receptive field through all-to-all boundary-to-target evaluation

    Architecture overview (see paper Section 3):
    1. Communication hyperlayers propagate latent information between boundary
       condition partitions (Section 3.4)
    2. Each hyperlayer uses multiscale kernels operating at different reference
       length scales (Section 3.3)
    3. Final hyperlayer evaluates fields at user-specified query points
    4. Learnable per-field calibration transforms applied to outputs

    Args:
        n_spatial_dims: Number of spatial dimensions (2 or 3).
        output_fields: Dictionary mapping field names to types ("scalar" or "vector").
            These are the physical fields predicted at query points in the final
            hyperlayer.
        boundary_condition_names: Sequence of boundary condition type identifiers
            (e.g., ["no_slip", "freestream", "slip"]). Each BC type gets its own
            set of kernels. Names must not contain "." for TensorDict compatibility.
        boundary_condition_n_source_scalars: Dictionary mapping each BC type name to
            the number of scalar features per boundary face for that BC type.
        boundary_condition_n_source_vectors: Dictionary mapping each BC type name to
            the number of vector features per boundary face for that BC type. The
            face normal vector is automatically added, so don't include it in the count.
        reference_length_names: Sequence of identifiers for reference length scales
            (e.g., ["viscous_length", "chord_length"]). Each creates a separate kernel
            branch in the multiscale composition.
        reference_area: Scalar tensor used to nondimensionalize face areas. Typically
            a characteristic area of the problem (e.g., chord^2 for airfoils).
        n_global_scalars: Number of global scalar features (e.g., Reynolds number,
            Mach number). These are shared across all source faces.
        n_global_vectors: Number of global vector features (e.g., freestream velocity
            direction). These are shared across all source faces.
        n_communication_hyperlayers: Number of boundary-to-boundary communication
            layers before final evaluation. Higher values enable more information
            exchange between boundary partitions. Default: 2.
        n_latent_scalars: Number of scalar latent channels propagated between
            hyperlayers. Default: 12.
        n_latent_vectors: Number of vector latent channels propagated between
            hyperlayers. Default: 6.
        smoothing_radius: Small value for numerical stability in magnitude computations.
            Default: 1e-8.
        hidden_layer_sizes: Hidden layer sizes for kernel neural networks. If None,
            defaults to [64, 64, 64].
        n_spherical_harmonics: Number of Legendre polynomial terms used for
            angle-dependent features in kernel functions. Default: 4.

    Attributes:
        kernel_layers: ModuleList of communication hyperlayers, each containing a
            ModuleDict mapping BC type names to MultiscaleKernel instances.
        final_field_transforms: ModuleDict of per-field linear calibration layers.

    Example:
        >>> model = GLOBE(
        ...     n_spatial_dims=3,
        ...     output_fields={"pressure": "scalar", "velocity": "vector"},
        ...     boundary_condition_names=["no_slip", "freestream"],
        ...     boundary_condition_n_source_scalars={"no_slip": 0, "freestream": 0},
        ...     boundary_condition_n_source_vectors={"no_slip": 0, "freestream": 0},
        ...     reference_length_names=["delta_FS", "chord"],
        ...     reference_area=torch.tensor(1.0),
        ...     n_global_scalars=0,
        ...     n_global_vectors=0,
        ... )
        >>> result = model(
        ...     prediction_points=torch.randn(100, 3),
        ...     boundary_meshes=[wing_mesh, fuselage_mesh],
        ...     reference_lengths={"delta_FS": torch.tensor(0.01), "chord": torch.tensor(1.0)},
        ... )
    """

    def __init__(
        self,
        n_spatial_dims: int,
        output_fields: dict[str, Literal["scalar", "vector"]],
        boundary_condition_names: Sequence[str],
        boundary_condition_n_source_scalars: dict[str, int],
        boundary_condition_n_source_vectors: dict[str, int],
        reference_length_names: Sequence[str],
        reference_area: torch.Tensor,
        n_global_scalars: int,
        n_global_vectors: int,
        n_communication_hyperlayers: int = 2,
        n_latent_scalars: int = 12,
        n_latent_vectors: int = 6,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
    ):
        ### Sets defaults
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64, 64]

        # Validate inputs
        if not torch.compiler.is_compiling():
            for field_name, field_type in output_fields.items():
                if field_type not in ["scalar", "vector"]:
                    raise ValueError(
                        f"In `output_fields`, for {field_name=!r}, got {field_type=!r};\n"
                        "This must be one of ['scalar', 'vector']"
                    )
            for bc_name in boundary_condition_names:
                if "." in bc_name:
                    raise ValueError(
                        f"In `boundary_condition_names`, got {bc_name=!r};\n"
                        "This must not contain any `.` characters, for nested-TensorDict compatibility."
                    )

        super().__init__()

        # Store arguments
        self.n_spatial_dims = n_spatial_dims
        self.output_fields = output_fields
        self.boundary_condition_names = boundary_condition_names
        self.boundary_condition_n_source_scalars = boundary_condition_n_source_scalars
        self.boundary_condition_n_source_vectors = boundary_condition_n_source_vectors
        self.reference_length_names = reference_length_names
        self.reference_area = reference_area
        self.n_global_scalars = n_global_scalars
        self.n_global_vectors = n_global_vectors
        self.n_communication_hyperlayers = n_communication_hyperlayers
        self.n_latent_scalars = n_latent_scalars
        self.n_latent_vectors = n_latent_vectors
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics

        kernel_layers = []

        for i in range(self.n_communication_hyperlayers + 1):
            is_first_hyperlayer = i == 0
            is_last_hyperlayer = i == self.n_communication_hyperlayers

            kernel_layers.append(
                nn.ModuleDict(
                    {
                        source_bc_type: MultiscaleKernel(
                            n_spatial_dims=n_spatial_dims,
                            output_fields=output_fields
                            if is_last_hyperlayer
                            else {
                                **{
                                    f"strengths.{name}": "scalar"
                                    for name in reference_length_names
                                },
                                **{
                                    f"latent_scalars.{i}": "scalar"
                                    for i in range(n_latent_scalars)
                                },
                                **{
                                    f"latent_vectors.{i}": "vector"
                                    for i in range(n_latent_vectors)
                                },
                            },
                            reference_length_names=reference_length_names,
                            n_source_scalars=(
                                boundary_condition_n_source_scalars[source_bc_type]
                                + (0 if is_first_hyperlayer else n_latent_scalars)
                            ),
                            n_source_vectors=(
                                boundary_condition_n_source_vectors[source_bc_type]
                                + (0 if is_first_hyperlayer else n_latent_vectors)
                                + 1  # +1 for the normal vector
                            ),
                            n_global_scalars=n_global_scalars,
                            n_global_vectors=n_global_vectors,
                            smoothing_radius=smoothing_radius,
                            hidden_layer_sizes=hidden_layer_sizes,
                            n_spherical_harmonics=n_spherical_harmonics,
                        )
                        for source_bc_type in boundary_condition_names
                    }
                )
            )

        self.kernel_layers = nn.ModuleList(kernel_layers)
        self.final_field_transforms = nn.ModuleDict(
            {
                field_name: nn.Linear(
                    in_features=1,
                    out_features=1,
                    bias=(output_fields[field_name] == "scalar"),
                )
                for field_name in output_fields.keys()
            }
        )

    def forward(
        self,
        prediction_points: torch.Tensor,
        boundary_meshes: Sequence[BoundaryMesh],
        reference_lengths: dict[str, torch.Tensor],
        global_scalars: TensorDict | None = None,
        global_vectors: TensorDict | None = None,
        chunk_size: None | int | Literal["auto"] = None,
        verbose: bool = True,
    ) -> TensorDict:
        """Evaluates GLOBE model to predict fields at target points from boundary conditions.

        This method implements the full GLOBE forward pass through all communication
        hyperlayers. The process:

        1. **BC grouping and merging**: Boundary meshes are grouped by BC type and merged
           to create one unified mesh per BC type.

        2. **Hyperlayer communication** (layers 0 to n_communication_hyperlayers-1):
           - Each BC type evaluates its multiscale kernel from source faces to target faces
           - Targets are the face centers of all boundary meshes (enabling BC-to-BC communication)
           - Source strengths are weighted by face areas normalized by reference_area
           - Outputs include per-branch strengths for the next layer and latent scalars/vectors
           - Results from all source BC types are summed at each target face

        3. **Final evaluation** (layer n_communication_hyperlayers):
           - Each BC type evaluates its kernel from source faces to prediction_points
           - Source strengths use the values propagated through hyperlayers
           - Outputs are the requested physical fields (not latents)
           - Results from all source BC types are summed

        4. **Field calibration**: Per-field linear transforms applied (affine for scalars,
           scale-only for vectors to preserve rotation-equivariance).

        Args:
            prediction_points: Target points for field evaluation, shape (n_points, n_spatial_dims).
                These are typically interior domain points where you want to predict the solution.
            boundary_meshes: Sequence of BoundaryMesh objects representing the problem boundaries.
                Each mesh must have a boundary_condition_type matching one of the model's
                boundary_condition_names. Multiple meshes can share the same BC type (they will
                be merged automatically). Each mesh's face_data should contain the source
                scalars/vectors specified during model initialization.
            reference_lengths: Dictionary mapping reference length names to scalar tensors.
                Keys must match the model's reference_length_names. Each value should be a
                scalar tensor (shape ()) representing the physical length scale for that branch.
            global_scalars: Optional TensorDict with batch_size=() containing problem-level
                scalar features (e.g., Reynolds number). The total concatenated length must
                match n_global_scalars. Defaults to empty if None.
            global_vectors: Optional TensorDict with batch_size=(n_spatial_dims,) containing
                problem-level vector features (e.g., freestream direction). The total
                concatenated length must match n_global_vectors. Defaults to empty if None.
            chunk_size: Controls memory usage during kernel evaluation:
                - None: Evaluate all target points at once (fastest but high memory)
                - int: Process target points in chunks of this size (trades speed for memory)
                - "auto": Automatically determine chunk size targeting ~1GB per chunk
                Default: None.
            verbose: If True, prints progress information during evaluation. Default: True.

        Returns:
            TensorDict with batch_size=(n_points,) containing the predicted fields.
            Keys are the field names from output_fields. Scalar fields have shape (n_points,),
            vector fields have shape (n_points, n_spatial_dims).

        Raises:
            ValueError: If input dimensions don't match model configuration, if boundary
                condition types in boundary_meshes are not recognized, or if reference_lengths
                keys don't match expected names.

        Note:
            - Face areas are automatically normalized by reference_area to preserve
              discretization-invariance
            - The face normal vector is automatically added to source_vectors for each mesh
            - Hyperlayer communication enables long-range coupling between boundary partitions,
              analogous to the influence coefficient matrix solve in traditional boundary-element
              methods (but learned rather than explicitly computed)

        Example:
            >>> # Create boundary meshes with appropriate BC types
            >>> wing = BoundaryMesh.from_polydata(wing_surface, "no_slip")
            >>> freestream = BoundaryMesh.from_polydata(farfield_surface, "freestream")
            >>> # Generate interior evaluation points
            >>> points = torch.randn(1000, 3)
            >>> # Evaluate model
            >>> result = model(
            ...     prediction_points=points,
            ...     boundary_meshes=[wing, freestream],
            ...     reference_lengths={"delta_FS": torch.tensor(0.001), "chord": torch.tensor(1.0)},
            ...     global_scalars=TensorDict({"Re": torch.tensor([1e6])}, batch_size=()),
            ... )
            >>> pressure = result["pressure"]  # shape (1000,)
            >>> velocity = result["velocity"]  # shape (1000, 3)
        """
        device = prediction_points.device

        ### Set defaults
        if global_scalars is None:
            global_scalars = TensorDict({}, batch_size=torch.Size([]), device=device)
        if global_vectors is None:
            global_vectors = TensorDict(
                {}, batch_size=torch.Size([self.n_spatial_dims]), device=device
            )

        # Extract boundary condition types early so we can use it for validation
        bc_types_from_input: set[str] = set(
            bm.boundary_condition_type for bm in boundary_meshes
        )

        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            # Check that input lengths are consistent with the model
            for name, (actual, expected) in {
                "reference lengths": (
                    set(reference_lengths.keys()),
                    set(self.reference_length_names),
                ),
                "global scalars": (
                    concatenated_length(global_scalars),
                    self.n_global_scalars,
                ),
                "global vectors": (
                    concatenated_length(global_vectors),
                    self.n_global_vectors,
                ),
            }.items():
                if actual != expected:
                    raise ValueError(
                        f"This model was instantiated to expect {expected} {name},\n"
                        f"but the forward-method input gives {actual} {name}."
                    )

            # Check that dimensionality of all boundary meshes is consistent with the model
            if not all(
                bm.n_spatial_dims == self.n_spatial_dims for bm in boundary_meshes
            ):
                raise ValueError(
                    f"The input gives boundary meshes with these numbers of spatial dimensions:\n"
                    f"{list(bm.n_spatial_dims for bm in boundary_meshes)}\n"
                    f"but the model was instantiated to expect that these should all be equal to:\n"
                    f"{self.n_spatial_dims=!r}\n"
                )

            # Check that all boundary condition types are in the model
            if not bc_types_from_input.issubset(self.boundary_condition_names):
                raise ValueError(
                    f"The input gives boundary meshes with these boundary condition types:\n"
                    f"{bc_types_from_input=!r}\n"
                    f"but the model was instantiated to expect only these boundary condition types:\n"
                    f"{self.boundary_condition_names=!r}\n"
                    f"Please ensure that the input boundary meshes are a subset of the model's boundary condition types."
                )

        ### Group and merge boundary meshes by boundary condition type
        # Do the grouping + merging
        bc_meshes_list: dict[str, list[BoundaryMesh]] = defaultdict(list)
        for bm in boundary_meshes:
            bc_meshes_list[bm.boundary_condition_type].append(bm)

        bc_meshes: dict[str, BoundaryMesh] = {
            bc_type: BoundaryMesh.merge(meshes)
            for bc_type, meshes in bc_meshes_list.items()
        }

        ### Initialize the latent data that's passed between hyperlayers
        latent_data: dict[str, TensorDict] = {
            source_bc_type: TensorDict(
                {
                    "strengths": {
                        name: torch.ones(source_bc_mesh.n_faces, device=device)
                        for name in self.reference_length_names
                    },
                    "latent_scalars": {},
                    "latent_vectors": {},
                },
                batch_size=torch.Size([source_bc_mesh.n_faces]),
                device=device,
            )
            for source_bc_type, source_bc_mesh in bc_meshes.items()
        }

        ### Kernel evaluations
        for i in range(self.n_communication_hyperlayers + 1):
            is_last_hyperlayer = i == self.n_communication_hyperlayers

            if verbose:
                print(f"Evaluating hypernetwork layer {i}...")

            def evaluate_hyperlayer(target_points: torch.Tensor) -> TensorDict:
                result_pieces: list[TensorDict] = []

                for source_bc_type, source_bc_mesh in bc_meshes.items():
                    source_latent_data: TensorDict = latent_data[source_bc_type]
                    source_strengths: TensorDict = source_latent_data["strengths"]  # ty: ignore[invalid-assignment]
                    source_strengths = source_strengths.apply(
                        lambda x: x * (source_bc_mesh.face_areas / self.reference_area)
                    )

                    source_data_by_rank: dict[int, TensorDict] = split_by_leaf_rank(
                        source_bc_mesh.face_data
                    )
                    source_scalars = combine_tensordicts(
                        source_data_by_rank[0],
                        source_latent_data["latent_scalars"],  # ty: ignore[invalid-argument-type]
                    )
                    source_vectors = combine_tensordicts(
                        source_data_by_rank[1],
                        source_latent_data["latent_vectors"],  # ty: ignore[invalid-argument-type]
                    )
                    source_vectors["normals"] = source_bc_mesh.face_normals
                    source_vectors.batch_size = torch.Size(
                        [source_bc_mesh.n_faces, self.n_spatial_dims]
                    )

                    kernel: MultiscaleKernel = self.kernel_layers[i][source_bc_type]
                    result_from_kernel: TensorDict = kernel(
                        source_points=source_bc_mesh.face_centers,
                        source_scalars=source_scalars,
                        source_vectors=source_vectors,
                        source_strengths=source_strengths,
                        target_points=target_points,
                        reference_lengths=reference_lengths,
                        global_scalars=global_scalars,
                        global_vectors=global_vectors,
                        verbose=verbose,
                        chunk_size=chunk_size,
                    )
                    result_pieces.append(result_from_kernel.unflatten_keys())

                result: TensorDict = reduce(operator.add, result_pieces)
                return result

            if is_last_hyperlayer:
                result: TensorDict = evaluate_hyperlayer(prediction_points)
            else:
                latent_data = {
                    target_bc_type: evaluate_hyperlayer(target_bc_mesh.face_centers)
                    for target_bc_type, target_bc_mesh in bc_meshes.items()
                }

        for field_name, field_tensor in result.items():
            original_shape = field_tensor.shape
            result[field_name] = self.final_field_transforms[field_name](
                field_tensor.view(-1, 1)
            ).view(original_shape)

        return result


if __name__ == "__main__":
    torch._logging.set_logs(graph_breaks=True, recompiles=True)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(0.99)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Make some example data
    mesh = BoundaryMesh.from_polydata(
        pv.examples.load_airplane(), boundary_condition_type="no_slip"
    ).to(device)  # Surface mesh

    model = GLOBE(
        n_spatial_dims=3,
        reference_area=torch.tensor(100.0, device=device),
        output_fields={
            "pressure": "scalar",
            "velocity": "vector",
        },
        boundary_condition_names=["no_slip", "freestream", "slip"],
        boundary_condition_n_source_scalars={
            "no_slip": 0,
            "freestream": 0,
            "slip": 0,
        },
        boundary_condition_n_source_vectors={
            "no_slip": 0,
            "freestream": 0,
            "slip": 0,
        },
        n_communication_hyperlayers=2,
        n_latent_scalars=16,
        n_latent_vectors=3,
        reference_length_names=["delta_FS", "chord"],
        n_global_scalars=0,
        n_global_vectors=0,
    ).to(device)
    model.eval()
    # model = torch.compile(model, dynamic=True, fullgraph=True)

    # Generate 100 random-uniform points within the bounding box of the airplane surface
    min_bounds = mesh.points.min(dim=0).values
    max_bounds = mesh.points.max(dim=0).values
    n_points = 20
    prediction_points = torch.rand((n_points, 3), device=device)
    prediction_points = prediction_points * (max_bounds - min_bounds) + min_bounds

    reference_lengths = {
        "delta_FS": torch.tensor(0.01, device=device),
        "chord": torch.tensor(1.0, device=device),
    }

    n_interactions = len(prediction_points) * mesh.n_faces
    print(
        f"{len(prediction_points)} prediction points, {mesh.n_faces} faces --> {n_interactions} interactions"
    )
    print("Warming up model...")
    for _ in range(2):
        with torch.no_grad():
            result = model(
                prediction_points=prediction_points,
                boundary_meshes=[mesh],
                reference_lengths=reference_lengths,
                chunk_size=None,
                verbose=False,
            )

    import time

    import tqdm

    N_runs = 40
    start_time = time.perf_counter()
    for i in tqdm.trange(N_runs, desc="Benchmarking model performance", unit=" runs"):
        with torch.no_grad():
            result = model(
                prediction_points=prediction_points,
                boundary_meshes=[mesh],
                reference_lengths=reference_lengths,
                chunk_size=None,
                verbose=False,
            )
    elapsed = time.perf_counter() - start_time
    print(f"Done. Elapsed time: {elapsed:.3f} s")
    from aerosandbox.tools.string_formatting import eng_string

    print(f"Speed: {eng_string(n_interactions * N_runs / elapsed)} interactions/s")

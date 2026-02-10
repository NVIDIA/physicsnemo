# ruff: noqa: S101
import pytest
import torch

from physicsnemo.models.globe.model import GLOBE
from test.models.globe.test_boundary_mesh import make_tetrahedron_mesh


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_globe_inference(device: str) -> None:
    """Instantiate `GLOBE` and run a basic inference; validate keys and shapes."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ### Create model and mesh
    model = GLOBE(
        n_spatial_dims=3,
        output_fields={
            "pressure": "scalar",
            "velocity": "vector",
        },
        boundary_condition_names=["no_slip"],
        boundary_condition_n_source_scalars={"no_slip": 0},
        boundary_condition_n_source_vectors={"no_slip": 0},
        reference_length_names=["test_length"],
        reference_area=torch.tensor(1.0, device=device),
        n_global_scalars=0,
        n_global_vectors=0,
        hidden_layer_sizes=[8],
    ).to(device)
    model.eval()

    mesh = make_tetrahedron_mesh().to(device)

    ### Set up prediction points and reference lengths
    # Two prediction points near the origin; non-coplanar w.r.t. the full mesh
    prediction_points = torch.tensor(
        [
            [0.10, 0.10, 0.10],
            [0.25, 0.15, 0.05],
        ],
        dtype=torch.float32,
        device=device,
    )
    reference_lengths = {
        "test_length": torch.tensor(1.0, dtype=torch.float32, device=device)
    }

    ### Run inference
    with torch.no_grad():
        outputs = model(
            prediction_points=prediction_points,
            boundary_meshes=[mesh],
            reference_lengths=reference_lengths,
            chunk_size=None,
            verbose=False,
        )

    ### Validate outputs
    assert set(outputs.keys()) == {"pressure", "velocity"}
    assert outputs["pressure"].shape == (2,)
    assert outputs["velocity"].shape == (2, 3)
    assert outputs["pressure"].device.type == device
    assert outputs["velocity"].device.type == device

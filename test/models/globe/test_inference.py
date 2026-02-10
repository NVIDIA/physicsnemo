# ruff: noqa: S101
import pytest
import torch

from physicsnemo.mesh.primitives.procedural import lumpy_sphere
from physicsnemo.models.globe.model import GLOBE

# Number of prediction points to evaluate at
N_PREDICTION_POINTS = 5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_globe_inference(device: str) -> None:
    """Instantiate `GLOBE` and run inference on a lumpy-sphere boundary mesh."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ### Create model
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

    ### Create a nontrivial boundary mesh (lumpy sphere, 1 subdivision -> 80 triangles)
    mesh = lumpy_sphere.load(subdivisions=1, device=device)

    ### Prediction points scattered near the surface
    generator = torch.Generator(device=device).manual_seed(0)
    prediction_points = torch.randn(
        N_PREDICTION_POINTS, 3, generator=generator, device=device
    )
    reference_lengths = {
        "test_length": torch.tensor(1.0, dtype=torch.float32, device=device)
    }

    ### Run inference
    with torch.no_grad():
        outputs = model(
            prediction_points=prediction_points,
            boundary_meshes={"no_slip": mesh},
            reference_lengths=reference_lengths,
            chunk_size=None,
            verbose=False,
        )

    ### Validate output structure and shapes
    assert set(outputs.keys()) == {"pressure", "velocity"}
    assert outputs["pressure"].shape == (N_PREDICTION_POINTS,)
    assert outputs["velocity"].shape == (N_PREDICTION_POINTS, 3)
    assert outputs["pressure"].device.type == device
    assert outputs["velocity"].device.type == device

    ### Validate outputs are finite (no NaN or Inf from the forward pass)
    assert torch.all(torch.isfinite(outputs["pressure"]))
    assert torch.all(torch.isfinite(outputs["velocity"]))

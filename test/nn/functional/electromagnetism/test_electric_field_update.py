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

import pytest
import torch

from physicsnemo.nn.functional.electromagnetism import ElectricFieldUpdate
from test.conftest import requires_module

_MATERIAL_MODES = ("scalar-scalar", "scalar-field", "field-scalar", "field-field")


# Build cloneable inputs for one benchmark case.
def _clone_case(args, kwargs):
    (
        electric_field,
        magnetic_field,
        eps,
        sigma_e,
        spacing,
        dt,
    ) = args

    eps_clone = eps.clone() if isinstance(eps, torch.Tensor) else eps
    sigma_clone = sigma_e.clone() if isinstance(sigma_e, torch.Tensor) else sigma_e

    cloned_args = (
        electric_field.clone(),
        magnetic_field.clone(),
        eps_clone,
        sigma_clone,
        spacing.clone(),
        dt,
    )
    cloned_kwargs = dict(kwargs)
    impressed_current = cloned_kwargs.get("impressed_current")
    if isinstance(impressed_current, torch.Tensor):
        cloned_kwargs["impressed_current"] = impressed_current.clone()
    return cloned_args, cloned_kwargs


# Build deterministic test inputs for explicit material/current variants.
def _build_case(
    device: str,
    material_mode: str,
    use_current: bool,
    grid_n: int = 10,
    seed: int = 2026,
):
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(seed)

    electric_field = torch.randn(
        3,
        grid_n,
        grid_n,
        grid_n,
        generator=generator,
        device=torch_device,
        dtype=torch.float32,
    )
    magnetic_field = torch.randn(
        3,
        grid_n,
        grid_n,
        grid_n,
        generator=generator,
        device=torch_device,
        dtype=torch.float32,
    )

    eps_field = torch.empty(
        grid_n,
        grid_n,
        grid_n,
        device=torch_device,
        dtype=torch.float32,
    ).uniform_(1.0, 6.0, generator=generator)
    sigma_field = torch.empty(
        grid_n,
        grid_n,
        grid_n,
        device=torch_device,
        dtype=torch.float32,
    ).uniform_(0.0, 0.06, generator=generator)

    if material_mode == "scalar-scalar":
        eps = 2.5
        sigma_e = 0.01
    elif material_mode == "scalar-field":
        eps = 2.5
        sigma_e = sigma_field
    elif material_mode == "field-scalar":
        eps = eps_field
        sigma_e = 0.01
    elif material_mode == "field-field":
        eps = eps_field
        sigma_e = sigma_field
    else:
        raise ValueError(f"unsupported material mode: {material_mode}")

    spacing = torch.tensor(
        [0.010, 0.011, 0.013],
        device=torch_device,
        dtype=torch.float32,
    )
    dt = 0.00125

    kwargs = {
        "impressed_current": None,
        "impressed_current_offset": (-2, 1, 3),
        "inplace": False,
    }
    if use_current:
        current_n = max(grid_n // 2, 2)
        kwargs["impressed_current"] = torch.randn(
            3,
            current_n,
            current_n,
            current_n,
            generator=generator,
            device=torch_device,
            dtype=torch.float32,
        )

    args = (
        electric_field,
        magnetic_field,
        eps,
        sigma_e,
        spacing,
        dt,
    )
    return args, kwargs


# Validate warp and torch parity for forward and backward passes.
@requires_module("warp")
def test_electric_field_update_forward_backward_parity(device: str):
    args_torch, kwargs_torch = _build_case(
        device=device,
        material_mode="field-field",
        use_current=True,
        grid_n=10,
        seed=2026,
    )
    args_warp, kwargs_warp = _clone_case(args_torch, kwargs_torch)

    electric_torch = args_torch[0].detach().requires_grad_(True)
    magnetic_torch = args_torch[1].detach().requires_grad_(True)
    electric_warp = args_warp[0].detach().requires_grad_(True)
    magnetic_warp = args_warp[1].detach().requires_grad_(True)

    args_torch = (electric_torch, magnetic_torch) + args_torch[2:]
    args_warp = (electric_warp, magnetic_warp) + args_warp[2:]

    if isinstance(kwargs_torch.get("impressed_current"), torch.Tensor):
        kwargs_torch["impressed_current"] = (
            kwargs_torch["impressed_current"].detach().requires_grad_(True)
        )
        kwargs_warp["impressed_current"] = (
            kwargs_warp["impressed_current"].detach().requires_grad_(True)
        )

    out_torch = ElectricFieldUpdate.dispatch(
        *args_torch,
        implementation="torch",
        **kwargs_torch,
    )
    out_warp = ElectricFieldUpdate.dispatch(
        *args_warp,
        implementation="warp",
        **kwargs_warp,
    )

    ElectricFieldUpdate.compare(out_warp, out_torch)

    grad_output = torch.randn_like(out_torch)
    out_torch.backward(grad_output)
    out_warp.backward(grad_output)

    torch.testing.assert_close(
        electric_warp.grad,
        electric_torch.grad,
        atol=8e-5,
        rtol=1e-4,
        msg=f"electric grad mismatch on '{device}'",
    )
    torch.testing.assert_close(
        magnetic_warp.grad,
        magnetic_torch.grad,
        atol=8e-5,
        rtol=1e-4,
        msg=f"magnetic grad mismatch on '{device}'",
    )

    imp_torch = kwargs_torch.get("impressed_current")
    imp_warp = kwargs_warp.get("impressed_current")
    if isinstance(imp_torch, torch.Tensor):
        assert isinstance(imp_warp, torch.Tensor)
        torch.testing.assert_close(
            imp_warp.grad,
            imp_torch.grad,
            atol=8e-5,
            rtol=1e-4,
            msg=f"current grad mismatch on '{device}'",
        )


# Validate warp and torch parity for all material/current variants.
@requires_module("warp")
@pytest.mark.parametrize("material_mode", _MATERIAL_MODES)
@pytest.mark.parametrize("use_current", [False, True])
def test_electric_field_update_forward_backward_parity_all_variants(
    device: str,
    material_mode: str,
    use_current: bool,
):
    args_torch, kwargs_torch = _build_case(
        device=device,
        material_mode=material_mode,
        use_current=use_current,
        grid_n=8,
        seed=1337,
    )
    args_warp, kwargs_warp = _clone_case(args_torch, kwargs_torch)

    electric_torch = args_torch[0].detach().requires_grad_(True)
    magnetic_torch = args_torch[1].detach().requires_grad_(True)
    electric_warp = args_warp[0].detach().requires_grad_(True)
    magnetic_warp = args_warp[1].detach().requires_grad_(True)

    eps_torch = args_torch[2]
    eps_warp = args_warp[2]
    if isinstance(eps_torch, torch.Tensor):
        eps_torch = eps_torch.detach().requires_grad_(True)
        eps_warp = eps_warp.detach().requires_grad_(True)

    sigma_torch = args_torch[3]
    sigma_warp = args_warp[3]
    if isinstance(sigma_torch, torch.Tensor):
        sigma_torch = sigma_torch.detach().requires_grad_(True)
        sigma_warp = sigma_warp.detach().requires_grad_(True)

    args_torch = (
        electric_torch,
        magnetic_torch,
        eps_torch,
        sigma_torch,
        args_torch[4],
        args_torch[5],
    )
    args_warp = (
        electric_warp,
        magnetic_warp,
        eps_warp,
        sigma_warp,
        args_warp[4],
        args_warp[5],
    )

    if isinstance(kwargs_torch.get("impressed_current"), torch.Tensor):
        kwargs_torch["impressed_current"] = (
            kwargs_torch["impressed_current"].detach().requires_grad_(True)
        )
        kwargs_warp["impressed_current"] = (
            kwargs_warp["impressed_current"].detach().requires_grad_(True)
        )

    out_torch = ElectricFieldUpdate.dispatch(
        *args_torch,
        implementation="torch",
        **kwargs_torch,
    )
    out_warp = ElectricFieldUpdate.dispatch(
        *args_warp,
        implementation="warp",
        **kwargs_warp,
    )
    ElectricFieldUpdate.compare(out_warp, out_torch)

    grad_output = torch.randn_like(out_torch)
    out_torch.backward(grad_output)
    out_warp.backward(grad_output)

    torch.testing.assert_close(
        electric_warp.grad,
        electric_torch.grad,
        atol=8e-5,
        rtol=1e-4,
        msg=f"electric grad mismatch for mode='{material_mode}' current={use_current}",
    )
    torch.testing.assert_close(
        magnetic_warp.grad,
        magnetic_torch.grad,
        atol=8e-5,
        rtol=1e-4,
        msg=f"magnetic grad mismatch for mode='{material_mode}' current={use_current}",
    )

    if isinstance(eps_torch, torch.Tensor):
        assert isinstance(eps_warp, torch.Tensor)
        torch.testing.assert_close(
            eps_warp.grad,
            eps_torch.grad,
            atol=8e-5,
            rtol=1e-4,
            msg=f"eps grad mismatch for mode='{material_mode}' current={use_current}",
        )
    if isinstance(sigma_torch, torch.Tensor):
        assert isinstance(sigma_warp, torch.Tensor)
        torch.testing.assert_close(
            sigma_warp.grad,
            sigma_torch.grad,
            atol=8e-5,
            rtol=1e-4,
            msg=f"sigma grad mismatch for mode='{material_mode}' current={use_current}",
        )

    imp_torch = kwargs_torch.get("impressed_current")
    imp_warp = kwargs_warp.get("impressed_current")
    if isinstance(imp_torch, torch.Tensor):
        assert isinstance(imp_warp, torch.Tensor)
        torch.testing.assert_close(
            imp_warp.grad,
            imp_torch.grad,
            atol=8e-5,
            rtol=1e-4,
            msg=f"current grad mismatch for mode='{material_mode}' current={use_current}",
        )


# In-place mode must reject differentiable inputs.
@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_electric_field_update_inplace_requires_grad_guard(
    device: str,
    implementation: str,
):
    args, kwargs = _build_case(
        device=device,
        material_mode="scalar-scalar",
        use_current=False,
        grid_n=8,
        seed=707,
    )

    electric = args[0].detach().requires_grad_(True)
    args = (electric,) + args[1:]
    kwargs["inplace"] = True

    with pytest.raises(ValueError, match="requires gradients"):
        ElectricFieldUpdate.dispatch(
            *args,
            implementation=implementation,
            **kwargs,
        )


# In-place mode must reject differentiable material tensors too.
@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_electric_field_update_inplace_requires_grad_guard_materials(
    device: str,
    implementation: str,
):
    args, kwargs = _build_case(
        device=device,
        material_mode="scalar-scalar",
        use_current=False,
        grid_n=8,
        seed=808,
    )

    eps = args[2]
    if not isinstance(eps, torch.Tensor):
        nx, ny, nz = args[0].shape[1:]
        eps = torch.full((nx, ny, nz), 2.0, device=args[0].device, dtype=torch.float32)
    eps = eps.detach().requires_grad_(True)
    args = args[:2] + (eps,) + args[3:]
    kwargs["inplace"] = True

    with pytest.raises(ValueError, match="requires gradients"):
        ElectricFieldUpdate.dispatch(
            *args,
            implementation=implementation,
            **kwargs,
        )


# Spacing gradients are intentionally unsupported for this functional.
@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_electric_field_update_spacing_grad_rejected(
    device: str,
    implementation: str,
):
    args, kwargs = _build_case(
        device=device,
        material_mode="field-field",
        use_current=False,
        grid_n=8,
        seed=111,
    )

    spacing = args[4].detach().requires_grad_(True)
    args = args[:4] + (spacing,) + args[5:]

    with pytest.raises(ValueError, match="spacing gradients are not supported"):
        ElectricFieldUpdate.dispatch(
            *args,
            implementation=implementation,
            **kwargs,
        )


# In-place and out-of-place should match numerically when gradients are disabled.
@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_electric_field_update_inplace_matches_out_of_place(
    device: str,
    implementation: str,
):
    args, kwargs = _build_case(
        device=device,
        material_mode="field-field",
        use_current=True,
        grid_n=10,
        seed=909,
    )

    kwargs["inplace"] = False
    out_ref = ElectricFieldUpdate.dispatch(
        *args,
        implementation=implementation,
        **kwargs,
    )

    args_inplace, kwargs_inplace = _clone_case(args, kwargs)
    kwargs_inplace["inplace"] = True
    electric_inplace = args_inplace[0]
    out_inplace = ElectricFieldUpdate.dispatch(
        *args_inplace,
        implementation=implementation,
        **kwargs_inplace,
    )

    assert (
        out_inplace.data_ptr() == electric_inplace.data_ptr()
    ), f"in-place path did not return the input tensor for '{implementation}'"
    torch.testing.assert_close(
        out_inplace,
        out_ref,
        atol=5e-5,
        rtol=1e-4,
        msg=f"in-place mismatch on '{device}'",
    )


# In-place and out-of-place should match for all material/current variants.
@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("material_mode", _MATERIAL_MODES)
@pytest.mark.parametrize("use_current", [False, True])
def test_electric_field_update_inplace_matches_out_of_place_all_variants(
    device: str,
    implementation: str,
    material_mode: str,
    use_current: bool,
):
    args, kwargs = _build_case(
        device=device,
        material_mode=material_mode,
        use_current=use_current,
        grid_n=10,
        seed=777,
    )
    args, kwargs = _clone_case(args, kwargs)

    kwargs["inplace"] = False
    out_ref = ElectricFieldUpdate.dispatch(
        *args,
        implementation=implementation,
        **kwargs,
    )

    args_inplace, kwargs_inplace = _clone_case(args, kwargs)
    kwargs_inplace["inplace"] = True
    electric_inplace = args_inplace[0]
    out_inplace = ElectricFieldUpdate.dispatch(
        *args_inplace,
        implementation=implementation,
        **kwargs_inplace,
    )

    assert out_inplace.data_ptr() == electric_inplace.data_ptr()
    torch.testing.assert_close(
        out_inplace,
        out_ref,
        atol=5e-5,
        rtol=1e-4,
        msg=(
            "in-place mismatch for "
            f"impl='{implementation}', mode='{material_mode}', current={use_current}"
        ),
    )


# Material tensors with shape (1, nx, ny, nz) should be accepted.
@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("use_current", [False, True])
def test_electric_field_update_channel_dim_materials(
    device: str,
    implementation: str,
    use_current: bool,
):
    args, kwargs = _build_case(
        device=device,
        material_mode="field-field",
        use_current=use_current,
        grid_n=8,
        seed=31415,
    )
    eps = args[2]
    sigma_e = args[3]
    assert isinstance(eps, torch.Tensor)
    assert isinstance(sigma_e, torch.Tensor)

    args_3d = args
    args_4d = (args[0], args[1], eps.unsqueeze(0), sigma_e.unsqueeze(0), args[4], args[5])

    out_3d = ElectricFieldUpdate.dispatch(
        *args_3d,
        implementation=implementation,
        **kwargs,
    )
    out_4d = ElectricFieldUpdate.dispatch(
        *args_4d,
        implementation=implementation,
        **kwargs,
    )
    torch.testing.assert_close(out_4d, out_3d, atol=5e-5, rtol=1e-4)


# A fully out-of-bounds current offset should match the no-current branch.
@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("material_mode", _MATERIAL_MODES)
def test_electric_field_update_out_of_bounds_current_offset(
    device: str,
    implementation: str,
    material_mode: str,
):
    args, kwargs = _build_case(
        device=device,
        material_mode=material_mode,
        use_current=True,
        grid_n=8,
        seed=27182,
    )
    kwargs_none = dict(kwargs)
    kwargs_none["impressed_current"] = None
    kwargs_none["impressed_current_offset"] = (0, 0, 0)

    kwargs_oob = dict(kwargs)
    kwargs_oob["impressed_current_offset"] = (10_000, -10_000, 20_000)

    out_none = ElectricFieldUpdate.dispatch(
        *args,
        implementation=implementation,
        **kwargs_none,
    )
    out_oob = ElectricFieldUpdate.dispatch(
        *args,
        implementation=implementation,
        **kwargs_oob,
    )
    torch.testing.assert_close(out_oob, out_none, atol=5e-5, rtol=1e-4)


# Warp path should reject non-contiguous tensors.
@requires_module("warp")
@pytest.mark.parametrize(
    "target", ["electric", "magnetic", "eps", "sigma", "spacing", "current"]
)
def test_electric_field_update_warp_requires_contiguous(
    device: str,
    target: str,
):
    args, kwargs = _build_case(
        device=device,
        material_mode="field-field",
        use_current=True,
        grid_n=8,
        seed=16180,
    )
    electric_field, magnetic_field, eps, sigma_e, spacing, dt = args
    assert isinstance(eps, torch.Tensor)
    assert isinstance(sigma_e, torch.Tensor)
    assert isinstance(kwargs["impressed_current"], torch.Tensor)

    if target == "electric":
        electric_field = electric_field.transpose(1, 2)
    elif target == "magnetic":
        magnetic_field = magnetic_field.transpose(1, 2)
    elif target == "eps":
        eps = eps.transpose(0, 1)
    elif target == "sigma":
        sigma_e = sigma_e.transpose(0, 1)
    elif target == "spacing":
        spaced = torch.empty(6, device=spacing.device, dtype=spacing.dtype)
        spaced[::2] = spacing
        spacing = spaced[::2]
    elif target == "current":
        kwargs["impressed_current"] = kwargs["impressed_current"].transpose(1, 2)

    with pytest.raises(ValueError, match="contiguous"):
        ElectricFieldUpdate.dispatch(
            electric_field,
            magnetic_field,
            eps,
            sigma_e,
            spacing,
            dt,
            implementation="warp",
            **kwargs,
        )

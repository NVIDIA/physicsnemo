Differentiable Deformation Energies
===================================

.. currentmodule:: physicsnemo.mesh.deformation

Deformation energies turn geometric requirements into differentiable scalar
objectives. They can regularize a deformation predicted by a model or produced
by an optimizer while gradients continue to the deformed coordinates and the
parameters that generated them.

These functions provide penalty terms, not a deformation solver or exact
constraint mechanism. A finite penalty weight does not guarantee that an area,
volume, or inversion limit is satisfied exactly. The caller chooses an
optimizer, combines the terms with the application objective, and verifies the
result after optimization. Connectivity remains fixed throughout. Topology
changes and collision handling are outside this API.

The mesh-level functions accept a reference
:class:`~physicsnemo.mesh.mesh.Mesh` and a tensor of current point coordinates.
The reference mesh supplies both the rest coordinates and the fixed topology.
Reference and current coordinates must use matching ``torch.float32`` or
``torch.float64`` dtypes.

Choosing an Energy
------------------

``simplex_strain_energy``
    A dimensionally generic St. Venant--Kirchhoff strain energy for edges,
    triangles, and tetrahedra. It penalizes
    metric distortion relative to the reference configuration and is invariant
    to rigid motion. Metric strain alone cannot distinguish a reflection from a
    rotation, so use ``simplex_inversion_energy`` as well when orientation must
    be preserved. For an embedded triangle membrane, the Lamé parameters are
    effective two-dimensional coefficients. Plane-stress or plane-strain
    reduction and thickness scaling remain the caller's responsibility.

``simplex_measure_energy``
    Penalizes length, area, or volume change in each simplex. This is a local
    regularizer: one expanding element cannot cancel one contracting element.
    Full-dimensional ratios are signed relative to each reference simplex, so
    this term is also sensitive to reflection. Embedded-simplex ratios are
    unsigned.

``total_measure_energy``
    Penalizes only the change in the sum of simplex measures. It preserves a
    total length, area, or volume while allowing measure to move between
    elements. For full-dimensional meshes the sum is an algebraic signed
    measure relative to the reference cells. It does not prevent a local
    collapse or cancellation between inverted and non-inverted cells. The
    reference mesh must contain at least one simplex.

``simplex_inversion_energy``
    Penalizes signed Jacobian ratios below ``minimum_jacobian``. It applies to
    full-dimensional oriented simplices, such as triangles in 2D and
    tetrahedra in 3D. This determinant-based term is not defined for embedded
    simplices and rejects them.

``closed_surface_volume_energy``
    Penalizes the enclosed-volume change of a closed triangular surface in 3D.
    It accepts one edge-connected component in either inward or outward
    orientation, and every edge must have two oppositely directed incident
    triangles. Evaluate disconnected components separately. The mesh wrapper
    validates this contract, but the tensor functional assumes it. The term
    does not test vertex-manifoldness or prevent self-intersection.

``surface_bending_energy``
    Penalizes changes in the dihedral angle at interior edges of a triangular
    surface in 3D. It is a geometric hinge regularizer. It is not a calibrated
    shell constitutive law unless the caller supplies the appropriate material,
    thickness, and discretization scaling outside this function. The mesh must
    contain unique triangles, with at most two incident triangles per edge.
    Boundary edges are omitted because they do not define a hinge.

Local and Total Area Control in 2D
----------------------------------

The following pattern optimizes RBF control displacements while retaining a
target handle motion. The strain term distributes the deformation, the total
measure term limits net area change, and the inversion term discourages
orientation loss.

.. code:: python

    import torch

    from physicsnemo.mesh.deformation import (
        simplex_inversion_energy,
        simplex_strain_energy,
        total_measure_energy,
    )
    from physicsnemo.mesh.primitives.planar import unit_square
    from physicsnemo.nn.functional import radial_basis_function_deform_points

    device = "cuda" if torch.cuda.is_available() else "cpu"
    implementation = "warp" if device == "cuda" else "torch"
    reference = unit_square.load(subdivisions=3).to(device)
    controls = reference.points.new_tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.0, 0.5],
            [0.5, 0.5],
            [0.5, 1.0],
        ]
    )
    auxiliary_displacements = torch.nn.Parameter(
        controls.new_zeros((4, 2))
    )
    target_displacement = controls.new_tensor([0.36, 0.48])
    optimizer = torch.optim.Adam([auxiliary_displacements], lr=3.0e-2)

    for _ in range(350):
        optimizer.zero_grad()
        control_displacements = torch.cat(
            (
                torch.zeros_like(controls[:4]),
                auxiliary_displacements,
                target_displacement.unsqueeze(0),
            )
        )
        points = radial_basis_function_deform_points(
            reference.points,
            controls,
            control_displacements,
            implementation=implementation,
        )
        energy = (
            0.05 * simplex_strain_energy(reference, points)
            + 80.0 * total_measure_energy(reference, points)
            + 500.0
            * simplex_inversion_energy(
                reference,
                points,
                minimum_jacobian=0.4,
            )
        )
        energy.backward()
        optimizer.step()

In a practical design problem, fixed controls or boundary conditions can be
enforced in the parameterization, as above, or included in the objective.

The figure compares the same RBF target with and without geometric penalties.
The third panel is *penalty-regularized*: finite weights reduce, but do not
strictly eliminate, the geometric residuals.

.. figure:: /img/mesh/square_deformation_energies.png
   :alt: Original square and unregularized and penalty-regularized RBF deformations
   :width: 100%

Volume and Bending Control in 3D
--------------------------------

For a closed surface, enclosed-volume and bending terms can be combined while
optimizing the parameters of a global deformation:

.. code:: python

    from physicsnemo.mesh.deformation import (
        closed_surface_volume_energy,
        surface_bending_energy,
    )

    volume_penalty = closed_surface_volume_energy(reference, points)
    bending_penalty = surface_bending_energy(reference, points)
    loss = target_loss + 200.0 * volume_penalty + 0.05 * bending_penalty
    loss.backward()

The volume term controls only the scalar enclosed volume. Bending regularizes
changes between adjacent triangle normals, but neither term detects global
self-intersection.

.. figure:: /img/mesh/sphere_deformation_energies.png
   :alt: Original sphere and unregularized and volume-and-bending-regularized RBF deformations
   :width: 100%

Differentiability and Backends
------------------------------

The Torch backend supports higher-order derivatives. The Warp backend is
intended for first-order optimization on CUDA. Its backward pass scatters
per-element contributions to shared vertices with atomic operations, so the
last bits of a CUDA result may vary between executions. Use the Torch backend
when deterministic higher-order derivatives are required.

Automatic dispatch selects Torch on CPU and Warp on CUDA when Warp is
available. Both backends support gradients with respect to the current and
reference coordinates. Cell and hinge indices are discrete topology and are
not differentiable. Simplex energies in coordinate dimensions above three use
the Torch backend.

The first mesh-wrapper call after topology changes validates index bounds and
connectivity on the host. For a CUDA mesh, that one-time validation synchronizes
the device and is not safe inside CUDA Graph capture. Call the required energy
once before capture to populate the topology cache. The tensor functionals avoid
this synchronization by treating valid index values as part of their input
contract.

Reference simplices, hinge edges, and hinge triangles must be nondegenerate.
Invalid reference geometry produces ``NaN`` instead of receiving an implicit
regularization. A current hinge that collapses also produces ``NaN``. The
wrapped hinge angle is nondifferentiable at a relative fold of plus or minus
pi, as expected for a signed angle branch cut.

Unsigned embedded-simplex measure is also nonsmooth at exact collapse. The
measure energies choose a zero current-coordinate subgradient there. They can
discourage collapse while the simplex has positive measure, but cannot reopen
an already collapsed embedded simplex without another objective term or a
noncollapsed initialization.

API Reference
-------------

.. autofunction:: simplex_strain_energy

.. autofunction:: simplex_measure_energy

.. autofunction:: total_measure_energy

.. autofunction:: simplex_inversion_energy

.. autofunction:: surface_bending_energy

.. autofunction:: closed_surface_volume_energy

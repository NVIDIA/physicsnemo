Shell Contact Functionals
=========================

Contact between thin shells as stateless functions over tensors: discover the
node-to-facet pairs whose shell clearance lies inside an activation band,
encode them as edge features, measure penetration depths, and apply the
evaluation-time geometric corrections (envelope projection, rigid-wall clamp).
The broad phase queries a Warp BVH, the narrow phase runs a fused Warp kernel on
CUDA float32 geometry (with a Torch reference deciding numerically ambiguous
pairs), and the live geometry of the selected pairs is recomputed in Torch so
every output carries autograd through the node positions.

Pair Search
-----------

.. autoclass:: physicsnemo.nn.functional.ShellContactSearch

.. autoclass:: physicsnemo.nn.functional.ShellContactPairs

.. autofunction:: physicsnemo.nn.functional.shell_contact_pairs

.. code:: python

    import torch
    from physicsnemo.nn.functional import ShellContactSearch, shell_contact_pairs

    # One triangular shell in z = 0 and two nodes that started 3 mm above it;
    # node 4 has since passed through the shell.
    reference = torch.tensor(
        [[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [2, 2, 3.0], [3, 2, 3.0]]
    )
    positions = reference.clone()
    positions[3, 2], positions[4, 2] = 1.0, -1.0
    faces = torch.tensor([[0, 1, 2]])
    thickness = torch.full((5,), 0.5)

    pairs = shell_contact_pairs(
        positions,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(
            activation_distance=5.0, implementation="torch", side_from_reference=True
        ),
    )
    pairs.query          # tensor([3, 4])
    pairs.gap            # tensor([0.5, -1.5]): clearance, negative once crossed
    pairs.crossed        # tensor([False, True])

.. autofunction:: physicsnemo.nn.functional.shell_contact.uses_fused_narrow_phase

Edge Features
-------------

.. autofunction:: physicsnemo.nn.functional.shell_contact_edge_features

.. autofunction:: physicsnemo.nn.functional.smooth_contact_cutoff

.. autofunction:: physicsnemo.nn.functional.shell_contact.contact_edge_features

.. autofunction:: physicsnemo.nn.functional.shell_contact.facet_area_quadrature

.. autodata:: physicsnemo.nn.functional.shell_contact.CONTACT_FEATURE_DIM

.. autodata:: physicsnemo.nn.functional.shell_contact.KINEMATIC_CONTACT_FEATURE_DIM

Penetration Depths
------------------

.. autofunction:: physicsnemo.nn.functional.shell_penetration_depths

.. autofunction:: physicsnemo.nn.functional.wall_penetration_depths

Geometric Corrections
---------------------

.. autofunction:: physicsnemo.nn.functional.project_shell_penetrations

.. autofunction:: physicsnemo.nn.functional.clamp_to_wall

.. code:: python

    from physicsnemo.nn.functional import clamp_to_wall, project_shell_penetrations

    corrected = project_shell_penetrations(
        positions,
        faces,
        thickness,
        reference,
        search=ShellContactSearch(
            activation_distance=5.0, implementation="torch", side_from_reference=True
        ),
        iterations=2,
        damping=0.5,
    )
    # Cap the x coordinate at a rigid wall plane x = 100 (admissible side x <= 100).
    corrected = clamp_to_wall(corrected, 100.0, axis=0, sign=1.0)

Geometry Primitives
-------------------

.. autofunction:: physicsnemo.nn.functional.shell_contact.project_onto_facets

.. autofunction:: physicsnemo.nn.functional.shell_contact.reference_facet_projection

.. autofunction:: physicsnemo.nn.functional.shell_contact.facet_side

.. autofunction:: physicsnemo.nn.functional.shell_contact.crossing_flags

.. autofunction:: physicsnemo.nn.functional.shell_contact.candidate_activation_gaps

.. autofunction:: physicsnemo.nn.functional.shell_contact.linear_closest_approach

.. autofunction:: physicsnemo.nn.functional.shell_contact.node_maximum

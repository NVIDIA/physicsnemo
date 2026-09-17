Visualization
=============

.. currentmodule:: physicsnemo.mesh.visualization

PhysicsNeMo-Mesh supports two visualization backends:

**Matplotlib**
    2D and 3D static plots. The default backend when PyVista is not available or
    when rendering in non-interactive environments (for example, Jupyter notebooks
    without a GPU display).

**PyVista**
    Interactive 3D rendering with hardware acceleration. Preferred for
    exploratory visualization of large meshes.

The primary entry point is :func:`draw`, which is also accessible as the
``mesh.draw()`` method. The backend is selected automatically based on
availability, or can be specified explicitly.

Scalar data can be visualized as colormapped overlays on points or cells.
For vector fields, the L2 norm is computed automatically for colormapping.

.. code:: python

    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)
    mesh.point_data["height"] = mesh.points[:, 2]

    # Quick visualization via Mesh method
    mesh.draw(point_scalars="height", show_edges=True)

    # Standalone function
    from physicsnemo.mesh.visualization import draw
    draw(mesh, point_scalars="height")

``draw_mesh`` remains available as a pending-deprecation compatibility name.

API Reference
-------------

.. automodule:: physicsnemo.mesh.visualization
   :members:
   :show-inheritance:

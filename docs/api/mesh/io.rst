I/O
===

.. currentmodule:: physicsnemo.mesh.io

Conversion between PhysicsNeMo :class:`~physicsnemo.mesh.mesh.Mesh` objects and
`PyVista <https://pyvista.org/>`_ meshes. Because PyVista supports a wide range of
file formats (VTK, STL, PLY, OBJ, and many others), this module serves as the
primary I/O gateway for PhysicsNeMo-Mesh.

:func:`from_pyvista`
    Convert a ``pyvista.PolyData`` or ``pyvista.UnstructuredGrid`` to a
    :class:`~physicsnemo.mesh.mesh.Mesh`. Point data and cell data arrays are
    carried over.

:func:`to_pyvista`
    Convert a :class:`~physicsnemo.mesh.mesh.Mesh` to a ``pyvista.PolyData``
    (for surface meshes) or ``pyvista.UnstructuredGrid`` (for volume meshes).

.. code:: python

    import pyvista as pv
    from physicsnemo.mesh.io import from_pyvista, to_pyvista

    # Load any format PyVista supports
    pv_mesh = pv.read("geometry.stl")
    mesh = from_pyvista(pv_mesh)

    # Work with the mesh in PhysicsNeMo...
    mesh = mesh.subdivide(levels=1, filter="loop")

    # Export back to PyVista for saving or visualization
    pv_out = to_pyvista(mesh)
    pv_out.save("refined.vtk")

API Reference
-------------

.. automodule:: physicsnemo.mesh.io
   :members:
   :show-inheritance:

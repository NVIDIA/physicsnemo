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

.. tip::

   PyVista conversions preserve ``float32`` and ``float64`` point coordinates.
   Integer, complex, and reduced-precision coordinates are converted to
   ``float32`` as part of PhysicsNeMo's PyVista compatibility policy. To
   normalize imported geometry and its floating data explicitly, use
   ``mesh = from_pyvista(pv_mesh).to(torch.float32)``. Likewise, use
   ``to_pyvista(mesh.to(torch.float32))`` before export. Retaining ``float64``
   coordinates doubles their storage relative to ``float32`` and may keep
   downstream PyVista computations in double precision.

API Reference
-------------

.. automodule:: physicsnemo.mesh.io
   :members:
   :show-inheritance:

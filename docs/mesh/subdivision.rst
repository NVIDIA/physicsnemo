Subdivision
===========

.. currentmodule:: physicsnemo.mesh.subdivision

Mesh subdivision refines a mesh by splitting each n-simplex into
:math:`2^n` child simplices (for example, each triangle becomes four triangles).
New vertices are inserted at or near edge midpoints, and existing data
is interpolated onto the refined mesh.

Three schemes are available:

**Linear** (midpoint)
    New vertices are placed at exact edge midpoints. This is an `interpolating`
    scheme, in that the original vertices can be found, without repositioning,
    in the subdivided mesh. This scheme is also the fastest and most
    dimensionally-generic of the provided schemes. However, it does not provide
    any smoothing. This can result in visible `meta-facets` in the subdivided
    mesh, each of which corresponds to an original cell of the parent mesh.

**Loop** (`Loop, 1987 <https://www.microsoft.com/en-us/research/publication/smooth-subdivision-surfaces-based-on-triangles/>`_)
    Valence-based weighted averaging produces :math:`C^2`-smooth limit surfaces.
    This is an `approximating` scheme, in that the original vertices are
    repositioned. Hence, the subdivided mesh will not contain the original
    vertices. This scheme is the standard choice for generating smooth surfaces
    from coarse meshes.

**Butterfly** (`Zorin et al., 1996 <https://cims.nyu.edu/gcl/papers/zorin1996ism.pdf>`_)
    Weighted stencil subdivision that is interpolating (original vertices stay
    fixed) while still producing smooth surfaces. In practice, this reduced
    geometric flexibility compared to the Loop scheme can result in less robust
    performance across varying mesh topologies. Preferred when exact
    interpolation of existing data is required.

All schemes propagate ``point_data`` and ``cell_data`` to the refined mesh
via appropriate interpolation.

.. code:: python

    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=0)

    # Via standalone functions
    from physicsnemo.mesh.subdivision import subdivide_linear, subdivide_loop
    refined = subdivide_linear(mesh)
    smooth = subdivide_loop(mesh)

    # Via the Mesh method
    refined = mesh.subdivide(levels=2, filter="linear")
    smooth = mesh.subdivide(levels=2, filter="loop")
    interp = mesh.subdivide(levels=2, filter="butterfly")

API Reference
-------------

.. automodule:: physicsnemo.mesh.subdivision
   :members:
   :show-inheritance:

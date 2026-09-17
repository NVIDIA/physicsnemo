Rendering Functionals
=====================

Rendering functionals convert tensor fields and geometric primitives into image
buffers. They follow the same stateless functional pattern as the rest of
``physicsnemo.nn.functional``: tensors in, tensors out, with implementation
dispatch handled through ``FunctionSpec``.

Isosurface Render
-----------------

.. autofunction:: physicsnemo.nn.functional.isosurface_render

.. rubric:: Visualization

This animation ray-marches a moving sphere isosurface from a scalar field and
colors the hit surface with an RGB volume.

.. figure:: /img/nn/functional/rendering/isosurface_render.gif
   :alt: Isosurface render animation of a moving sphere
   :width: 55%

Mesh Raycast
------------

.. autofunction:: physicsnemo.nn.functional.mesh_raycast

.. rubric:: Visualization

This animation renders a rotating cube mesh with per-vertex colors.

.. figure:: /img/nn/functional/rendering/mesh_raycast.gif
   :alt: Mesh raycast animation of a rotating colored cube
   :width: 55%

Scalar Field To RGBA
--------------------

.. autofunction:: physicsnemo.nn.functional.scalar_field_to_rgba

Line Integral Convolution
-------------------------

.. autofunction:: physicsnemo.nn.functional.line_integral_convolution

.. rubric:: Visualization

This animation shows a zoomed-out center slice through a 3D LIC field computed
from a rotating dipole vector field. The LIC texture modulates a jet-colored
field-magnitude image after starting from fixed random noise.

.. figure:: /img/nn/functional/rendering/line_integral_convolution.gif
   :alt: Line integral convolution animation of a rotating dipole field
   :width: 55%

This animation renders a steady 3D dipole LIC field as an RGBA volume with
``volume_render`` and overlays a rotating wireframe cube for spatial context.

.. figure:: /img/nn/functional/rendering/line_integral_convolution_3d.gif
   :alt: Three-dimensional line integral convolution volume render with rotating cube
   :width: 55%

Vector Field To RGBA
--------------------

.. autofunction:: physicsnemo.nn.functional.vector_field_to_rgba

Volume Render
-------------

.. autofunction:: physicsnemo.nn.functional.volume_render

Point Cloud Render
------------------

.. autofunction:: physicsnemo.nn.functional.point_cloud_render

Wireframe Render
----------------

.. autofunction:: physicsnemo.nn.functional.wireframe_render

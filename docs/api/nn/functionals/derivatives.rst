Derivative Functionals
======================

The figures compare representative inputs (left) with functional outputs
(right). The mesh examples use the PyTorch implementation on the same
deterministic irregular triangular mesh.

.. note::

   The ``rectilinear_grid_divergence``, ``rectilinear_grid_curl``, and
   ``rectilinear_grid_laplacian`` functionals support periodic boundary
   conditions only.

   Where available, Warp implementations of the derivative functionals on this
   page compute internally in ``float32``. Non-``float32`` floating-point
   inputs are cast to ``float32`` for computation, and outputs are cast back to
   the input field or value dtype. Consequently, ``float64`` outputs retain
   their dtype but are limited to ``float32`` numerical precision.

.. contents:: On this page
   :local:
   :depth: 1

Gradient
--------

.. autofunction:: physicsnemo.nn.functional.uniform_grid_gradient

.. figure:: /img/nn/functional/derivatives/uniform_grid_gradient.png
   :alt: Uniform grid gradient example
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.rectilinear_grid_gradient

.. figure:: /img/nn/functional/derivatives/rectilinear_grid_gradient.png
   :alt: Rectilinear grid gradient example
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.rectilinear_grid_divergence

.. figure:: /img/nn/functional/derivatives/rectilinear_grid_divergence.png
   :alt: Rectilinear grid divergence example
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.rectilinear_grid_curl

.. figure:: /img/nn/functional/derivatives/rectilinear_grid_curl.png
   :alt: Rectilinear grid curl example
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.rectilinear_grid_laplacian

.. figure:: /img/nn/functional/derivatives/rectilinear_grid_laplacian.png
   :alt: Rectilinear grid Laplacian example
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.mesh_lsq_gradient

.. figure:: /img/nn/functional/derivatives/mesh_lsq_gradient.png
   :alt: Mesh LSQ gradient example
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.mesh_green_gauss_gradient

.. figure:: /img/nn/functional/derivatives/mesh_green_gauss_gradient.png
   :alt: Mesh Green-Gauss gradient example
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.spectral_grid_gradient

.. figure:: /img/nn/functional/derivatives/spectral_grid_gradient.png
   :alt: Spectral grid gradient example
   :width: 100%

Mesh Divergence
---------------

.. autofunction:: physicsnemo.nn.functional.mesh_lsq_divergence

.. figure:: /img/nn/functional/derivatives/mesh_lsq_divergence.png
   :alt: LSQ mesh divergence of a vector field on an irregular triangular mesh
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.mesh_cotan_divergence

.. figure:: /img/nn/functional/derivatives/mesh_cotan_divergence.png
   :alt: Cotangent DEC divergence of a vector field on an irregular triangular mesh
   :width: 100%

Mesh Curl
---------

.. autofunction:: physicsnemo.nn.functional.mesh_lsq_curl

.. figure:: /img/nn/functional/derivatives/mesh_lsq_curl.png
   :alt: LSQ curl of a vortex field on an irregular triangular mesh
   :width: 100%

Mesh Laplacian
--------------

.. autofunction:: physicsnemo.nn.functional.mesh_lsq_laplacian

.. figure:: /img/nn/functional/derivatives/mesh_lsq_laplacian.png
   :alt: Double-LSQ Laplacian of a scalar field on an irregular triangular mesh
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.mesh_cotan_laplacian

.. figure:: /img/nn/functional/derivatives/mesh_cotan_laplacian.png
   :alt: Cotangent Laplacian of a scalar field on an irregular triangular mesh
   :width: 100%

Other Derivatives
-----------------

.. autofunction:: physicsnemo.nn.functional.meshless_fd_derivatives

Uniform Grid Vector Calculus
----------------------------

.. note::

   The ``uniform_grid_divergence``, ``uniform_grid_curl``, and
   ``uniform_grid_laplacian`` functionals support periodic boundary conditions
   only.

   Their Warp implementations compute internally in ``float32``. Non-``float32``
   floating-point inputs are cast to ``float32`` for computation, and outputs
   are cast back to the input field or value dtype. Consequently, ``float64``
   outputs retain their dtype but are limited to ``float32`` numerical precision.

.. autofunction:: physicsnemo.nn.functional.uniform_grid_divergence

.. figure:: /img/nn/functional/derivatives/uniform_grid_divergence.png
   :alt: Uniform grid divergence example
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.uniform_grid_curl

.. figure:: /img/nn/functional/derivatives/uniform_grid_curl.png
   :alt: Uniform grid curl example
   :width: 100%

.. autofunction:: physicsnemo.nn.functional.uniform_grid_laplacian

.. figure:: /img/nn/functional/derivatives/uniform_grid_laplacian.png
   :alt: Uniform grid Laplacian example
   :width: 100%

.. _diffusion_multi_diffusion:

Multi-Diffusion
===============

.. currentmodule:: physicsnemo.diffusion.multi_diffusion

.. note::

    The multi-diffusion (patch-based diffusion) utilities are under active
    development and will be released in an upcoming version of PhysicsNeMo.

    Multi-diffusion is a technique for scaling diffusion models to large spatial
    domains by splitting the full domain into overlapping patches, running the
    diffusion model on each patch independently (potentially in parallel), and
    fusing the generated patches back into a coherent full-domain output.  This
    is particularly useful for problems where the full domain does not fit into
    GPU memory.

    For a preview of the approach, see the positional embedding architectures
    :class:`~physicsnemo.models.diffusion_unets.SongUNetPosEmbd` and
    :class:`~physicsnemo.models.diffusion_unets.SongUNetPosLtEmbd` in the
    :ref:`Diffusion UNets documentation <diffusion_unets>`, which provide the
    backbone support for patch-based methods.  The
    `CorrDiff example <../../examples/weather/corrdiff/README.rst>`_ demonstrates
    a full application of patch-based diffusion for weather downscaling.

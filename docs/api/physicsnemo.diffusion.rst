PhysicsNeMo Diffusion
=====================

Patching Utils
--------------

Patching utilities are particularly useful for *patch-based* diffusion, also called
*multi-diffusion*. This approach is used to scale diffusion to very large images.
The following patching utilities extract patches from 2D images, and typically gather
them in the batch dimension. A batch of patches is therefore composed of multiple
smaller patches that are extracted from each sample in the original batch of larger
images. Diffusion models can then process these patches independently. These
utilities also support fusing operations to reconstruct the entire predicted
image from the individual predicted patches.

.. automodule:: physicsnemo.diffusion.multi_diffusion
    :members:
    :show-inheritance:

Diffusion Utils
----------------

Tools for working with diffusion models and other generative approaches,
including deterministic and stochastic sampling utilities.

.. automodule:: physicsnemo.diffusion.samplers.deterministic_sampler
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.diffusion.samplers.stochastic_sampler
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.diffusion.utils
    :members:
    :show-inheritance:
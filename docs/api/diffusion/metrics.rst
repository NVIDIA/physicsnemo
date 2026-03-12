.. _diffusion_metrics:

Metrics and Losses
==================

.. currentmodule:: physicsnemo.diffusion.metrics

This module provides two categories of tools: 

* **training losses** for learning diffusion models
* **evaluation metrics** for measuring the quality of generated samples


Training Losses
---------------

The standard training objective for diffusion models is *denoising score
matching* (DSM).  The model is trained to recover clean data from a noisy
version, with the :ref:`noise scheduler <diffusion_noise_schedulers>` handling
time sampling, noise injection, and loss weighting.

:class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss` implements the
MSE-based DSM loss and supports both x0-predictor and score-predictor
training.  :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss`
extends it with an element-wise weight tensor for masking specific spatial
regions or channels (for example, land versus ocean in weather applications).

.. code-block:: python

    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.metrics.losses import MSEDSMLoss

    scheduler = EDMNoiseScheduler()

    # x0-predictor training (default)
    loss_fn = MSEDSMLoss(model, scheduler)

    # Score-predictor training
    loss_fn_score = MSEDSMLoss(
        model, scheduler,
        prediction_type="score",
        score_to_x0_fn=scheduler.score_to_x0,
    )


Evaluation Metrics
------------------

The framework provides evaluation metrics for assessing the quality of
generated samples. The Fréchet Inception Distance (FID) is
available using
:func:`~physicsnemo.diffusion.metrics.fid.calculate_fid_from_inception_stats`,
which computes the FID from precomputed Inception-v3 statistics.


API Reference
-------------

:code:`MSEDSMLoss`
~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.metrics.losses.MSEDSMLoss
    :members:
    :exclude-members: __init__

:code:`WeightedMSEDSMLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss
    :members:
    :exclude-members: __init__

:code:`calculate_fid_from_inception_stats`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: physicsnemo.diffusion.metrics.fid.calculate_fid_from_inception_stats

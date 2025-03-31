PhysicsNeMo Metrics
====================

.. automodule:: physicsnemo.metrics
.. currentmodule:: physicsnemo.metrics

Basics
-------

PhysicsNeMo provides several general and domain-specific metric calculations you can
leverage in your custom training and inference workflows. These metrics are optimized to
operate on PyTorch tensors. 

General Metrics and Statistical Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is a summary of general purpose statistical methods and metrics that are available:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Metric
     - Description
   * - `physicsnemo.metrics.general.mse.mse <#physicsnemo.metrics.general.mse.mse>`_
     - Mean Squared error between two tensors
   * - `physicsnemo.metrics.general.mse.rmse <#physicsnemo.metrics.general.mse.rmse>`_
     - Root Mean Squared error between two tensors
   * - `physicsnemo.metrics.general.histogram.histogram <#physicsnemo.metrics.general.histogram.histogram>`_
     - Histogram of a set of tensors over the leading dimension
   * - `physicsnemo.metrics.general.histogram.cdf <#physicsnemo.metrics.general.histogram.cdf>`_
     - Cumulative density function of a set of tensors over the leading dimension
   * - `physicsnemo.metrics.general.histogram.normal_cdf <#physicsnemo.metrics.general.histogram.normal_cdf>`_
     - Cumulative density function of a normal variable with given mean and standard deviation
   * - `physicsnemo.metrics.general.histogram.normal_pdf <#physicsnemo.metrics.general.histogram.normal_pdf>`_
     - Probability density function of a normal variable with given mean and standard deviation
   * - `physicsnemo.metrics.general.calibration.find_rank <#physicsnemo.metrics.general.calibration.find_rank>`_
     - Find the rank of the observation with respect to the given counts and bins
   * - `physicsnemo.metrics.general.calibration.rank_probability_score <#physicsnemo.metrics.general.calibration.rank_probability_score>`_
     - Rank Probability Score for the passed ranks
   * - `physicsnemo.metrics.general.entropy.entropy_from_counts <#physicsnemo.metrics.general.entropy.entropy_from_counts>`_
     - Computes the statistical entropy of a random variable using a histogram.
   * - `physicsnemo.metrics.general.entropy.relative_entropy_from_counts <#physicsnemo.metrics.general.entropy.relative_entropy_from_counts>`_
     - Computes the relative statistical entropy, or KL Divergence of two random variables using their histograms.
   * - `physicsnemo.metrics.general.crps.crps <#physicsnemo.metrics.general.crps.crps>`_
     - Local Continuous Ranked Probability Score (CRPS) by computing a histogram and CDF of the predictions
   * - `physicsnemo.metrics.general.wasserstein.wasserstein <#physicsnemo.metrics.general.wasserstein.wasserstein>`_
     - 1-Wasserstein distance between two discrete CDF functions
   * - `physicsnemo.metrics.general.reduction.WeightedMean <#physicsnemo.metrics.general.reduction.WeightedMean>`_
     - Weighted Mean
   * - `physicsnemo.metrics.general.reduction.WeightedStatistic <#physicsnemo.metrics.general.reduction.WeightedStatistic>`_
     - Weighted Statistic
   * - `physicsnemo.metrics.general.reduction.WeightedVariance <#physicsnemo.metrics.general.reduction.WeightedVariance>`_
     - Weighted Variance

Below shows some examples of how to use these metrics in your own workflows. 


To compute RMSE metric:

.. code:: python

    >>> import torch
    >>> from physicsnemo.metrics.general.mse import rmse
    >>> pred_tensor = torch.randn(16, 32)
    >>> targ_tensor = torch.randn(16, 32)
    >>> rmse(pred_tensor, targ_tensor)
    tensor(1.4781)


To compute the histogram of samples:

.. code:: python

    >>> import torch
    >>> from physicsnemo.metrics.general import histogram
    >>> x = torch.randn(1_000)
    >>> bins, counts = histogram.histogram(x, bins = 10)
    >>> bins
    tensor([-3.7709, -3.0633, -2.3556, -1.6479, -0.9403, -0.2326,  0.4751,  1.1827,
            1.8904,  2.5980,  3.3057])
    >>> counts
    tensor([  3,   9,  43, 150, 227, 254, 206,  81,  24,   3])
  

To use compute the continuous density function (CDF):

.. code:: python

    >>> bins, cdf = histogram.cdf(x, bins = 10)
    >>> bins
    tensor([-3.7709, -3.0633, -2.3556, -1.6479, -0.9403, -0.2326,  0.4751,  1.1827,
            1.8904,  2.5980,  3.3057])
    >>> cdf
    tensor([0.0030, 0.0120, 0.0550, 0.2050, 0.4320, 0.6860, 0.8920, 0.9730, 0.9970,
            1.0000])

To use the histogram for statistical entropy calculations:

.. code:: python

    >> from physicsnemo.metrics.general import entropy
    >>> entropy.entropy_from_counts(counts, bins)
    tensor(0.4146)

Many of the functions operate over batches. For example, if one has a collection of two dimensional
data, then we can compute the histogram over the collection:

.. code:: python

    >>> import torch
    >>> from physicsnemo.metrics.general import histogram, entropy
    >>> x = torch.randn((1_000, 3, 3))
    >>> bins, counts = histogram.histogram(x, bins = 10)
    >>> bins.shape, counts.shape
    (torch.Size([11, 3, 3]), torch.Size([10, 3, 3]))
    >>> entropy.entropy_from_counts(counts, bins)
    tensor([[0.5162, 0.4821, 0.3976],
            [0.5099, 0.5309, 0.4519],
            [0.4580, 0.4290, 0.5121]])

There are additional metrics to compute differences between distributions: Ranks, Continuous Rank
Probability Skill, and Wasserstein metric.

CRPS:

.. code:: python

    >>> from physicsnemo.metrics.general import crps
    >>> x = torch.randn((1_000,1))
    >>> y = torch.randn((1,))
    >>> crps.crps(x, y)
    tensor([0.8023])

Ranks:

.. code:: python

    >>> from physicsnemo.metrics.general import histogram, calibration
    >>> x = torch.randn((1_000,1))
    >>> y = torch.randn((1,))
    >>> bins, counts = histogram.histogram(x, bins = 10)
    >>> ranks = calibration.find_rank(bins, counts, y)
    tensor([0.1920])

Wasserstein Metric:

.. code:: python 

    >>> from physicsnemo.metrics.general import wasserstein, histogram
    >>> x = torch.randn((1_000,1))
    >>> y = torch.randn((1_000,1))
    >>> bins, cdf_x = histogram.cdf(x)
    >>> bins, cdf_y = histogram.cdf(y, bins = bins)
    >>> wasserstein(bins, cdf_x, cdf_y)
    >>> wasserstein.wasserstein(bins, cdf_x, cdf_y)
    tensor([0.0459])
  

Weighted Reductions
^^^^^^^^^^^^^^^^^^^
PhysicsNeMo currently offers classes for weighted mean and variance reductions.

.. code:: python

    >>> from physicsnemo.metrics.general import reduction
    >>> x = torch.randn((1_000,))
    >>> weights = torch.cos(torch.linspace(-torch.pi/4, torch.pi/4, 1_000))
    >>> wm = reduction.WeightedMean(weights)
    >>> wm(x, dim = 0)
    tensor(0.0365)
    >>> wv = reduction.WeightedVariance(weights)
    >>> wv(x, dim = 0)
    tensor(1.0148)


Online Statistical Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^
PhysicsNeMo current offers routines for computing online, or out-of-memory, means,
variances, and histograms.

.. code:: python 

  >>> import torch
  >>> from physicsnemo.metrics.general import ensemble_metrics as em
  >>> x = torch.randn((1_000, 2)) # Interpret as 1_000 members of size (2,).
  >>> torch.mean(x, dim = 0) # Compute mean of entire data.
  tensor([-0.0545,  0.0267])
  >>> x0, x1 = x[:500], x[500:] # Split data into two.
  >>> M = em.Mean(input_shape = (2,)) # Must pass shape of data
  >>> M(x0) # Compute mean of initial batch.
  tensor([-0.0722,  0.0414])
  >>> M.update(x1) # Update with second batch.
  tensor([-0.0545,  0.0267])


Climate Related Metrics
^^^^^^^^^^^^^^^^^^^^^^^

To compute the Anomaly Correlation Coefficient, a metric widely used in weather and
climate sciences:

.. code:: python

    >>> import torch
    >>> import numpy as np
    >>> from physicsnemo.metrics.climate.acc import acc
    >>> channels = 1
    >>> img_shape = (32, 64)
    >>> time_means = np.pi / 2 * np.ones((channels, img_shape[0], img_shape[1]), dtype=np.float32)
    >>> x = np.linspace(-180, 180, img_shape[1], dtype=np.float32)
    >>> y = np.linspace(-90, 90, img_shape[0], dtype=np.float32)
    >>> xv, yv = np.meshgrid(x, y)
    >>> pred_tensor_np = np.cos(2 * np.pi * yv / (180))
    >>> targ_tensor_np = np.cos(np.pi * yv / (180))
    >>> pred_tensor = torch.from_numpy(pred_tensor_np).expand(channels, -1, -1)
    >>> targ_tensor = torch.from_numpy(targ_tensor_np).expand(channels, -1, -1)
    >>> means_tensor = torch.from_numpy(time_means)
    >>> lat = torch.from_numpy(y)
    >>> acc(pred_tensor, targ_tensor, means_tensor, lat)
    tensor([0.9841])


.. autosummary::
   :toctree: generated

General
---------

.. automodule:: physicsnemo.metrics.general.mse
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.metrics.general.histogram
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.metrics.general.entropy
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.metrics.general.calibration
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.metrics.general.crps
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.metrics.general.ensemble_metrics
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.metrics.general.reduction
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.metrics.general.wasserstein
    :members:
    :show-inheritance:

Weather and climate metrics
---------------------------

.. automodule:: physicsnemo.metrics.climate.acc
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.metrics.climate.efi
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.metrics.climate.reduction
    :members:
    :show-inheritance:



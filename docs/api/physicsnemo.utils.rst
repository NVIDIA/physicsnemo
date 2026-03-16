PhysicsNeMo Utils
==================

.. automodule:: physicsnemo.utils
.. currentmodule:: physicsnemo.utils

The PhysicsNeMo Utils module provides a comprehensive set of utilities that support various aspects of scientific computing,
machine learning, and physics simulations. These utilities range from optimization helpers and distributed computing tools
to specialized functions for weather and climate modeling, and geometry processing. The module is designed to simplify common
tasks while maintaining high performance and scalability.

.. autosummary::
   :toctree: generated

Weather and Climate Utils
--------------------------

Specialized utilities for weather and climate modeling, including calculations for solar radiation
and atmospheric parameters. These utilities are used extensively in weather prediction models.

.. automodule:: physicsnemo.utils.insolation
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.zenith_angle
    :show-inheritance:

.. _patching_utils:


Checkpointing
-------------

.. automodule:: physicsnemo.utils.checkpoint
    :members:
    :show-inheritance:


Profiling Utils
---------------

Utilities for profiling the performance of a model.

.. automodule:: physicsnemo.utils.profiling
    :members:
    :show-inheritance:


Optimization Utils
------------------

The optimization utilities provide tools for capturing and managing training states, gradients, and optimization processes.
These are particularly useful when implementing custom training loops or specialized optimization strategies.

.. automodule:: physicsnemo.utils.capture
    :members:
    :show-inheritance:


PhysicsNeMo Logging
===================

.. automodule:: physicsnemo.utils.logging
.. currentmodule:: physicsnemo.utils.logging

The PhysicsNeMo Logging module provides a comprehensive and flexible logging system for machine learning experiments
and physics simulations. It offers multiple logging backends including console output, MLflow, and Weights & Biases (W&B),
allowing users to track metrics, artifacts, and experiment parameters across different platforms. The module is designed to
work seamlessly in both single-process and distributed training environments.

Key Features:
- Unified logging interface across different backends
- Support for distributed training environments
- Automatic metric aggregation and synchronization
- Flexible configuration and customization options
- Integration with popular experiment tracking platforms

Consider the following example usage:

.. code:: python

    from physicsnemo.utils.logging import LaunchLogger
    
    # Initialize the logger
    logger = LaunchLogger.initialize(use_mlflow=True)

    # Training loop
    for epoch in range(num_epochs):

        # Training logger
        with LaunchLogger(
            "train", epoch = epoch, num_mini_batch = len(training_datapipe), epoch_alert_freq = 1
        ) as logger:
            for batch in training_datapipe:
                # Training loop
                ... # training code
                logger.log_metrics({"train_loss": training_loss})

        # Validation logger
        with LaunchLogger(
            "val", epoch = epoch, num_mini_batch = len(validation_datapipe), epoch_alert_freq = 1
        ) as logger:
            for batch in validation_datapipe:
                # Validation loop
                ... # validation code
                logger.log_minibatch({"val_loss": validation_loss})
        
        learning_rate = ... # get the learning rate at the end of the epoch from the optimizer
        logger.log_epoch({"learning_rate": learning_rate}) # log the learning rate at the end of the epoch

This example shows how to use the LaunchLogger to log metrics during training and
validation. The LaunchLogger is initialized with the MLflow backend, and the logger
is created for each epoch, a separate logger is created for training and validation.
You can use the `.log_minibatch` method to log metrics during training and validation.
You can use the `.log_epoch` method to log the learning rate at the end of the epoch.

For a more detailed example, refer to the `Logging and Checkpointing recipe <../../user-guide/simple_logging_and_checkpointing.html>`_ .

.. autosummary::
   :toctree: generated

Launch Logger
-------------

The LaunchLogger serves as the primary interface for logging in PhysicsNeMo. It provides a unified API that works
consistently across different logging backends and training environments. The logger automatically handles metric
aggregation in distributed settings and ensures proper synchronization across processes.

.. automodule:: physicsnemo.utils.logging.launch
    :members:
    :show-inheritance:

Console Logger
--------------

A simple but powerful console-based logger that provides formatted output to the terminal. It's particularly useful
during development and debugging, offering clear visibility into training progress and metrics.

.. automodule:: physicsnemo.utils.logging.console
    :members:
    :show-inheritance:

MLflow Logger
-------------

Integration with MLflow for experiment tracking and model management. This utility enables systematic tracking of
experiments, including metrics, parameters, artifacts, and model versions. It's particularly useful for teams
that need to maintain reproducibility and compare different experiments. Users should initialize the MLflow backend
before using the LaunchLogger.

.. automodule:: physicsnemo.utils.logging.mlflow
    :members:
    :show-inheritance:

Example usage:

.. code:: python

    from physicsnemo.utils.logging.mlflow import initialize_mlflow
    from physicsnemo.utils.logging import LaunchLogger
    
    # Initialize MLflow
    initialize_mlflow(
        experiment_name="weather_prediction",
        user_name="physicsnemo_user",
        mode="offline",
    )
    
    # Create logger with MLflow backend
    logger = LaunchLogger.initialize(use_mlflow=True)

Weights and Biases Logger
-------------------------

Integration with Weights & Biases (W&B) for experiment tracking and visualization. This utility provides rich
visualization capabilities and easy experiment comparison, making it ideal for projects that require detailed
analysis of training runs and model performance. You must initialize the W&B backend before using the LaunchLogger.

.. automodule:: physicsnemo.utils.logging.wandb
    :members:
    :show-inheritance:

Example usage:

.. code:: python

    from physicsnemo.utils.logging.wandb import initialize_wandb
    from physicsnemo.utils.logging import LaunchLogger
    
    # Initialize W&B
    initialize_wandb(
        project="physics_simulation",
        entity="my_team"
    )
    
    # Create logger with W&B backend
    logger = LaunchLogger.initialize(use_wandb=True)

Logging Utils
-------------

Utility functions and helpers for logging operations.

.. automodule:: physicsnemo.utils.logging.utils
    :members:
    :show-inheritance:


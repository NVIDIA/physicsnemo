PhysicsNeMo Launch Logging
===========================

.. automodule:: physicsnemo.launch.logging
.. currentmodule:: physicsnemo.launch.logging

The PhysicsNeMo Launch Logging module provides a comprehensive and flexible logging system for machine learning experiments
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

    from physicsnemo.launch.logging import LaunchLogger
    
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
We can use the `.log_minibatch` method to log metrics during training and validation.
We can use the `.log_epoch` method to log the learning rate at the end of the epoch.

For a more detailed example, please refer to the `Logging and Checkpointing recipe <../../user-guide/simple_logging_and_checkpointing.html>`_ .

.. autosummary::
   :toctree: generated

Launch Logger
-------------

The LaunchLogger serves as the primary interface for logging in PhysicsNeMo. It provides a unified API that works
consistently across different logging backends and training environments. The logger automatically handles metric
aggregation in distributed settings and ensures proper synchronization across processes.

.. automodule:: physicsnemo.launch.logging.launch
    :members:
    :show-inheritance:

Console Logger
--------------

A simple but powerful console-based logger that provides formatted output to the terminal. It's particularly useful
during development and debugging, offering clear visibility into training progress and metrics.

.. automodule:: physicsnemo.launch.logging.console
    :members:
    :show-inheritance:

MLflow Logger
-------------

Integration with MLflow for experiment tracking and model management. This utility enables systematic tracking of
experiments, including metrics, parameters, artifacts, and model versions. It's particularly useful for teams
that need to maintain reproducibility and compare different experiments. Users should initialize the MLflow backend
before using the LaunchLogger.

.. automodule:: physicsnemo.launch.logging.mlflow
    :members:
    :show-inheritance:

Example usage:

.. code:: python

    from physicsnemo.launch.logging.mlflow import initialize_mlflow
    from physicsnemo.launch.logging import LaunchLogger
    
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
analysis of training runs and model performance. Users should initialize the W&B backend before using the LaunchLogger.

.. automodule:: physicsnemo.launch.logging.wandb
    :members:
    :show-inheritance:

Example usage:

.. code:: python

    from physicsnemo.launch.logging.wandb import initialize_wandb
    from physicsnemo.launch.logging import LaunchLogger
    
    # Initialize W&B
    initialize_wandb(
        project="physics_simulation",
        entity="my_team"
    )
    
    # Create logger with W&B backend
    logger = LaunchLogger.initialize(use_wandb=True)

Logging utils
-------------

Utility functions and helpers for logging operations.

.. automodule:: physicsnemo.launch.logging.utils
    :members:
    :show-inheritance:


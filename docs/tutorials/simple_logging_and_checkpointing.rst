Logging and Checkpointing recipe
==================================

Logging and checkpointing are important components of model training workflow. It allows
users to keep a record of the model hyperparameters and its performance on training.

In this tutorial we will look at some of the utilities from PhysicsNeMo to simplify this
important aspect of model training. 

Logging in PhysicsNeMo
-----------------------

PhysicsNeMo provides utilities to standardize the logs of different training runs. Using the
logging utilites from PhysicsNeMo, you would have the flexibility of choosing between the
good-old console logging to more advanced ML experiments trackers like MLFlow and 
Weights & Biases. You can always implement these loggers yourself, but in this example,
we will use the utilites from PhysicsNeMo that will not only simplify this process but also
provide a standardized output format. Let's get started.

Console logging
^^^^^^^^^^^^^^^^^

The below example shows a setup using the console logging.

.. literalinclude:: ../test_scripts/test_console_logger.py
   :language: python
   :start-after: [imports] 
   :end-before: [imports]

.. literalinclude:: ../test_scripts/test_console_logger.py
   :language: python
   :start-after: [code] 
   :end-before: [code]

The logger output can be seen below.

.. code-block:: bash

   Warp 0.10.1 initialized:
      CUDA Toolkit: 11.5, Driver: 12.2
      Devices:
        "cpu"    | x86_64
        "cuda:0" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:1" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:2" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:3" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:4" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:5" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:6" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:7" | Tesla V100-SXM2-16GB-N (sm_70)
      Kernel cache: /root/.cache/warp/0.10.1
   /usr/local/lib/python3.10/dist-packages/pydantic/_internal/_fields.py:128: UserWarning: Field "model_server_url" has conflict with protected namespace "model_".

   You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
     warnings.warn(
   /usr/local/lib/python3.10/dist-packages/pydantic/_internal/_config.py:317: UserWarning: Valid config keys have changed in V2:
   * 'schema_extra' has been renamed to 'json_schema_extra'
     warnings.warn(message, UserWarning)
   [21:23:57 - main - INFO] Starting Training!
   Module physicsnemo.datapipes.benchmarks.kernels.initialization load on device 'cuda:0' took 73.06 ms
   Module physicsnemo.datapipes.benchmarks.kernels.utils load on device 'cuda:0' took 314.91 ms
   Module physicsnemo.datapipes.benchmarks.kernels.finite_difference load on device 'cuda:0' took 149.86 ms
   [21:24:02 - train - INFO] Epoch 0 Metrics: Learning Rate =  4.437e-03, Loss =  1.009e+00
   [21:24:02 - train - INFO] Epoch Execution Time:  5.664e+00s, Time/Iter:  1.133e+03ms
   [21:24:06 - train - INFO] Epoch 1 Metrics: Learning Rate =  1.969e-03, Loss =  6.040e-01
   [21:24:06 - train - INFO] Epoch Execution Time:  4.013e+00s, Time/Iter:  8.025e+02ms
   ...
   [21:25:32 - train - INFO] Epoch 19 Metrics: Learning Rate =  8.748e-10, Loss =  1.384e-01
   [21:25:32 - train - INFO] Epoch Execution Time:  4.010e+00s, Time/Iter:  8.020e+02ms
   [21:25:32 - main - INFO] Finished Training!


MLFlow logging
^^^^^^^^^^^^^^^^^

The below example shows a setup using the MLFlow logging. The only difference from
the previous example is that, we will use ``initialize_mlflow`` function to initialize
the MLFlow client and also set ``use_mlflow=True`` when initializing the ``LaunchLogger``.

.. literalinclude:: ../test_scripts/test_mlflow_logger.py
   :language: python
   :start-after: [imports] 
   :end-before: [imports]

.. literalinclude:: ../test_scripts/test_mlflow_logger.py
   :language: python
   :start-after: [code] 
   :end-before: [code]

During the run, you will notice a directory named as ``mlruns_0`` created which stores
the mlflow logs. To visulaize the logs interactively, you can run the following:

.. code-block:: bash

    mlflow ui --backend-store-uri mlruns_0/

And then navigate to localhost:5000 in your favorite browser.

.. warning::

    Currently the MLFlow logger will log the output of each processor separately. So in
    multi-processor runs, you will see multiple directories being created. This is a known
    issue and will be fixed in the future releases.


Weight and Biases logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The below example shows a  setup using the Weights and Biases logging. The only
difference from the previous example is that, we will use ``initialize_wandb`` function
to initialize the Weights and Biases logger and also set ``use_wandb=True`` when 
initializing the ``LaunchLogger``.

.. literalinclude:: ../test_scripts/test_wandb_logger.py
   :language: python
   :start-after: [imports] 
   :end-before: [imports]

.. literalinclude:: ../test_scripts/test_wandb_logger.py
   :language: python
   :start-after: [code] 
   :end-before: [code]

During the run, you will notice a directory named as ``wandb`` created which stores
the wandb logs. 

The logger output can also be seen below.

.. code-block:: bash

   Warp 0.10.1 initialized:
      CUDA Toolkit: 11.5, Driver: 12.2
      Devices:
        "cpu"    | x86_64
        "cuda:0" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:1" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:2" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:3" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:4" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:5" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:6" | Tesla V100-SXM2-16GB-N (sm_70)
        "cuda:7" | Tesla V100-SXM2-16GB-N (sm_70)
      Kernel cache: /root/.cache/warp/0.10.1
   /usr/local/lib/python3.10/dist-packages/pydantic/_internal/_fields.py:128: UserWarning: Field "model_server_url" has conflict with protected namespace "model_".

   You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
     warnings.warn(
   /usr/local/lib/python3.10/dist-packages/pydantic/_internal/_config.py:317: UserWarning: Valid config keys have changed in V2:
   * 'schema_extra' has been renamed to 'json_schema_extra'
     warnings.warn(message, UserWarning)
   wandb: Tracking run with wandb version 0.15.12
   wandb: W&B syncing is set to `offline` in this directory.  
   wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
   [21:26:38 - main - INFO] Starting Training!
   Module physicsnemo.datapipes.benchmarks.kernels.initialization load on device 'cuda:0' took 74.11 ms
   Module physicsnemo.datapipes.benchmarks.kernels.utils load on device 'cuda:0' took 310.06 ms
   Module physicsnemo.datapipes.benchmarks.kernels.finite_difference load on device 'cuda:0' took 151.24 ms
   [21:26:48 - train - INFO] Epoch 0 Metrics: Learning Rate =  1.969e-03, Loss =  7.164e-01
   [21:26:48 - train - INFO] Epoch Execution Time:  9.703e+00s, Time/Iter:  9.703e+02ms
   ...
   [21:29:47 - train - INFO] Epoch 19 Metrics: Learning Rate =  7.652e-17, Loss =  3.519e-01
   [21:29:47 - train - INFO] Epoch Execution Time:  1.125e+01s, Time/Iter:  1.125e+03ms
   [21:29:47 - main - INFO] Finished Training!
   wandb: Waiting for W&B process to finish... (success).
   wandb: 
   wandb: Run history:
   wandb:                    epoch ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
   wandb:     train/Epoch Time (s) ▃▁▃▃▃▃▁█▁▁▁▃▃▃▃▆▁▃▃▆
   wandb:      train/Learning Rate █▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
   wandb:               train/Loss █▁▂▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
   wandb: train/Time per iter (ms) ▃▁▃▃▃▃▁█▁▁▁▃▃▃▃▆▁▃▃▆
   wandb: 
   wandb: Run summary:
   wandb:                    epoch 19
   wandb:     train/Epoch Time (s) 11.24806
   wandb:      train/Learning Rate 0.0
   wandb:               train/Loss 0.35193
   wandb: train/Time per iter (ms) 1124.80645
   wandb: 
   wandb: You can sync this run to the cloud by running:
   wandb: wandb sync /workspace/physicsnemo/docs/test_scripts/wandb/wandb/offline-run-20231115_212638-ib4ylq4e
   wandb: Find logs at: ./wandb/wandb/offline-run-20231115_212638-ib4ylq4e/logs


To visualize the logs interactively, simply follow the instructions printed in the outputs. 


Checkpointing in PhysicsNeMo
-----------------------------

PhysicsNeMo provides easy utilities to save and load the checkpoints of the model, optimizer,
scheduler, and scaler during training and inference. Similar to logging, custom
implementation can be used, but in this example we will see the utilites from PhysicsNeMo and
some of its benefits.

Loading and saving checkpoints during training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example below shows how you can save and load a checkpoint during training. The implementation
allows the model training to be resumed from the last saved checkpoint. Here, we will
demonstrate the use of ``load_checkpoint`` and the ``save_checkpoint`` functions. 

.. literalinclude:: ../test_scripts/test_basic_checkpointing.py
   :language: python
   :start-after: [imports] 
   :end-before: [imports]

.. literalinclude:: ../test_scripts/test_basic_checkpointing.py
   :language: python
   :start-after: [code] 
   :end-before: [code]

The output of the above script when loaded from a partially trained model will be
something like below. 

.. code-block:: bash

    >>> python test_scripts/test_basic_checkpointing.py
    ...
    [23:11:09 - checkpoint - INFO] Loaded model state dictionary /workspace/release_23.11/docs_upgrade/physicsnemo/docs/checkpoints/FourierNeuralOperator.0.10.mdlus to device cuda
    [23:11:09 - checkpoint - INFO] Loaded checkpoint file /workspace/release_23.11/docs_upgrade/physicsnemo/docs/checkpoints/checkpoint.0.10.pt to device cuda
    [23:11:09 - checkpoint - INFO] Loaded optimizer state dictionary
    [23:11:09 - checkpoint - INFO] Loaded scheduler state dictionary
    ...
    [23:11:11 - checkpoint - INFO] Saved model state dictionary: /workspace/release_23.11/docs_upgrade/physicsnemo/docs/checkpoints/FourierNeuralOperator.0.10.mdlus
    [23:11:12 - checkpoint - INFO] Saved training checkpoint: /workspace/release_23.11/docs_upgrade/physicsnemo/docs/checkpoints/checkpoint.0.10.pt
    [23:11:16 - checkpoint - INFO] Saved model state dictionary: /workspace/release_23.11/docs_upgrade/physicsnemo/docs/checkpoints/FourierNeuralOperator.0.15.mdlus
    [23:11:16 - checkpoint - INFO] Saved training checkpoint: /workspace/release_23.11/docs_upgrade/physicsnemo/docs/checkpoints/checkpoint.0.15.pt
    [23:11:21 - checkpoint - INFO] Saved model state dictionary: /workspace/release_23.11/docs_upgrade/physicsnemo/docs/checkpoints/FourierNeuralOperator.0.20.mdlus
    [23:11:21 - checkpoint - INFO] Saved training checkpoint: /workspace/release_23.11/docs_upgrade/physicsnemo/docs/checkpoints/checkpoint.0.20.pt


Loading checkpoints during inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For loading the checkpoint in inference, the process is, and you can refer the samples
provided in :ref:`running-inference-on-trained-models` and :ref:`saving-and-loading-physicsnemo-models`.


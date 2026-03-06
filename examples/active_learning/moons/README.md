# Moons with Active Learning

This example is intended to give a quick, high level overview of one kind of active learning
experiments that can be put together using the `physicsnemo` active learning modules and
protocols.

The experiment that is being done in `moon_example.py` is to use a simple MLP classifier
to label 2D coordinates from the famous two-moons data distribution. The platform for
the experiment is to initially show the MLP a minimal set of data (with some class imbalance),
and use the prediction uncertainties from the model to query points that will be the most
informative to it.

The main thing to monitor in this experiment is the `f1_metrology.json` output, which is
a product of the `F1Metrology` strategy: in here, we compute precision/recall/F1 values
as a function of the number of active learning cycles. Due to class imbalance, the initial
precision will be quite poor as the predictions will heavily bias towards false negatives,
but as more samples (chosen by `ClassifierUQQuery`) are added to the training set, the
precision and subsequently F1 scores will improve.

## Quick Start

To run this example:

```bash
python moon_example.py
```

This will create an `active_learning_logs/<run_id>/` directory containing:

- **Model checkpoints**: `.mdlus` files saved according to `checkpoint_interval`
- **Driver logs**: `driver_log.json` tracking the active learning process
- **Metrology outputs**: `f1_metrology.json` with precision/recall/F1 scores over iterations
- **Console logs**: `console.log` with detailed execution logs

The `<run_id>` is an 8-character UUID prefix that uniquely identifies each run. You can
specify a custom `run_id` in `DriverConfig` if needed.

## Implementation notes

To illustrate a simple active learning process, this example implements the bare necessary
ingredients:

1. **Training step logic** - `training_step` function defines the per-batch training logic,
computing the loss from model predictions. This is passed to the `Driver` which uses it
within the training loop.

2. **Training loop** - The example uses `DefaultTrainingLoop`, a built-in training loop
that handles epoch iteration, progress bars, validation, and static capture optimizations.
For reference, a custom `training_loop` function is also defined in the example but not used,
showing how you could implement your own if needed.

3. **Query strategy** - `moon_strategies.ClassifierUQQuery` uses the classifier uncertainty
to rank data indices from the full (not in training) sample set, selecting points where
the model is most uncertain (predictions closest to 0.5).

4. **Label strategy** - `DummyLabelStrategy` handles obtaining data labels.
Because ground truths are already known for this dataset, it's essentially a
no-op but the `Driver` pipeline relies on it to append labeled data to the
training set.

5. **Metrology strategy** - `F1Metrology` computes precision/recall/F1 scores and serializes
them to JSON. This makes it easy to track how model performance improves with each active
learning iteration, helping inform hyperparameter choices for future experiments.

## Configuration

The rest is configuration: we take the components we've written, and compose them in
the various configuration dataclasses in `moon_example.py::main`.

### TrainingConfig

The `train_datapool` specifies what set of data to train on. We configure the training
loop using `DefaultTrainingLoop` with progress bars enabled. The `OptimizerConfig` specifies
which optimizer to use and its hyperparameters. You can configure different epoch counts
for initial training vs. subsequent fine-tuning iterations.

```python
# configure how training/fine-tuning is done within active learning
training_config = c.TrainingConfig(
    train_datapool=dataset,
    optimizer_config=c.OptimizerConfig(
        torch.optim.SGD,
        optimizer_kwargs={"lr": 0.01},
    ),
    # configure different times for initial training and subsequent
    # fine-tuning
    max_training_epochs=10,
    max_fine_tuning_epochs=5,
    # this configures the training loop
    train_loop_fn=DefaultTrainingLoop(
        use_progress_bars=True,
    ),
)
```

**Key options:**

- `max_training_epochs`: Epochs for initial training (step 0)
- `max_fine_tuning_epochs`: Epochs for subsequent fine-tuning steps
- `DefaultTrainingLoop(use_progress_bars=True)`: Built-in loop with tqdm progress bars
- `val_datapool`: Optional validation dataset (not used in this example)

### StrategiesConfig

The `StrategiesConfig` localizes all of the different active learning components
into one place. The `queue_cls` is used to pipeline query samples to label processes.
Because we're carrying out a single process workflow, `queue.Queue` is sufficient,
but multiprocess variants, up to constructs like Redis Queue, can be used to pass
data around the pipeline.

```python
strategy_config = c.StrategiesConfig(
    query_strategies=[ClassifierUQQuery(max_samples=10)],
    queue_cls=queue.Queue,
    label_strategy=DummyLabelStrategy(),
    metrology_strategies=[F1Metrology()],
)
```

**Key components:**

- `query_strategies`: List of strategies for selecting samples (can have multiple)
- `queue_cls`: Queue implementation for passing data between phases (e.g., `queue.Queue`)
- `label_strategy`: Single strategy for labeling queried samples
- `metrology_strategies`: List of strategies for measuring model performance
- `unlabeled_datapool`: Optional pool of unlabeled data for query strategies (not shown here)

### DriverConfig

Finally, the `DriverConfig` specifies orchestration parameters that control the overall
active learning loop execution:

```python
driver_config = c.DriverConfig(
    batch_size=16,
    max_active_learning_steps=70,
    fine_tuning_lr=0.005,
    device=torch.device("cpu"),  # set to other accelerators if needed
)
driver = Driver(
    config=driver_config,
    learner=uq_model,
    strategies_config=strategy_config,
    training_config=training_config,
)
# our model doesn't implement a `training_step` method but in principle
# it could be implemented, and we wouldn't need to pass the step function here
driver(train_step_fn=training_step)
```

**Key parameters:**

- `batch_size`: Batch size for training and validation dataloaders
- `max_active_learning_steps`: Total number of active learning iterations
- `fine_tuning_lr`: Learning rate to switch to after the first AL step (optional)
- `device`: Device for computation (e.g., `torch.device("cpu")`, `torch.device("cuda:0")`)
- `dtype`: Data type for tensors (defaults to `torch.get_default_dtype()`)
- `skip_training`: Set to `True` to skip training phase (default: `False`)
- `skip_metrology`: Set to `True` to skip metrology phase (default: `False`)
- `skip_labeling`: Set to `True` to skip labeling phase (default: `False`)
- `checkpoint_interval`: Save model every N steps (default: 1, set to 0 to disable)
- `root_log_dir`: Directory for logs and checkpoints (default: `"active_learning_logs"`)
- `dist_manager`: Optional `DistributedManager` for multi-GPU training

### Running the Driver

The final `driver(...)` call is syntactic sugar for `driver.run(...)`, which executes the
full active learning loop. The `train_step_fn` argument provides the per-batch training logic.

**Two ways to provide training logic:**

1. **Pass as function** (shown in example):

   ```python
   driver(train_step_fn=training_step)
   ```

1. **Implement in model** (alternative):

   ```python
   class MLP(Module):
       def training_step(self, data):
           # training logic here
           ...

   driver()  # no train_step_fn needed
   ```

**Optional validation step:**

You can also provide a `validate_step_fn` parameter:

```python
driver(train_step_fn=training_step, validate_step_fn=validation_step)
```

### Active Learning Workflow

Under the hood, `Driver.active_learning_step` is called repeatedly for the number of
iterations specified in `max_active_learning_steps`. Each iteration follows this sequence:

1. **Training Phase**: Train model on current `train_datapool` using the
training loop
2. **Metrology Phase**: Compute performance metrics via metrology strategies
3. **Query Phase**: Select new samples to label via query strategies →
`query_queue`
4. **Labeling Phase**: Label queued samples via label strategy → `label_queue`
→ append to `train_datapool`

The logic for each phase is in methods like `Driver._training_phase`,
`Driver._query_phase`, etc.

## Advanced Customization

### Custom Training Loops

While `DefaultTrainingLoop` is suitable for most use cases, you can write
custom training loops that implement the `TrainingLoop` protocol, which is the
overarching logic for how to carry out model training and validation over some
number of epochs. Custom loops are useful when you need:

- Specialized training logic (e.g., alternating, or multiple optimizers)
- Custom logging or checkpointing within the loop
- Non-standard epoch/batch iteration patterns

### Custom Strategies

All strategies must implement their respective protocols:

- **QueryStrategy**: Implement `sample(query_queue, *args, **kwargs)` and
`attach(driver)`
- **LabelStrategy**: Implement `label(queue_to_label, serialize_queue, *args,
**kwargs)` and `attach(driver)`
- **MetrologyStrategy**: Implement `compute(*args, **kwargs)`,
`serialize_records(*args, **kwargs)`, and `attach(driver)`

The `attach(driver)` method gives your strategy access to the driver's
attributes like `driver.learner`, `driver.train_datapool`,
`driver.unlabeled_datapool`, etc.

### Static Capture and Performance

The `DefaultTrainingLoop` supports static capture via CUDA graphs for
performance optimization:

```python
train_loop_fn=DefaultTrainingLoop(
    enable_static_capture=True,  # Enable CUDA graph capture (default)
    use_progress_bars=True,
)
```

For custom training loops, you can use:

- `StaticCaptureTraining` for training steps
- `StaticCaptureEvaluateNoGrad` for validation/inference steps

### Distributed Training

To use multiple GPUs, provide a `DistributedManager` in `DriverConfig`:

```python
from physicsnemo.distributed import DistributedManager

dist_manager = DistributedManager()
driver_config = c.DriverConfig(
    batch_size=16,
    max_active_learning_steps=70,
    dist_manager=dist_manager,  # Handles device placement and DDP
)
```

The driver will automatically wrap the model in `DistributedDataParallel` and use
`DistributedSampler` for dataloaders.

## Experiment Ideas

Here, we perform a relatively straightforward experiment without a baseline; suitable
ones could be to train a model using the full data, and see how the precision/recall/F1
scores differ between the `ClassifierUQQuery` learner to the full data model (i.e. use
the latter as a roofline).

A suitable baseline to compare against would be random selection: to check the efficacy
of `ClassifierUQQuery`, samples could be chosen uniformly and see if and how the same
metrology scores differ. If the UQ is performing as intended, then precision/recall/F1
should improve at a faster rate.

# Active Learning Module

The `physicsnemo.active_learning` namespace is used for defining the "scaffolding"
that can be used to construct automated, end-to-end active learning workflows.
For areas of science that are difficult to source ground-truths to train on
(of which there are many), an active learning curriculum attempts to train a
model with improved data efficiency; better generalization performance but requiring
fewer training samples.

Generally, an active learning workflow can be decomposed into three "phases"
that are - in the simplest case - run sequentially:

- **Training/fine-tuning**: A "learner" or surrogate model is initially trained
on available data, and in subsequent active learning iterations, is fine-tuned
with the new data appended on the original dataset.
- **Querying**: One or more strategies that encode some heuristics for what
new data is most informative for the learner. Examples of this include
uncertainty-based methods, which may screen a pool of unlabeled data for
those the model is least confident with.
- **Labeling**: A method of obtaining ground truth (labels) for new data
points, pipelined from the querying stage. This may entail running an
expensive solver, or acquiring experimental data.

The three phases are repeated until the learner converges. Because "convergence"
may not be easily defined, we define an additional phase which we call
**metrology**: this represents a phase most similar to querying, but allows
a user to define some set of criteria to monitor over the course of active
learning *beyond* simple validation metrics to ensure the model can be used
with confidence as surrogates (e.g. within a simulation loop).

## How to use this module

With the context above in mind, inspecting the `driver` module will give you
a sense for how the end-to-end workflow functions; the `Driver` class acts
as an orchestrator for all the phases of active learning we described above.

From there, you should realize that `Driver` is written in a highly abstract
way: we need concrete *strategies* that implement querying, labeling, and metrology
concepts. The `protocols` module provides the scaffolding to do so - we implement
various components as `typing.Protocol` which are used for structural sub-typing:
they can be thought of as abstract classes that define an expected interface
in a function or class from which you can define your own classes by either
inheriting from them, or defining your own class that implements the expected
methods and attributes.

In order to perform the training portion of active learning, we provide a
minimal yet functional `DefaultTrainingLoop` inside the `loop` module. This
loop simply requires a `protocols.TrainingProtocol` to be passed, which is
a function that defines the logic for computing the loss per batch/training
step.

## Configuring workflows

The `config` module defines some simple `dataclass`es that can be used
to configure the behavior of various parts of active learning, e.g. how
training is conducted, etc. Because `Driver` is designed to be checkpointable,
with the exception of a few parts such as datasets, everything should be
JSON-serializable.

## Restarting workflows

For classes and functions that are created at runtime, checkpointing requires
that these components can be recreated when restarting from a checkpoint. To
that end, the `_registry` module provides a user-friendly way to instantiate
objects: user-defined strategy classes can be added to the registry to enable
their creation in checkpoint restarts.

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains base classes for active learning protocols.

These are protocols intended to be abstract, and importing these
classes specifically is intended to either be subclassed, or for
type annotations.

Protocol Architecture
---------------------
Python :class:`typing.Protocol` s are used for structural typing: essentially, they are
used to describe an expected interface in a way that is helpful for static type checkers
to make sure concrete implementations provide everything that is needed for a workflow
to function. :class:`typing.Protocol` s are not actually enforced at runtime, and
inheritance is not required for them to function: as long as the implementation
provides the expected attributes and methods, they will be compatible with the protocol.

The active learning framework is built around several key protocol abstractions
that work together to orchestrate the active learning workflow:

**Core Infrastructure Protocols:**
 - `AbstractQueue[T]` - Generic queue protocol for passing data between components
 - `DataPool[T]` - Protocol for data reservoirs that support appending and sampling
 - `ActiveLearningProtocol` - Base protocol providing common interface for all AL strategies

**Strategy Protocols (inherit from ActiveLearningProtocol):**
 - `QueryStrategy` - Defines how to select data points for labeling
 - `LabelStrategy` - Defines processes for adding ground truth labels to unlabeled data
 - `MetrologyStrategy` - Defines procedures that assess model improvements beyond validation metrics

**Model Interface Protocols:**
 - `TrainingProtocol` - Interface for training step functions
 - `ValidationProtocol` - Interface for validation step functions
 - `InferenceProtocol` - Interface for inference step functions
 - `TrainingLoop` - Interface for complete training loop implementations
 - `LearnerProtocol` - Comprehensive interface for learner modules (combines training/validation/inference)

**Orchestration Protocol:**
 - `DriverProtocol` - Main orchestrator that coordinates all components in the active learning loop

Active Learning Workflow
------------------------

The typical active learning workflow orchestrated by `DriverProtocol` follows this sequence:

1. **Training Phase**: Use `LearnerProtocol` or `TrainingLoop` to train the model on `training_pool`
2. **Metrology Phase** (optional): Apply `MetrologyStrategy` instances to assess model performance
3. **Query Phase**: Apply `QueryStrategy` instances to select samples from `unlabeled_pool` → `query_queue`
4. **Labeling Phase** (optional): Apply `LabelStrategy` instances to label queued samples → `label_queue`
5. **Data Integration**: Move labeled data from `label_queue` to `training_pool`

Type Parameters
---------------
- `T`: Data structure containing both inputs and ground truth labels
- `S`: Data structure containing only inputs (no ground truth labels)

----
"""

from __future__ import annotations

import inspect
import logging
from enum import StrEnum
from logging import Logger
from pathlib import Path
from typing import Any, Iterator, Protocol, TypeVar

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from physicsnemo.core import Module

# T is used to denote a data structure that contains inputs for a model and ground truths
T = TypeVar("T")
# S is used to denote a data structure that has inputs for a model, but no ground truth labels
S = TypeVar("S")


class ActiveLearningPhase(StrEnum):
    """
    An enumeration of the different phases of the active learning workflow.

    This is primarily used in the metadata for restarting an ongoing active
    learning experiment.

    See Also
    --------
    ActiveLearningProtocol : Base protocol for active learning strategies
    DriverProtocol : Main orchestrator that uses this enumeration
    """

    TRAINING = "training"
    METROLOGY = "metrology"
    QUERY = "query"
    LABELING = "labeling"
    DATA_INTEGRATION = "data_integration"


class AbstractQueue(Protocol[T]):
    """
    Defines a generic queue protocol for data that is passed between active
    learning components.

    This can be a simple local `queue.Queue`, or a more sophisticated
    distributed queue system.

    The primary use case for this is to allow a query strategy to
    enqueue some data structure for the labeling strategy to consume,
    and once the labeling is done, enqueue to a data serialization
    workflow. While there is no explcit restriction on the **type**
    of queue that is implemented, a reasonable assumption to make
    would be a FIFO queue, unless otherwise specified by the concrete
    implementation.

    Optional Serialization Methods
    -------------------------------
    Implementations may optionally provide `to_list()` and `from_list()`
    methods for checkpoint serialization. If not provided, the queue
    will be serialized using `torch.save()` as a fallback.

    Type Parameters
    ---------------
    T
        The type of items that will be stored in the queue.

    See Also
    --------
    QueryStrategy : Enqueues data to be labeled
    LabelStrategy : Dequeues data for labeling and enqueues labeled data
    DriverProtocol : Uses queues to pass data between strategies
    """

    def put(self, item: T) -> None:
        """
        Method to put a data structure into the queue.

        Parameters
        ----------
        item: T
            The data structure to put into the queue.
        """
        ...

    def get(self) -> T:
        """
        Method to get a data structure from the queue.

        This method should remove the data structure from the queue,
        and return it to a consumer.

        Returns
        -------
        T
            The data structure that was removed from the queue.
        """
        ...

    def empty(self) -> bool:
        """
        Method to check if the queue is empty/has been depleted.

        Returns
        -------
        bool
            True if the queue is empty, False otherwise.
        """
        ...


class DataPool(Protocol[T]):
    """
    An abstract protocol for some reservoir of data that is
    used for some part of active learning, parametrized such
    that it will return data structures of an arbitrary type ``T``.

    **All** methods are left abstract, and need to be defined
    by concrete implementations. For the most part, a `torch.utils.data.Dataset`
    would match this protocol, provided that it implements the :meth:`append` method
    which will allow data to be persisted to a filesystem.

    Methods
    -------
    __getitem__(self, index: int) -> T:
        Method to get a single data structure from the data pool.
    __len__(self) -> int:
        Method to get the length of the data pool.
    __iter__(self) -> Iterator[T]:
        Method to iterate over the data pool.
    append(self, item: T) -> None:
        Method to append a data structure to the data pool.

    See Also
    --------
    DriverProtocol : Uses data pools for training, validation, and unlabeled data
    AbstractQueue : Queue protocol for passing data between components
    """

    def __getitem__(self, index: int) -> T:
        """
        Method to get a data structure from the data pool.

        This method should retrieve an item from the pool by a
        flat index.

        Parameters
        ----------
        index: int
            The index of the data structure to get.

        Returns
        -------
        T
            The data structure at the given index.
        """
        ...

    def __len__(self) -> int:
        """
        Method to get the length of the data pool.

        Returns
        -------
        int
            The length of the data pool.
        """
        ...

    def __iter__(self) -> Iterator[T]:
        """
        Method to iterate over the data pool.

        This method should return an iterator over the data pool.

        Returns
        -------
        Iterator[T]
            An iterator over the data pool.
        """
        ...

    def append(self, item: T) -> None:
        """
        Method to append a data structure to the data pool.

        For persistent storage pools, this will actually mean that the
        ``item`` is serialized to a filesystem.

        Parameters
        ----------
        item: T
            The data structure to append to the data pool.
        """
        ...


class ActiveLearningProtocol(Protocol):
    """
    This protocol acts as a basis for all active learning protocols.

    This ensures that all protocols have some common interface, for
    example the ability to :meth:`attach` to another object for scope
    management.

    Attributes
    ----------
    __protocol_name__: str
        The name of the protocol. This is primarily used for ``repr``
        and ``str`` f-strings. This should be defined by concrete
        implementations.
    _args: dict[str, Any]
        A dictionary of arguments that were used to instantiate the protocol.
        This is used for serialization and deserialization of the protocol,
        and follows the same pattern as the ``_args`` attribute of
        :class:`physicsnemo.Module`.

    Methods
    -------
    attach(self, other: object) -> None:
        This method is used to attach the current object to another,
        allowing the protocol to access the attached object's scope.
        The use case for this is to allow a protocol access to the
        driver's scope to access dataset, model, etc. as needed.
        This needs to be implemented by concrete implementations.
    is_attached: bool
        Whether the current object is attached to another object.
        This is left abstract, as it depends on how :meth:`attach` is implemented.
    logger: Logger
        The logger for this protocol. This is used to log information
        about the protocol's progress.
    _setup_logger(self) -> None:
        This method is used to setup the logger for the protocol.
        The default implementation is to configure the logger similarly
        to how ``physicsnemo`` loggers are configured.

    See Also
    --------
    QueryStrategy : Query strategy protocol (child)
    LabelStrategy : Label strategy protocol (child)
    MetrologyStrategy : Metrology strategy protocol (child)
    DriverProtocol : Main orchestrator that uses these protocols
    """

    __protocol_name__: str
    __protocol_type__: ActiveLearningPhase
    _args: dict[str, Any]

    def __new__(cls, *args: Any, **kwargs: Any) -> ActiveLearningProtocol:
        """
        Wrapper for instantiating any subclass of `ActiveLearningProtocol`.

        This method will use `inspect` to capture arguments and keyword
        arguments that were used to instantiate the protocol, and stash
        them into the `_args` attribute of the instance, following
        what is done with :class:`physicsnemo.Module`.

        This approach is useful for reconstructing strategies from checkpoints.

        Parameters
        ----------
        args: Any
            Arguments to pass to the protocol's constructor.
        kwargs: Any
            Keyword arguments to pass to the protocol's constructor.

        Returns
        -------
        ActiveLearningProtocol
            A new instance of the protocol class. The instance will have an
            `_args` attribute that contains the keys `__name__`, `__module__`,
            and `__args__` as metadata for the protocol.
        """
        out = super().__new__(cls)

        # Get signature of __init__ function
        sig = inspect.signature(cls.__init__)

        # Bind args and kwargs to signature
        bound_args = sig.bind_partial(
            *([None] + list(args)), **kwargs
        )  # Add None to account for self
        bound_args.apply_defaults()

        # Get args and kwargs (excluding self and unroll kwargs)
        instantiate_args = {}
        for param, (k, v) in zip(sig.parameters.values(), bound_args.arguments.items()):
            # Skip self
            if k == "self":
                continue

            # Add args and kwargs to instantiate_args
            if param.kind == param.VAR_KEYWORD:
                instantiate_args.update(v)
            else:
                instantiate_args[k] = v

        # Store args needed for instantiation
        out._args = {
            "__name__": cls.__name__,
            "__module__": cls.__module__,
            "__args__": instantiate_args,
        }
        return out

    def attach(self, other: object) -> None:
        """
        This method is used to attach another object to the current protocol,
        allowing the attached object to access the scope of this protocol.
        The primary reason for this is to allow the protocol to access
        things like the dataset, the learner model, etc. as needed.

        Example use cases would be for a query strategy to access the ``unlabeled_pool``;
        for a metrology strategy to access the ``validation_pool``, and for any
        strategy to be able to access the surrogate/learner model.

        This method can be as simple as setting ``self.driver = other``, but
        is left abstract in case there are other potential use cases
        where multiple protocols could share information.

        Parameters
        ----------
        other: object
            The object to attach to.
        """
        ...

    @property
    def is_attached(self) -> bool:
        """
        Property to check if the current object is already attached.

        This is left abstract, as it depends on how ``attach`` is implemented.

        Returns
        -------
        bool
            True if the current object is attached, False otherwise.
        """
        ...

    @property
    def logger(self) -> Logger:
        """
        Property to access the logger for this protocol.

        If the logger has not been configured yet, the property
        will call the `_setup_logger` method to configure it.

        Returns
        -------
        Logger
            The logger for this protocol.
        """
        if not hasattr(self, "_logger"):
            self._setup_logger()
        return self._logger

    @logger.setter
    def logger(self, logger: Logger) -> None:
        """
        Setter for the logger for this protocol.

        Parameters
        ----------
        logger: Logger
            The logger to set for this protocol.
        """
        self._logger = logger

    def _setup_logger(self) -> None:
        """
        Method to setup the logger for all active learning protocols.

        Each protocol should have their own logger
        """
        self.logger = logging.getLogger(
            f"core.active_learning.{self.__protocol_name__}"
        )
        # Don't add handlers here - let the parent logger handle formatting
        # This prevents duplicate console output
        self.logger.setLevel(logging.WARNING)

    @property
    def strategy_dir(self) -> Path:
        """
        Returns the directory where the underlying strategy can use
        to persist data.

        Depending on the strategy abstraction, further nesting may be
        required (e.g active learning step index, phase, etc.).

        Returns
        -------
        Path
            The directory where the metrology strategy will persist
            its records.

        Raises
        ------
        RuntimeError
            If the metrology strategy is not attached to a driver yet.
        """
        if not self.is_attached:
            raise RuntimeError(
                f"{self.__class__.__name__} is not attached to a driver yet."
            )
        path = (
            self.driver.log_dir / str(self.__protocol_type__) / self.__class__.__name__
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoint_dir(self) -> Path:
        """
        Utility property for strategies to conveniently access the checkpoint directory.

        This is useful for (de)serializing data tied to checkpointing.

        Returns
        -------
        Path
            The checkpoint directory, which includes the active learning step index.

        Raises
        ------
        RuntimeError
            If the strategy is not attached to a driver yet.
        """
        if not self.is_attached:
            raise RuntimeError(
                f"{self.__class__.__name__} is not attached to a driver yet."
            )
        path = (
            self.driver.log_dir
            / "checkpoints"
            / f"step_{self.driver.active_learning_step_idx}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path


class QueryStrategy(ActiveLearningProtocol):
    """
    This protocol defines a query strategy for active learning.

    A query strategy is responsible for selecting data points for labeling.
    In the most general sense, concrete instances of this protocol
    will specify how many samples to query, and the heuristics for
    selecting samples.

    Attributes
    ----------
    max_samples: int
        The maximum number of samples to query. This can be interpreted
        as the exact number of samples to query, or as an upper limit
        for querying methods that are threshold based.

    See Also
    --------
    ActiveLearningProtocol : Base protocol for all active learning strategies
    AbstractQueue : Queue protocol for enqueuing data
    LabelStrategy : Consumes queued data for labeling
    DriverProtocol : Orchestrates query strategies
    """

    max_samples: int
    __protocol_type__ = ActiveLearningPhase.QUERY

    def sample(self, query_queue: AbstractQueue[T], *args: Any, **kwargs: Any) -> None:
        """
        Method that implements the logic behind querying data to be labeled.

        This method should be implemented by concrete implementations,
        and assume that an active learning driver will pass a queue
        for this method to enqueue data to be labeled.

        Additional ``args`` and ``kwargs`` are passed to the method,
        and can be used to pass additional information to the query strategy.

        This method will enqueue in place, and should not return anything.

        Parameters
        ----------
        query_queue: AbstractQueue[T]
            The queue to enqueue data to be labeled.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...

    def __call__(
        self, query_queue: AbstractQueue[T], *args: Any, **kwargs: Any
    ) -> None:
        """
        Syntactic sugar for the ``sample`` method.

        This allows the object to be called as a function, and will pass
        the arguments to the strategy's ``sample`` method.

        Parameters
        ----------
        query_queue: AbstractQueue[T]
            The queue to enqueue data to be labeled.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        self.sample(query_queue, *args, **kwargs)


class LabelStrategy(ActiveLearningProtocol):
    """
    This protocol defines a label strategy for active learning.

    A label strategy is responsible for labeling data points; this may
    be an simple Python function for demonstrating a concept, or an external,
    potentially time consuming and complex, process.

    Attributes
    ----------
    __is_external_process__: bool
        Whether the label strategy is running in an external process.
    __provides_fields__: set or None
        The fields that the label strategy provides. This should be
        set by concrete implementations, and should be used to write
        and map labeled data to fields within the data structure ``T``.

    See Also
    --------
    ActiveLearningProtocol : Base protocol for all active learning strategies
    AbstractQueue : Queue protocol for dequeuing and enqueuing data
    QueryStrategy : Produces queued data for labeling
    DriverProtocol : Orchestrates the label strategy
    """

    __is_external_process__: bool
    __provides_fields__: set[str] | None = None
    __protocol_type__ = ActiveLearningPhase.LABELING

    def label(
        self,
        queue_to_label: AbstractQueue[T],
        serialize_queue: AbstractQueue[T],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Method that implements the logic behind labeling data.

        This method should be implemented by concrete implementations,
        and assume that an active learning driver will pass a queue
        for this method to dequeue data to be labeled.

        Parameters
        ----------
        queue_to_label: AbstractQueue[T]
            Queue containing data structures to be labeled. Generally speaking,
            this should be passed over after running query strateg(ies).
        serialize_queue: AbstractQueue[T]
            Queue for enqueing labeled data to be serialized.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...

    def __call__(
        self,
        queue_to_label: AbstractQueue[T],
        serialize_queue: AbstractQueue[T],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Syntactic sugar for the ``label`` method.

        This allows the object to be called as a function, and will pass
        the arguments to the strategy's ``label`` method.

        Parameters
        ----------
        queue_to_label: AbstractQueue[T]
            Queue containing data structures to be labeled.
        serialize_queue: AbstractQueue[T]
            Queue for enqueing labeled data to be serialized.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        self.label(queue_to_label, serialize_queue, *args, **kwargs)


class MetrologyStrategy(ActiveLearningProtocol):
    """
    This protocol defines a metrology strategy for active learning.

    A metrology strategy is responsible for assessing the improvements to the underlying
    model, beyond simple validation metrics. This should reflect the application
    requirements of the model, which may include running a simulation.

    Attributes
    ----------
    records: list
        A sequence of record data structures that records the
        history of the active learning process, as viewed by
        this particular metrology view.

    See Also
    --------
    ActiveLearningProtocol : Base protocol for all active learning strategies
    DriverProtocol : Orchestrates metrology strategies
    DataPool : Data pool protocol for accessing validation data
    """

    records: list[S]
    __protocol_type__ = ActiveLearningPhase.METROLOGY

    def append(self, record: S) -> None:
        """
        Method to append a record to the metrology strategy.

        Parameters
        ----------
        record: S
            The record to append to the metrology strategy.
        """
        self.records.append(record)

    def __len__(self) -> int:
        """
        Method to get the length of the metrology strategy.

        Returns
        -------
        int
            The length of the metrology strategy.
        """
        return len(self.records)

    def serialize_records(
        self, path: Path | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """
        Method to serialize the records of the metrology strategy.

        This should be defined by a concrete implementation, which dictates
        how the records are persisted, e.g. to a JSON file, database, etc.

        The `strategy_dir` property can be used to determine the directory where
        the records should be persisted.

        Parameters
        ----------
        path: Path | None
            The path to serialize the records to. If not provided, the strategy
            should provide a reasonable default, such as with the checkpointing
            or within the corresponding metrology directory via `strategy_dir`.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...

    def load_records(self, path: Path | None = None, *args: Any, **kwargs: Any) -> None:
        """
        Method to load the records of the metrology strategy, i.e.
        the reverse of `serialize_records`.

        This should be defined by a concrete implementation, which dictates
        how the records are loaded, e.g. from a JSON file, database, etc.

        If no path is provided, the strategy should load the latest records
        as sensible defaults. The `records` attribute should then be overwritten
        in-place.

        Parameters
        ----------
        path: Path | None
            The path to load the records from. If not provided, the strategy
            should load the latest records as sensible defaults.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...

    def compute(self, *args: Any, **kwargs: Any) -> None:
        """
        Method to compute the metrology strategy. No data is passed to
        this method, as it is expected that the data be drawn as needed
        from various ``DataPool`` connected to the driver.

        This method defines the core logic for computing a particular view
        of performance by the underlying model on the data. Once computed,
        the data needs to be formatted into a record data structure ``S``,
        that is then appended to the ``records`` attribute.

        Parameters
        ----------
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
        Syntactic sugar for the ``compute`` method.

        This allows the object to be called as a function, and will pass
        the arguments to the strategy's ``compute`` method.

        Parameters
        ----------
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        self.compute(*args, **kwargs)

    def reset(self) -> None:
        """
        Method to reset any stateful attributes of the metrology strategy.

        By default, the ``records`` attribute is reset to an empty list.
        """
        self.records = []


class TrainingProtocol(Protocol):
    """
    This protocol defines the interface for training steps: given
    a model and some input data, compute the reduced, differentiable
    loss tensor and return it.

    A concrete implementation can simply be a function with a signature that
    matches what is defined in :meth:`__call__`.

    See Also
    --------
    TrainingLoop : Training loop protocol that uses this protocol
    LearnerProtocol : Learner protocol with a training_step method
    ValidationProtocol : Validation step protocol
    """

    def __call__(
        self, model: Module, data: T, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        Implements the training logic for a single training sample or batch.

        For a PhysicsNeMo :class:`physicsnemo.Module` with trainable parameters, the output
        of this function should correspond to a PyTorch tensor that is
        ``backward``-ready. If there are any logging operations associated
        with training, they should be performed within this function.

        For ideal performance, this function should also be wrappable with
        ``StaticCaptureTraining`` for optimization.

        Parameters
        ----------
        model: :class:`physicsnemo.Module`
            The model to train.
        data: T
            The data to train on. This data structure should comprise
            both input and ground truths to compute the loss.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.

        Returns
        -------
        torch.Tensor
            The reduced, differentiable loss tensor.

        Example
        -------
        Minimum viable implementation:

        >>> import torch
        >>> def training_step(model, data):
        ...     output = model(data)
        ...     loss = torch.sum(torch.pow(output - data, 2))
        ...     return loss
        """
        ...


class ValidationProtocol(Protocol):
    """
    This protocol defines the interface for validation steps: given
    a model and some input data, compute metrics of interest and if
    relevant to do so, log the results.

    A concrete implementation can simply be a function with a signature that
    matches what is defined in :meth:`__call__`.

    See Also
    --------
    TrainingLoop : Training loop protocol that uses this protocol
    LearnerProtocol : Learner protocol with a validation_step method
    TrainingProtocol : Training step protocol
    """

    def __call__(self, model: Module, data: T, *args: Any, **kwargs: Any) -> None:
        """
        Implements the validation logic for a single sample or batch.

        This method will be called in validation steps **only**, and not used
        for training, query, or metrology steps. In those cases,
        implement the :meth:`InferenceProtocol.__call__` method instead.

        This function should not return anything, but should contain the logic
        for computing metrics of interest over a validation/test set. If there
        are any logging operations that need to be performed, they should also
        be performed here.

        Depending on the type of model architecture, consider wrapping this method
        with ``StaticCaptureEvaluateNoGrad`` for performance optimizations. This
        should be used if the model does not require autograd as part of its
        forward pass.

        Parameters
        ----------
        model: :class:`physicsnemo.Module`
            The model to validate.
        data: T
            The data to validate on. This data structure should comprise
            both input and ground truths to compute the loss.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.

        Example
        -------
        Minimum viable implementation:

        >>> import torch
        >>> def validation_step(model, data):
        ...     output = model(data)
        ...     loss = torch.sum(torch.pow(output - data, 2))
        ...     return loss
        """
        ...


class InferenceProtocol(Protocol):
    """
    This protocol defines the interface for inference steps: given
    a model and some input data, return the output of the model's forward pass.

    A concrete implementation can simply be a function with a signature that
    matches what is defined in :meth:`__call__`.

    See Also
    --------
    LearnerProtocol : Learner protocol with an inference_step method
    QueryStrategy : Uses inference for query strategies
    MetrologyStrategy : Uses inference for metrology strategies
    """

    def __call__(self, model: Module, data: S, *args: Any, **kwargs: Any) -> Any:
        """
        Implements the inference logic for a single sample or batch.

        This method will be called in query and metrology steps, and should
        return the output of the model's forward pass, likely minimally processed
        so that any transformations can be performed by strategies that utilize
        this protocol.

        The key difference between this protocol and the other two training and
        validation protocols is that the data structure ``S`` does not need
        to contain ground truth values to compute a loss.

        Similar to :class:`ValidationProtocol`, if relevant to the underlying architecture,
        consider wrapping a concrete implementation of this protocol with
        ``StaticCaptureInference`` for performance optimizations.

        Parameters
        ----------
        model: :class:`physicsnemo.Module`
            The model to infer on.
        data: S
            The data to infer on. This data structure should comprise
            only input values to compute the forward pass.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.

        Returns
        -------
        Any
            The output of the model's forward pass.

        Example
        -------
        Minimum viable implementation:

        >>> def inference_step(model, data):
        ...     output = model(data)
        ...     return output
        """
        ...


class TrainingLoop(Protocol):
    """
    Defines a protocol that implements a training loop.

    This protocol is intended to be called within the active learning loop
    during the training phase, where the model is trained on a specified
    number of epochs or training steps, and optionally validated on a dataset.

    If a :class:`LearnerProtocol` is provided, then ``train_fn`` and ``validate_fn``
    become optional as they will be defined within the :class:`LearnerProtocol`. If
    they are provided, however, then they should override the :class:`LearnerProtocol`
    variants.

    If graph capture/compilation is intended, then ``train_fn`` and ``validate_fn``
    should be wrapped with ``StaticCaptureTraining`` and ``StaticCaptureEvaluateNoGrad``,
    respectively.

    See Also
    --------
    DriverProtocol : Uses training loops in the training phase
    TrainingProtocol : Training step protocol
    ValidationProtocol : Validation step protocol
    LearnerProtocol : Learner protocol with training/validation methods
    """

    def __call__(
        self,
        model: Module | LearnerProtocol,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader | None = None,
        train_step_fn: TrainingProtocol | None = None,
        validate_step_fn: ValidationProtocol | None = None,
        max_epochs: int | None = None,
        max_train_steps: int | None = None,
        max_val_steps: int | None = None,
        lr_scheduler: _LRScheduler | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Defines the signature for a minimal viable training loop.

        The protocol defines a ``model`` with trainable parameters
        tracked by ``optimizer`` will go through multiple epochs or
        training steps. In the latter, the ``train_dataloader`` will be
        exhausted ``max_epochs`` times, while the mutually exclusive
        ``max_train_steps`` will limit the number of training batches,
        which can be greater or less than the length of the ``train_dataloader``.

        (Optional) Validation is intended to be performed either at the end of a training
        epoch, or when the maximum number of training steps is reached. The
        ``max_val_steps`` parameter can be used to limit the number of batches to validate with
        on a per-epoch basis. Validation is only performed if a ``validate_step_fn`` is provided,
        alongside ``validation_dataloader``.

        The pseudocode for training to ``max_epochs`` would look like this:

        .. code-block:: python

           max_epochs = 10
           for epoch in range(max_epochs):
               for train_idx, batch in enumerate(train_dataloader):
                   optimizer.zero_grad()
                   loss = train_step_fn(model, batch)
                   loss.backward()
                   optimizer.step()
                   if train_idx + 1 == max_train_steps:
                       break
               if validate_step_fn and validation_dataloader:
                   for val_idx, batch in enumerate(validation_dataloader):
                       validate_step_fn(model, batch)
                       if val_idx + 1 == max_val_steps:
                           break

        The pseudocode for training with a :class:`LearnerProtocol` would look like this:

        .. code-block:: python

           for epoch in range(max_epochs):
               for train_idx, batch in enumerate(train_dataloader):
                   loss = model.training_step(batch)
                   if train_idx + 1 == max_train_steps:
                       break
               if validation_dataloader:
                   for val_idx, batch in enumerate(validation_dataloader):
                       model.validation_step(batch)
                       if val_idx + 1 == max_val_steps:
                           break

        The key difference between specifying ``train_step_fn`` and :class:`LearnerProtocol`
        is that the former excludes the backward pass and optimizer step logic,
        whereas the latter encapsulates them.

        The ``device`` and ``dtype`` parameters are used to specify the device and
        dtype to use for the training loop. If not provided, a reasonable default
        should be used (e.g. from ``torch.get_default_device()`` and ``torch.get_default_dtype()``).

        Parameters
        ----------
        model: :class:`physicsnemo.Module` or :class:`LearnerProtocol`
            The model to train.
        optimizer: `Optimizer`
            The optimizer to use for training.
        train_dataloader: `DataLoader`
            The dataloader to use for training.
        validation_dataloader: `DataLoader` or None
            The dataloader to use for validation.
        train_step_fn: :class:`TrainingProtocol` or None
            The training function to use for training. This is optional only
            if ``model`` implements the :class:`LearnerProtocol`. If this is
            provided and ``model`` implements the :class:`LearnerProtocol`,
            then this function will take precedence over the
            :meth:`LearnerProtocol.training_step` method.
        validate_step_fn: :class:`ValidationProtocol` or None
            The validation function to use for validation, only if it is
            provided alongside ``validation_dataloader``. If ``model`` implements
            the :class:`LearnerProtocol`, then this function will take precedence over
            the :meth:`LearnerProtocol.validation_step` method.
        max_epochs: int or None
            The maximum number of epochs to train for. Mututally exclusive
            with ``max_train_steps``.
        max_train_steps: int or None
            The maximum number of training steps to perform. Mututally exclusive
            with ``max_epochs``. If this value is greater than the length
            of ``train_dataloader``, then the training loop will recycle the data
            (i.e. more than one epoch) until the maximum number of training steps
            is reached.
        max_val_steps: int or None
            The maximum number of validation steps to perform per training
            epoch. If None, then the full validation set will be used.
        lr_scheduler: `_LRScheduler` or None
            The learning rate scheduler to use for training. If provided,
            this will be used to update the learning rate of the optimizer
            during training. If not provided, then the learning rate will
            not be adjusted within this function.
        device: str or `torch.device` or None
            The device to use for the training loop.
        dtype: `torch.dtype` or None
            The dtype to use for the training loop.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...


class LearnerProtocol:
    """
    This protocol represents the learner part of an active learning
    algorithm.

    This corresponds to a set of trainable parameters that are optimized,
    and subsequently used for inference and evaluation.

    The required methods make this classes that implement this protocol
    provide all the required functionality across all active learning steps.
    Keep in mind that, similar to all other protocols in this module, this
    is merely the required interface and not the actual implementation.

    See Also
    --------
    DriverProtocol : Uses the learner protocol in the active learning loop
    TrainingProtocol : Training step protocol
    ValidationProtocol : Validation step protocol
    InferenceProtocol : Inference step protocol
    TrainingLoop : Training loop protocol that can use a learner
    """

    def training_step(self, data: T, *args: Any, **kwargs: Any) -> None:
        """
        Implements the training logic for a single batch.

        This method will be called in training steps **only**, and not used
        for validation, query, or metrology steps. Specifically this means
        that gradients will be computed and used to update parameters.

        In cases where gradients are not needed, consider implementing the
        :meth:`validation_step` method instead.

        This should mirror the :class:`TrainingProtocol` definition, except that
        the model corresponds to this object.

        Parameters
        ----------
        data: T
            The data to train on. Typically assumed to be a batch
            of data.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...

    def validation_step(self, data: T, *args: Any, **kwargs: Any) -> None:
        """
        Implements the validation logic for a single batch.

        This can match the forward pass, without the need for weight updates.
        This method will be called in validation steps **only**, and not used
        for query or metrology steps. In those cases, implement the :meth:`inference_step`
        method instead.

        This should mirror the :class:`ValidationProtocol` definition, except that
        the model corresponds to this object.

        Parameters
        ----------
        data: T
            The data to validate on. Typically assumed to be a batch
            of data.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...

    def inference_step(self, data: T | S, *args: Any, **kwargs: Any) -> None:
        """
        Implements the inference logic for a single batch.

        This can match the forward pass exactly, but provides an opportunity
        to differentiate (or lack thereof, with no pun intended). Specifically,
        this method will be called during query and metrology steps.

        This should mirror the :class:`InferenceProtocol` definition, except that
        the model corresponds to this object.

        Parameters
        ----------
        data: T | S
            The data to infer on. Typically assumed to be a batch
            of data.
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...

    @property
    def parameters(self) -> Iterator[torch.Tensor]:
        """
        Returns an iterator over the parameters of the learner.

        If subclassing from `torch.nn.Module`, this will automatically return
        the parameters of the module.

        Returns
        -------
        Iterator[torch.Tensor]
            An iterator over the parameters of the learner.
        """
        ...

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Implements the forward pass for a single batch.

        This method is called between all active learning steps, and should
        contain the logic for how a model ingests data and produces predictions.

        Parameters
        ----------
        args: Any
            Additional arguments to pass to the model.
        kwargs: Any
            Additional keyword arguments to pass to the model.

        Returns
        -------
        Any
            The output of the model's forward pass.
        """
        ...


class DriverProtocol:
    """
    This protocol specifies the expected interface for an active learning
    driver: for a concrete implementation, refer to the :mod:`~physicsnemo.active_learning.driver`
    module instead. The specification is provided mostly as a reference, and for
    ease of type hinting to prevent circular imports.

    Attributes
    ----------
    learner: :class:`LearnerProtocol`
        The learner module that will be used as the surrogate within
        the active learning loop.
    query_strategies: list
        The query strategies that will be used for selecting data points to label.
        A list of :class:`QueryStrategy` instances can be included, and will sequentially be used to
        populate the ``query_queue`` that passes samples over to labeling.
    query_queue: :class:`AbstractQueue`
        The queue containing data samples to be labeled. :class:`QueryStrategy` instances
        should enqueue samples to this queue.
    label_strategy: :class:`LabelStrategy` or None
        The label strategy that will be used for labeling data points. In contrast
        to the other strategies, only a single label strategy is supported.
        This strategy will consume the ``query_queue`` and enqueue labeled data to
        the ``label_queue``.
    label_queue: :class:`AbstractQueue` or None
        The queue containing freshly labeled data. :class:`LabelStrategy` instances
        should enqueue labeled data to this queue, and the driver will subsequently
        serialize data contained within this queue to a persistent format.
    metrology_strategies: list or None
        The metrology strategies that will be used for assessing the performance
        of the surrogate. A list of :class:`MetrologyStrategy` instances can be included, and will sequentially
        be used to populate the ``metrology_queue`` that passes data over to the
        learner.
    training_pool: :class:`DataPool`
        The pool of data to be used for training. This data will be used to train
        the underlying model, and is assumed to be mutable in that additional data
        can be added to the pool over the course of active learning.
    validation_pool: :class:`DataPool` or None
        The pool of data to be used for validation. This data will be used for both
        conventional validation, as well as for metrology. This dataset is considered
        to be immutable, and should not be modified over the course of active learning.
        This dataset is considered optional, as both validation and metrology are.
    unlabeled_pool: :class:`DataPool` or None
        An optional pool of data to be used for querying and labeling. If supplied,
        this dataset can be depleted by a query strategy to select data points for labeling.
        In principle, this could also represent a generative model, i.e. not just a static
        dataset, but at a high level represents a distribution of data.

    See Also
    --------
    QueryStrategy : Query strategy protocol
    LabelStrategy : Label strategy protocol
    MetrologyStrategy : Metrology strategy protocol
    LearnerProtocol : Learner protocol
    DataPool : Data pool protocol
    AbstractQueue : Queue protocol
    """

    learner: LearnerProtocol
    query_strategies: list[QueryStrategy]
    query_queue: AbstractQueue[T]
    label_strategy: LabelStrategy | None
    label_queue: AbstractQueue[T] | None
    metrology_strategies: list[MetrologyStrategy] | None
    training_pool: DataPool[T]
    validation_pool: DataPool[T] | None
    unlabeled_pool: DataPool[T] | None

    def active_learning_step(self, *args: Any, **kwargs: Any) -> None:
        """
        Implements the active learning step.

        This step performs a single pass of the active learning loop, with the
        intended order being: training, metrology, query, labeling, with
        the metrology and labeling steps being optional.

        Parameters
        ----------
        args: Any
            Additional arguments to pass to the method.
        kwargs: Any
            Additional keyword arguments to pass to the method.
        """
        ...

    def _setup_logger(self) -> None:
        """
        Sets up the logger for the driver.

        The intended concrete method should account for the ability to
        scope logging, such that things like active learning iteration
        counts, etc. can be logged.
        """
        ...

    def attach_strategies(self) -> None:
        """
        Attaches all provided strategies.

        This method relies on the ``attach`` method of the strategies, which
        will subsequently give the strategy access to the driver's scope.

        Example use cases would be for any strategy (apart from label strategy)
        to access the underlying model (``LearnerProtocol``); for a query
        strategy to access the ``unlabeled_pool``; for a metrology strategy
        to access the ``validation_pool``.
        """
        for strategy in self.query_strategies:
            strategy.attach(self)
        if self.label_strategy:
            self.label_strategy.attach(self)
        if self.metrology_strategies:
            for strategy in self.metrology_strategies:
                strategy.attach(self)

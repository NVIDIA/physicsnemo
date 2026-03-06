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

import json
from queue import Queue
from typing import Any

import torch

from physicsnemo.active_learning import registry
from physicsnemo.active_learning.protocols import (
    DriverProtocol,
    LabelStrategy,
    MetrologyStrategy,
    QueryStrategy,
)

__all__ = ["ClassifierUQQuery", "DummyLabelStrategy", "F1Metrology"]


@registry.register("ClassifierUQQuery")
class ClassifierUQQuery(QueryStrategy):
    """
    This query strategy is representative of a more complex
    uncertainty-based query strategy: since our model produces
    logits, we can use the model's confidence in class label
    predictions to select data points for labeling: specifically,
    we pick ``max_samples`` each active learning iteration of
    the data points with the most uncertainty (closest to 0.5).
    """

    def __init__(self, max_samples: int):
        """
        Initialize the query strategy.

        Parameters
        ----------
        max_samples: int
            The maximum number of samples to query.
        """
        self.max_samples = max_samples

    def sample(self, query_queue: Queue) -> None:
        """
        Identify which data points that need labels by the query strategy.

        At a high level, this method will:
        1. Slice out the data indices not currently in the training set,
        2. Query the model for predictions on the 'unlabeled' data,
        3. Enqueue indices of data points with the class predictions closest to 0.5.

        Parameters
        ----------
        query_queue: Queue
            The queue to enqueue data to be labeled.
        """
        # strategy will be attached to a driver to access model and data
        model = self.driver.learner
        data = self.driver.train_datapool
        unlabeled_indices = data._sample_indices()
        # grab all of the data that's currently not labeled and obtain
        # predictions from the model
        unlabeled_coords = data.X_values[unlabeled_indices]
        unlabeled_coords = unlabeled_coords.to(model.device)
        model.eval()
        with torch.no_grad():
            pred_logits = model(unlabeled_coords)
            pred_probs = torch.sigmoid(pred_logits).squeeze()
        # find probabilities that are closet to 0.5; the lower this
        # value is, the more uncertain the model is
        uncertainties = torch.abs(pred_probs - 0.5)
        chosen_indices = torch.argsort(uncertainties)[: self.max_samples]
        # enqueue indices of the chosen data points
        for idx in chosen_indices:
            query_queue.put(unlabeled_indices[idx])

    def attach(self, driver: DriverProtocol) -> None:
        """Attach the driver to the query strategy."""
        self.driver = driver


@registry.register("DummyLabelStrategy")
class DummyLabelStrategy(LabelStrategy):
    """
    Since we have labels for all of our data already, this label strategy
    will simply just add the data points our model has chosen to the
    training set.
    """

    __is_external_process__ = False

    def __init__(self):
        super().__init__()

    def label(self, query_queue: Queue, serialize_queue: Queue) -> None:
        """
        Label the data points in the query queue.

        This is trivial because we are just passing indices from one queue
        to another, but in a real implementation this might call an external
        process to obtain ground truth data for a set of data points.

        Parameters
        ----------
        query_queue: Queue
            The queue to dequeue data from.
        serialize_queue: Queue
            The queue to enqueue labeled data to.
        """
        while not query_queue.empty():
            selected_idx = query_queue.get()
            serialize_queue.put(selected_idx)

    def attach(self, driver: DriverProtocol) -> None:
        """Attach the driver to the label strategy."""
        self.driver = driver


@registry.register("F1Metrology")
class F1Metrology(MetrologyStrategy):
    """
    While metrology is optional in the workflow, this provides observability
    into how the model is performing over the course of active learning.

    For a simple use case like the Moons dataset, the margin between validation
    and metrology is small, but for more complex use cases this strategy can
    potentially represent a workflow beyond simple metrics (e.g. using the model
    as a surrogate in a simulation loop).
    """

    def __init__(self):
        self.records = []

    def compute(self, *args: Any, **kwargs: Any) -> None:
        """Compute the F1 score of the model on the validation set."""
        model = self.driver.learner
        data = self.driver.train_datapool  # this can be any `DataPool`
        model.eval()
        indices = torch.arange(data.total_samples)
        input_data, labels = data.X_values[indices], data.y_values[indices]
        input_data = input_data.to(model.device)
        labels = labels.to(model.device)
        with torch.no_grad():
            # pack the entire dataset into a single batch
            pred_logits = model(input_data)
            pred_probs = torch.sigmoid(pred_logits).squeeze()
        pred_labels = torch.round(pred_probs)
        precision = self.precision(pred_labels, labels)
        recall = self.recall(pred_labels, labels)
        # compute the F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        iteration = self.driver.active_learning_step_idx
        num_train_samples = len(self.driver.train_datapool.train_indices)
        report = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "step": iteration,
            "num_train_samples": num_train_samples,
        }
        self.append(report)

    @staticmethod
    def precision(pred_labels: torch.Tensor, true_labels: torch.Tensor) -> float:
        """
        Calculate precision for class 0.

        Precision is the ratio of true positives to all predicted positives:
        how many of the samples predicted as class 0 are actually class 0.

        Parameters
        ----------
        pred_labels : torch.Tensor
            Predicted binary labels (0 or 1).
        true_labels : torch.Tensor
            Ground truth binary labels (0 or 1).

        Returns
        -------
        float
            Precision score for class 0.
        """
        true_positives = ((true_labels == 1) & (pred_labels == 1)).sum().item()
        predicted_positives = (pred_labels == 1).sum().item()
        if predicted_positives == 0:
            return 0.0
        return true_positives / predicted_positives

    @staticmethod
    def recall(pred_labels: torch.Tensor, true_labels: torch.Tensor) -> float:
        """
        Calculate recall for class 0.

        Recall is the ratio of true positives to all actual positives:
        how many of the actual class 0 samples were predicted as class 0.

        Parameters
        ----------
        pred_labels : torch.Tensor
            Predicted binary labels (0 or 1).
        true_labels : torch.Tensor
            Ground truth binary labels (0 or 1).

        Returns
        -------
        float
            Recall score for class 0.
        """
        true_positives = ((pred_labels == 0) & (true_labels == 0)).sum().item()
        actual_positives = (true_labels == 0).sum().item()
        if actual_positives == 0:
            return 0.0
        return true_positives / actual_positives

    def attach(self, driver: DriverProtocol) -> None:
        """Attach the driver to the metrology strategy."""
        self.driver = driver

    @property
    def is_attached(self) -> bool:
        """Check if the metrology strategy is attached to a driver."""
        return hasattr(self, "driver")

    def serialize_records(self, *args: Any, **kwargs: Any) -> None:
        """Serialize the records of the metrology strategy."""
        output_path = self.strategy_dir / f"step_{self.driver.active_learning_step_idx}"
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "f1_metrology.json", "w") as f:
            json.dump(self.records, f, indent=2)

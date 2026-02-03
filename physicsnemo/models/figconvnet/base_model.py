# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Any, Dict, Tuple

import torch

from physicsnemo.core.module import Module


class BaseModel(Module):
    r"""Base model class for FIGConvNet models.

    This abstract base class provides a common interface for all FIGConvNet-based
    models. It defines the standard methods that subclasses should implement for
    data conversion, loss computation, evaluation, and visualization.

    Subclasses must implement the abstract methods to provide model-specific
    functionality for training and inference pipelines.

    Parameters
    ----------
    None
        This base class does not define constructor parameters. Subclasses
        should define their own parameters and call ``super().__init__()``.

    Forward
    -------
    This base class does not define a forward method. Subclasses must implement
    their own forward method with appropriate inputs and outputs.

    Outputs
    -------
    Subclass-dependent. See specific model implementations.

    Note
    ----
    This is an abstract base class. Do not instantiate directly.

    See Also
    --------
    :class:`~physicsnemo.models.figconvnet.figconvunet.FIGConvUNet`
    :class:`~physicsnemo.core.module.Module`
    """

    def data_dict_to_input(self, data_dict: Dict[str, Any], **kwargs) -> Any:
        r"""Convert a data dictionary to model input format.

        This method transforms a dictionary of data (typically from a dataloader)
        into the appropriate input format expected by the model's forward method.

        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing input data from the dataloader. Expected keys
            and values depend on the specific model implementation.
        **kwargs : dict
            Additional keyword arguments for customizing the conversion.

        Returns
        -------
        Any
            Model input in the format expected by the forward method.
            The specific type depends on the model implementation.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def loss_dict(self, data_dict: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        r"""Compute the loss dictionary for training.

        This method computes all loss terms for the model given input data.
        The returned dictionary can contain multiple loss terms that will be
        combined during training.

        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing input data and ground truth labels.
        **kwargs : dict
            Additional keyword arguments for loss computation.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping loss names to their scalar tensor values.
            Common keys include ``"total_loss"``, ``"mse_loss"``, etc.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @torch.no_grad()
    def eval_dict(self, data_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        r"""Compute evaluation metrics for the model.

        This method computes evaluation metrics given input data. It is decorated
        with ``@torch.no_grad()`` to disable gradient computation during evaluation.

        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing input data and ground truth labels.
        **kwargs : dict
            Additional keyword arguments for evaluation.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping metric names to their values. Values can be
            scalars, tensors, or other types depending on the metric.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def image_pointcloud_dict(
        self, data_dict: Dict[str, Any], datamodule: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r"""Generate visualization data for images and point clouds.

        This method produces dictionaries containing data suitable for
        visualization of model inputs, outputs, and predictions. The returned
        data can be used for logging to visualization tools like TensorBoard
        or Weights & Biases.

        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing input data and model outputs.
        datamodule : Any
            The data module providing access to data transformations and
            metadata needed for visualization.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            A tuple containing:

            - ``image_dict``: Dictionary of 2D image visualizations
            - ``pointcloud_dict``: Dictionary of 3D point cloud visualizations

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

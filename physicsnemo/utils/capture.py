# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import functools
import logging
import os
import time
from contextlib import nullcontext
from logging import Logger
from typing import Any, Callable, Dict, NewType, Optional, Union

import torch

import physicsnemo
from physicsnemo.distributed import DistributedManager

float16 = NewType("float16", torch.float16)
bfloat16 = NewType("bfloat16", torch.bfloat16)
optim = NewType("optim", torch.optim)


class _StaticCapture(object):
    """Base class for StaticCapture decorator.

    This class should not be used, rather StaticCaptureTraining and StaticCaptureEvaluate
    should be used instead for training and evaluation functions.
    """

    # Grad scaler and checkpoint class variables use for checkpoint saving and loading
    # Since an instance of Static capture does not exist for checkpoint functions
    # one must use class functions to access state dicts
    _amp_scalers = {}
    _amp_scaler_checkpoints = {}
    _logger = logging.getLogger("capture")

    def __new__(cls, *args, **kwargs):
        obj = super(_StaticCapture, cls).__new__(cls)
        obj.amp_scalers = cls._amp_scalers
        obj.amp_scaler_checkpoints = cls._amp_scaler_checkpoints
        obj.logger = cls._logger
        return obj

    def __init__(
        self,
        model: "physicsnemo.Module",
        optim: Optional[optim] = None,
        logger: Optional[Logger] = None,
        use_graphs: bool = True,
        use_autocast: bool = True,
        use_gradscaler: bool = True,
        compile: bool = False,
        cuda_graph_warmup: int = 11,
        amp_type: Union[float16, bfloat16] = torch.float16,
        gradient_clip_norm: Optional[float] = None,
        label: Optional[str] = None,
    ):
        self.logger = logger if logger else self.logger
        # Checkpoint label (used for gradscaler)
        self.label = label if label else f"scaler_{len(self.amp_scalers.keys())}"

        # DDP fix
        if not isinstance(model, physicsnemo.models.Module) and hasattr(
            model, "module"
        ):
            model = model.module

        if not isinstance(model, physicsnemo.models.Module):
            self.logger.error("Model not a PhysicsNeMo Module!")
            raise ValueError("Model not a PhysicsNeMo Module!")
        if compile:
            model = torch.compile(model)

        self.model = model

        self.optim = optim
        self.eval = False
        self.no_grad = False
        self.gradient_clip_norm = gradient_clip_norm

        # Set up toggles for optimizations
        if not (amp_type == torch.float16 or amp_type == torch.bfloat16):
            raise ValueError("AMP type must be torch.float16 or torch.bfloat16")
        # CUDA device
        if "cuda" in str(self.model.device):
            # CUDA graphs
            if use_graphs and not self.model.meta.cuda_graphs:
                self.logger.warning(
                    f"Model {model.meta.name} does not support CUDA graphs, turning off"
                )
                use_graphs = False
            self.cuda_graphs_enabled = use_graphs

            # AMP GPU
            if not self.model.meta.amp_gpu:
                self.logger.warning(
                    f"Model {model.meta.name} does not support AMP on GPUs, turning off"
                )
                use_autocast = False
                use_gradscaler = False
            self.use_gradscaler = use_gradscaler
            self.use_autocast = use_autocast

            self.amp_device = "cuda"
            # Check if bfloat16 is suppored on the GPU
            if amp_type == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                self.logger.warning(
                    "Current CUDA device does not support bfloat16, falling back to float16"
                )
                amp_type = torch.float16
            self.amp_dtype = amp_type
            # Gradient Scaler
            scaler_enabled = self.use_gradscaler and amp_type == torch.float16
            self.scaler = self._init_amp_scaler(scaler_enabled, self.logger)

            self.replay_stream = torch.cuda.Stream(self.model.device)
        # CPU device
        else:
            self.cuda_graphs_enabled = False
            # AMP CPU
            if use_autocast and not self.model.meta.amp_cpu:
                self.logger.warning(
                    f"Model {model.meta.name} does not support AMP on CPUs, turning off"
                )
                use_autocast = False

            self.use_autocast = use_autocast
            self.amp_device = "cpu"
            # Only float16 is supported on CPUs
            # https://pytorch.org/docs/stable/amp.html#cpu-op-specific-behavior
            if amp_type == torch.float16 and use_autocast:
                self.logger.warning(
                    "torch.float16 not supported for CPU AMP, switching to torch.bfloat16"
                )
                amp_type = torch.bfloat16
            self.amp_dtype = torch.bfloat16
            # Gradient Scaler (not enabled)
            self.scaler = self._init_amp_scaler(False, self.logger)
            self.replay_stream = None

        if self.cuda_graphs_enabled:
            self.graph = torch.cuda.CUDAGraph()

        self.output = None
        self.iteration = 0
        self.cuda_graph_warmup = cuda_graph_warmup  # Default for DDP = 11

    def __call__(self, fn: Callable) -> Callable:
        self.function = fn

        @functools.wraps(fn)
        def decorated(*args: Any, **kwds: Any) -> Any:
            """Training step decorator function"""

            with torch.no_grad() if self.no_grad else nullcontext():
                if self.cuda_graphs_enabled:
                    self._cuda_graph_forward(*args, **kwds)
                else:
                    self._zero_grads()
                    self.output = self._amp_forward(*args, **kwds)

                if not self.eval:
                    # Update model parameters
                    self.scaler.step(self.optim)
                    self.scaler.update()

            return self.output

        return decorated

    def _cuda_graph_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward training step with CUDA graphs

        Returns
        -------
        Any
            Output of neural network forward
        """
        # Graph warm up
        if self.iteration < self.cuda_graph_warmup:
            self.replay_stream.wait_stream(torch.cuda.current_stream())
            self._zero_grads()
            with torch.cuda.stream(self.replay_stream):
                output = self._amp_forward(*args, **kwargs)
                self.output = output.detach()
            torch.cuda.current_stream().wait_stream(self.replay_stream)
        # CUDA Graphs
        else:
            # Graph record
            if self.iteration == self.cuda_graph_warmup:
                self.logger.warning(f"Recording graph of '{self.function.__name__}'")
                self._zero_grads()
                torch.cuda.synchronize()
                if DistributedManager().distributed:
                    torch.distributed.barrier()
                    # TODO: temporary workaround till this issue is fixed:
                    # https://github.com/pytorch/pytorch/pull/104487#issuecomment-1638665876
                    delay = os.environ.get("PHYSICSNEMO_CUDA_GRAPH_CAPTURE_DELAY", "10")
                    time.sleep(int(delay))
                with torch.cuda.graph(self.graph):
                    output = self._amp_forward(*args, **kwargs)
                    self.output = output.detach()
            # Graph replay
            self.graph.replay()

        self.iteration += 1
        return self.output

    def _zero_grads(self):
        """Zero gradients

        Default to `set_to_none` since this will in general have lower memory
        footprint, and can modestly improve performance.

        Note
        ----
        Zeroing gradients can potentially cause an invalid CUDA memory access in another
        graph. However if your graph involves gradients, you much set your gradients to none.
        If there is already a graph recorded that includes these gradients, this will error.
        Use the `NoGrad` version of capture to avoid this issue for inferencers / validators.
        """
        # Skip zeroing if no grad is being used
        if self.no_grad:
            return

        try:
            self.optim.zero_grad(set_to_none=True)
        except Exception:
            if self.optim:
                self.optim.zero_grad()
            # For apex optim support and eval mode (need to reset model grads)
            self.model.zero_grad(set_to_none=True)

    def _amp_forward(self, *args, **kwargs) -> Any:
        """Compute loss and gradients (if training) with AMP

        Returns
        -------
        Any
            Output of neural network forward
        """
        with torch.autocast(
            self.amp_device, enabled=self.use_autocast, dtype=self.amp_dtype
        ):
            output = self.function(*args, **kwargs)

        if not self.eval:
            # In training mode output should be the loss
            self.scaler.scale(output).backward()
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )

        return output

    def _init_amp_scaler(
        self, scaler_enabled: bool, logger: Logger
    ) -> torch.cuda.amp.GradScaler:
        # Create gradient scaler
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        # Store scaler in class variable
        self.amp_scalers[self.label] = scaler
        logging.debug(f"Created gradient scaler {self.label}")

        # If our checkpoint dictionary has weights for this scaler lets load
        if self.label in self.amp_scaler_checkpoints:
            try:
                scaler.load_state_dict(self.amp_scaler_checkpoints[self.label])
                del self.amp_scaler_checkpoints[self.label]
                self.logger.info(f"Loaded grad scaler state dictionary {self.label}.")
            except Exception as e:
                self.logger.error(
                    f"Failed to load grad scaler {self.label} state dict from saved "
                    + "checkpoints. Did you switch the ordering of declared static captures?"
                )
                raise ValueError(e)
        return scaler

    @classmethod
    def state_dict(cls) -> Dict[str, Any]:
        """Class method for accsessing the StaticCapture state dictionary.
        Use this in a training checkpoint function.

        Returns
        -------
        Dict[str, Any]
            Dictionary of states to save for file
        """
        scaler_states = {}
        for key, value in cls._amp_scalers.items():
            scaler_states[key] = value.state_dict()

        return scaler_states

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any]) -> None:
        """Class method for loading a StaticCapture state dictionary.
        Use this in a training checkpoint function.

        Returns
        -------
        Dict[str, Any]
            Dictionary of states to save for file
        """
        for key, value in state_dict.items():
            # If scaler has been created already load the weights
            if key in cls._amp_scalers:
                try:
                    cls._amp_scalers[key].load_state_dict(value)
                    cls._logger.info(f"Loaded grad scaler state dictionary {key}.")
                except Exception as e:
                    cls._logger.error(
                        f"Failed to load grad scaler state dict with id {key}."
                        + " Something went wrong!"
                    )
                    raise ValueError(e)
            # Otherwise store in checkpoints for later use
            else:
                cls._amp_scaler_checkpoints[key] = value

    @classmethod
    def reset_state(cls):
        cls._amp_scalers = {}
        cls._amp_scaler_checkpoints = {}


class StaticCaptureTraining(_StaticCapture):
    """A performance optimization decorator for PyTorch training functions.

    This class should be initialized as a decorator on a function that computes the
    forward pass of the neural network and loss function. The user should only call the
    defind training step function. This will apply optimizations including: AMP and
    Cuda Graphs.

    Parameters
    ----------
    model : physicsnemo.models.Module
        PhysicsNeMo Model
    optim : torch.optim
        Optimizer
    logger : Optional[Logger], optional
        PhysicsNeMo Launch Logger, by default None
    use_graphs : bool, optional
        Toggle CUDA graphs if supported by model, by default True
    use_amp : bool, optional
        Toggle AMP if supported by mode, by default True
    cuda_graph_warmup : int, optional
        Number of warmup steps for cuda graphs, by default 11
    amp_type : Union[float16, bfloat16], optional
        Auto casting type for AMP, by default torch.float16
    gradient_clip_norm : Optional[float], optional
        Threshold for gradient clipping
    label : Optional[str], optional
        Static capture checkpoint label, by default None

    Raises
    ------
    ValueError
        If the model provided is not a physicsnemo.models.Module. I.e. has no meta data.

    Example
    -------
    >>> # Create model
    >>> model = physicsnemo.models.mlp.FullyConnected(2, 64, 2)
    >>> input = torch.rand(8, 2)
    >>> output = torch.rand(8, 2)
    >>> # Create optimizer
    >>> optim = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> # Create training step function with optimization wrapper
    >>> @StaticCaptureTraining(model=model, optim=optim)
    ... def training_step(model, invar, outvar):
    ...     predvar = model(invar)
    ...     loss = torch.sum(torch.pow(predvar - outvar, 2))
    ...     return loss
    ...
    >>> # Sample training loop
    >>> for i in range(3):
    ...     loss = training_step(model, input, output)
    ...

    Note
    ----
    Static captures must be checkpointed when training using the `state_dict()` if AMP
    is being used with gradient scaler. By default, this requires static captures to be
    instantiated in the same order as when they were checkpointed. The label parameter
    can be used to relax/circumvent this ordering requirement.

    Note
    ----
    Capturing multiple cuda graphs in a single program can lead to potential invalid CUDA
    memory access errors on some systems. Prioritize capturing training graphs when this
    occurs.
    """

    def __init__(
        self,
        model: "physicsnemo.Module",
        optim: torch.optim,
        logger: Optional[Logger] = None,
        use_graphs: bool = True,
        use_amp: bool = True,
        compile: bool = False,
        cuda_graph_warmup: int = 11,
        amp_type: Union[float16, bfloat16] = torch.float16,
        gradient_clip_norm: Optional[float] = None,
        label: Optional[str] = None,
    ):
        super().__init__(
            model,
            optim,
            logger,
            use_graphs,
            use_amp,
            use_amp,
            compile,
            cuda_graph_warmup,
            amp_type,
            gradient_clip_norm,
            label,
        )


class StaticCaptureEvaluateNoGrad(_StaticCapture):

    """An performance optimization decorator for PyTorch no grad evaluation.

    This class should be initialized as a decorator on a function that computes run the
    forward pass of the model that does not require gradient calculations. This is the
    recommended method to use for inference and validation methods.

    Parameters
    ----------
    model : physicsnemo.models.Module
        PhysicsNeMo Model
    logger : Optional[Logger], optional
        PhysicsNeMo Launch Logger, by default None
    use_graphs : bool, optional
        Toggle CUDA graphs if supported by model, by default True
    use_amp : bool, optional
        Toggle AMP if supported by mode, by default True
    cuda_graph_warmup : int, optional
        Number of warmup steps for cuda graphs, by default 11
    amp_type : Union[float16, bfloat16], optional
        Auto casting type for AMP, by default torch.float16
    label : Optional[str], optional
        Static capture checkpoint label, by default None

    Raises
    ------
    ValueError
        If the model provided is not a physicsnemo.models.Module. I.e. has no meta data.

    Example
    -------
    >>> # Create model
    >>> model = physicsnemo.models.mlp.FullyConnected(2, 64, 2)
    >>> input = torch.rand(8, 2)
    >>> # Create evaluate function with optimization wrapper
    >>> @StaticCaptureEvaluateNoGrad(model=model)
    ... def eval_step(model, invar):
    ...     predvar = model(invar)
    ...     return predvar
    ...
    >>> output = eval_step(model, input)
    >>> output.size()
    torch.Size([8, 2])

    Note
    ----
    Capturing multiple cuda graphs in a single program can lead to potential invalid CUDA
    memory access errors on some systems. Prioritize capturing training graphs when this
    occurs.
    """

    def __init__(
        self,
        model: "physicsnemo.Module",
        logger: Optional[Logger] = None,
        use_graphs: bool = True,
        use_amp: bool = True,
        compile: bool = False,
        cuda_graph_warmup: int = 11,
        amp_type: Union[float16, bfloat16] = torch.float16,
        label: Optional[str] = None,
    ):
        super().__init__(
            model,
            None,
            logger,
            use_graphs,
            use_amp,
            compile,
            False,
            cuda_graph_warmup,
            amp_type,
            None,
            label,
        )
        self.eval = True  # No optimizer/scaler calls
        self.no_grad = True  # No grad context and no grad zeroing

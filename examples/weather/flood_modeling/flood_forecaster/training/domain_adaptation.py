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

r"""
Domain adaptation module for fine-tuning on target domain.

This module implements adversarial domain adaptation using gradient reversal layers
to enable transfer learning from source to target domains. Compatible with neuralop 2.0.0 API
and physicsnemo framework.

Key components:
- GradientReversal: Implements gradient reversal layer for adversarial training
- CNNDomainClassifier: CNN-based domain classifier for domain discrimination
- DomainAdaptationTrainer: Custom trainer for domain adaptation training loop
- adapt_model: High-level function to perform domain adaptation
"""

import os
import sys
import math
import random
from timeit import default_timer
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple, Any
from itertools import cycle

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from neuralop.training import AdamW
from neuralop.losses import LpLoss

import physicsnemo
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils.checkpoint import save_checkpoint, load_checkpoint


def _sanitize_args_for_json(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Recursively convert non-JSON-serializable objects in args_dict to serializable formats.
    
    This function handles:
    - DictConfig -> dict (using OmegaConf.to_container)
    - Other non-serializable objects are left as-is (will cause error if encountered)
    
    Parameters
    ----------
    args_dict : Dict[str, Any]
        Dictionary to sanitize (modified in-place).
    
    Returns
    -------
    Dict[str, Any]
        Sanitized dictionary (same object, modified in-place).
    """
    try:
        from omegaconf import DictConfig, OmegaConf
        has_omegaconf = True
    except ImportError:
        has_omegaconf = False
    
    def _convert_value(value: Any) -> Any:
        """Recursively convert DictConfig objects to dicts."""
        if has_omegaconf and isinstance(value, DictConfig):
            # Convert DictConfig to dict, resolving nested DictConfigs
            return OmegaConf.to_container(value, resolve=True)
        elif isinstance(value, dict):
            # Recursively process dictionary values
            return {k: _convert_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            # Recursively process list/tuple elements
            converted = [_convert_value(item) for item in value]
            return type(value)(converted)  # Preserve list/tuple type
        else:
            # Other types (int, float, str, bool, None) are already JSON-serializable
            return value
    
    # Process the dictionary recursively
    for key, value in list(args_dict.items()):
        args_dict[key] = _convert_value(value)
    
    return args_dict

from data_processing import LpLossWrapper

# Try to import comm for distributed training, fallback if not available
try:
    import neuralop.mpu.comm as comm
    _has_comm = True
except ImportError:
    _has_comm = False
    # Fallback: create a dummy comm module
    class _DummyComm:
        @staticmethod
        def get_local_rank():
            return 0
    comm = _DummyComm()

from datasets import FloodDatasetWithQueryPoints, NormalizedDataset
from utils.normalization import collect_all_fields, stack_and_fit_transform
from training.pretraining import create_scheduler
from training.trainer import save_model_checkpoint, _has_pytorch_submodules


class GradientReversalFunction(Function):
    r"""
    Custom autograd function for gradient reversal layer.
    
    This function implements the gradient reversal layer (GRL) used in adversarial
    domain adaptation. During forward pass, it returns the input unchanged. During
    backward pass, it negates and scales the gradients by lambda.
    
    Attributes
    ----------
    lambda_ : float
        Scaling factor for gradient reversal (typically scheduled during training).
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        r"""
        Forward pass: return input unchanged.

        Parameters
        ----------
        ctx : Any
            Context object to store lambda for backward pass.
        x : torch.Tensor
            Input tensor of arbitrary shape.
        lambda_ : float
            Scaling factor for gradient reversal.

        Returns
        -------
        torch.Tensor
            Cloned input tensor (same shape as input).
        """
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        r"""
        Backward pass: negate and scale gradients.

        Parameters
        ----------
        ctx : Any
            Context object containing lambda.
        grad_output : torch.Tensor
            Gradient from next layer.

        Returns
        -------
        Tuple[torch.Tensor, None]
            Tuple of (negated and scaled gradient, None for lambda).
        """
        return grad_output.neg().mul(ctx.lambda_), None


class GradientReversal(physicsnemo.Module):
    r"""
    Gradient reversal layer module for adversarial domain adaptation.
    
    This module wraps the GradientReversalFunction to provide a learnable
    gradient reversal layer. The lambda parameter can be dynamically updated
    during training to schedule the strength of adversarial training.
    
    Parameters
    ----------
    lambda_max : float, optional, default=1.0
        Maximum lambda value (typically 1.0).

    Forward
    -------
    x : torch.Tensor
        Input tensor of arbitrary shape.

    Outputs
    -------
    torch.Tensor
        Output tensor (same shape as input, but gradients will be reversed).
    
    Attributes
    ----------
    lambda_ : float
        Current scaling factor for gradient reversal.
    """
    
    def __init__(self, lambda_max: float = 1.0):
        r"""
        Initialize gradient reversal layer.

        Parameters
        ----------
        lambda_max : float, optional, default=1.0
            Maximum lambda value (typically 1.0).
        """
        super().__init__(meta=ModelMetaData(name="GradientReversal"))
        self.lambda_ = lambda_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass through gradient reversal layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        torch.Tensor
            Output tensor (same shape as input, but gradients will be reversed).
        """
        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if not isinstance(x, torch.Tensor):
                raise ValueError(
                    f"Expected input to be torch.Tensor, got {type(x)}"
                )
            if x.numel() == 0:
                raise ValueError(
                    f"Expected non-empty input tensor, got tensor with shape {tuple(x.shape)}"
                )
        
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, val: float) -> None:
        r"""
        Update lambda value for gradient reversal.

        Parameters
        ----------
        val : float
            New lambda value.
        """
        self.lambda_ = val


class CNNDomainClassifier(physicsnemo.Module):
    r"""
    CNN-based domain classifier for adversarial domain adaptation.
    
    This classifier takes latent features from the GINO model and predicts
    whether they come from the source or target domain. The gradient reversal
    layer ensures that the feature extractor learns domain-invariant features.
    
    Architecture:
        - Gradient Reversal Layer (GRL)
        - Convolutional layers (configurable)
        - Adaptive average pooling
        - Fully connected layer for binary classification

    Parameters
    ----------
    in_channels : int
        Number of input channels (should match fno_hidden_channels).
    lambda_max : float
        Maximum lambda for gradient reversal layer.
    da_cfg : Dict[str, Any]
        Configuration dict with keys:
        - conv_layers: List of dicts with 'out_channels', 'kernel_size', 'pool_size'
        - fc_dim: Output dimension of final fully connected layer

    Forward
    -------
    x : torch.Tensor
        Input features of shape :math:`(B, C, H, W)` where :math:`B` is batch size,
        :math:`C` is channels, and :math:`H, W` are spatial dimensions.

    Outputs
    -------
    torch.Tensor
        Logits for binary classification of shape :math:`(B, D_{fc})` where
        :math:`D_{fc}` is the fully connected layer dimension.
    """
    
    def __init__(self, in_channels: int, lambda_max: float, da_cfg: Dict[str, Any]):
        r"""
        Initialize domain classifier.

        Parameters
        ----------
        in_channels : int
            Number of input channels (should match fno_hidden_channels).
        lambda_max : float
            Maximum lambda for gradient reversal layer.
        da_cfg : Dict[str, Any]
            Configuration dict with keys:
            - conv_layers: List of dicts with 'out_channels', 'kernel_size', 'pool_size'
            - fc_dim: Output dimension of final fully connected layer

        Raises
        ------
        ValueError
            If required keys are missing from ``da_cfg`` or if conv_layers is empty.
        """
        # Convert DictConfig to regular dict if needed (for JSON serialization)
        # This must be done BEFORE super().__init__() so PhysicsNeMo's __new__ captures
        # the converted dict, not the DictConfig
        # OmegaConf.to_container recursively converts nested DictConfigs to dicts
        try:
            from omegaconf import DictConfig, OmegaConf
            if isinstance(da_cfg, DictConfig):
                # Convert to regular dict, resolving all nested DictConfigs recursively
                da_cfg = OmegaConf.to_container(da_cfg, resolve=True)
        except ImportError:
            # OmegaConf not available, assume da_cfg is already a dict
            pass
        except Exception:
            # If conversion fails for any reason, try to manually convert
            # This is a fallback in case OmegaConf.to_container doesn't work
            if hasattr(da_cfg, '__dict__'):
                # Try to convert manually
                da_cfg = dict(da_cfg)
        
        super().__init__(meta=ModelMetaData(name="CNNDomainClassifier"))
        
        # CRITICAL: Sanitize _args to convert any DictConfig objects to regular dicts
        # PhysicsNeMo's __new__ captures arguments before __init__ runs, so _args
        # may contain DictConfig objects that need to be converted for JSON serialization
        # We need to sanitize the entire _args structure, not just __args__
        if hasattr(self, '_args'):
            # Recursively sanitize the entire _args dictionary
            # This will convert any DictConfig objects anywhere in the structure
            _sanitize_args_for_json(self._args)
        if not da_cfg.get("conv_layers"):
            raise ValueError("da_cfg must contain 'conv_layers' list")
        if "fc_dim" not in da_cfg:
            raise ValueError("da_cfg must contain 'fc_dim'")
            
        self.grl = GradientReversal(lambda_max=lambda_max)
        layers = []
        c_in = in_channels
        
        for layer_spec in da_cfg["conv_layers"]:
            out_channels = layer_spec["out_channels"]
            kernel_size = layer_spec.get("kernel_size", 3)
            pool_size = layer_spec.get("pool_size", 2)
            
            layers.extend([
                nn.Conv2d(
                    c_in, out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size)
            ])
            c_in = out_channels
            
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.conv_net = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in, da_cfg["fc_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass through domain classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape :math:`(B, C, H, W)` where :math:`B` is batch size,
            :math:`C` is channels, and :math:`H, W` are spatial dimensions.

        Returns
        -------
        torch.Tensor
            Logits for binary classification of shape :math:`(B, D_{fc})` where
            :math:`D_{fc}` is the fully connected layer dimension.
        """
        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected 4D input tensor (B, C, H, W), got {x.ndim}D tensor "
                    f"with shape {tuple(x.shape)}"
                )
        
        # Apply gradient reversal, then conv layers, then flatten and classify
        x = self.grl(x)  # (B, C, H, W)
        x = self.conv_net(x)  # (B, C, 1, 1) after adaptive pooling
        x = x.view(x.size(0), -1)  # (B, C)
        return self.fc(x)  # (B, fc_dim)


class DomainAdaptationTrainer:
    r"""
    Custom trainer for domain adaptation compatible with neuralop 2.0.0.
    
    Implements adversarial domain adaptation using gradient reversal layers (GRL).
    The training process alternates between:
    1. Task loss: Regression loss on both source and target domains
    2. Adversarial loss: Domain classification loss with reversed gradients
    
    The GRL lambda is scheduled during training to gradually increase the strength
    of adversarial training.
    
    Parameters
    ----------
    model : nn.Module
        The main model to train (should support return_features=True).
    data_processor : nn.Module, optional
        Data processor for preprocessing/postprocessing.
    domain_classifier : nn.Module
        Domain classifier module.
    device : str or torch.device, optional, default="cuda"
        Device to train on ('cuda', 'cpu', or torch.device).
    verbose : bool, optional, default=True
        Whether to print training progress.
    logger : Any, optional
        Optional logger instance (if None, uses print when verbose=True).
    
    Attributes
    ----------
    model : nn.Module
        The main GINO model (wrapped with GINOWrapper).
    data_processor : nn.Module, optional
        Data processor for preprocessing/postprocessing.
    domain_classifier : nn.Module
        CNN-based domain classifier.
    device : str or torch.device
        Device to train on.
    verbose : bool
        Whether to print training progress.
    _eval_interval : int
        Interval for evaluation (default: 1).
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_processor: Optional[nn.Module],
        domain_classifier: nn.Module,
        device: Union[str, torch.device] = "cuda",
        verbose: bool = True,
        logger: Optional[Any] = None,
        wandb_step_offset: int = 0,
    ):
        r"""
        Initialize domain adaptation trainer.

        Parameters
        ----------
        model : nn.Module
            The main model to train (should support return_features=True).
        data_processor : nn.Module, optional
            Optional data processor for preprocessing.
        domain_classifier : nn.Module
            Domain classifier module.
        device : str or torch.device, optional, default="cuda"
            Device to train on ('cuda', 'cpu', or torch.device).
        verbose : bool, optional, default=True
            Whether to print training progress.
        logger : Any, optional
            Optional logger instance (if None, uses print when verbose=True).
        wandb_step_offset : int, optional, default=0
            Step offset for wandb logging to continue from pretraining step count.
        """
        self.model = model
        self.data_processor = data_processor
        self.domain_classifier = domain_classifier
        self.device = device
        self.verbose = verbose
        self.logger = logger
        self.wandb_step_offset = wandb_step_offset
        self._eval_interval = 1
        
    def train_domain_adaptation(
        self,
        src_loader: DataLoader,
        tgt_loader: Union[DataLoader, List[DataLoader]],
        optimizer,
        scheduler,
        training_loss,
        class_loss_weight: float = 0.1,
        adaptation_epochs: int = 100,
        save_every: int = None,
        save_dir: Union[str, Path] = "./ckpt",
        resume_from_dir: Union[str, Path] = None,
        resume_classifier_from_dir: Union[str, Path] = None,
        val_loaders: Optional[Dict[str, DataLoader]] = None,
    ):
        r"""
        Domain-adaptation training loop with adversarial classifier.
        
        Handles both a single target DataLoader and a list of target DataLoaders.
        This implementation exactly matches the original neuralop trainer's
        train_domain_adaptation method.

        Parameters
        ----------
        src_loader : DataLoader
            Source domain dataloader.
        tgt_loader : DataLoader or List[DataLoader]
            Target domain dataloader (single or list).
        optimizer : torch.optim.Optimizer
            Optimizer for both model and classifier.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler.
        training_loss : callable
            Loss function for main task.
        class_loss_weight : float, optional, default=0.1
            Weight for domain classification loss.
        adaptation_epochs : int, optional, default=100
            Number of epochs to train.
        save_every : int, optional
            Interval at which to save checkpoints.
        save_dir : str or Path, optional, default="./ckpt"
            Directory to save checkpoints.
        resume_from_dir : str or Path, optional
            Directory to resume training from.
        resume_classifier_from_dir : str or Path, optional
            Directory to resume classifier from.
        val_loaders : Dict[str, DataLoader], optional
            Dict of validation dataloaders.

        Returns
        -------
        nn.Module
            Trained model.
        """
        self.model = self.model.to(self.device)
        self.domain_classifier = self.domain_classifier.to(self.device)
        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
        
        # Domain classification loss (binary cross-entropy)
        adv_criterion = nn.BCEWithLogitsLoss()
        
        # Optionally resume model and classifier state
        start_epoch = 0
        if resume_from_dir is not None:
            start_epoch = self._resume_from_checkpoint(resume_from_dir, optimizer, scheduler)
        
        # Optionally resume classifier from separate directory (fallback mechanism)
        if resume_classifier_from_dir is not None:
            classifier_loaded = False
            resume_classifier_dir = Path(resume_classifier_from_dir)
            
            # Try PhysicsNeMo format first (classifier saved as second model)
            try:
                metadata_dict = {}
                load_checkpoint(
                    path=str(resume_classifier_dir),
                    models=[self.domain_classifier],
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    epoch=None,  # Load latest
                    metadata_dict=metadata_dict,
                    device=self.device,
                )
                msg = f"Loaded classifier from PhysicsNeMo checkpoint: {resume_classifier_dir}"
                if self.logger:
                    self.logger.info(msg)
                elif self.verbose:
                    print(msg)
                classifier_loaded = True
            except (FileNotFoundError, KeyError, ValueError):
                # Fall back to old format (separate classifier_state_dict.pt file)
                ckpt = resume_classifier_dir / "classifier_state_dict.pt"
                if ckpt.exists():
                    self.domain_classifier.load_state_dict(
                        torch.load(str(ckpt), map_location=self.device)
                    )
                    msg = f"Loaded classifier from neuralop checkpoint: {ckpt}"
                    if self.logger:
                        self.logger.info(msg)
                    elif self.verbose:
                        print(msg)
                    classifier_loaded = True
            
            if not classifier_loaded:
                msg = f"Warning: Could not load classifier from {resume_classifier_from_dir}"
                if self.logger:
                    self.logger.warning(msg)
                elif self.verbose:
                    print(msg)
        
        val_loaders = val_loaders or {}
        
        # Handle both single loader and list of loaders
        if not isinstance(tgt_loader, list):
            tgt_loaders = [tgt_loader]
        else:
            tgt_loaders = tgt_loader
        
        # Determine iteration strategy based on source loader
        base_batches = len(src_loader)
        total_iters = adaptation_epochs * base_batches
        
        # Create cycling iterators for all loaders
        src_iter = cycle(src_loader)
        tgt_iters = [cycle(loader) for loader in tgt_loaders]
        
        msg1 = f"Starting domain adaptation training for {adaptation_epochs} epochs"
        msg2 = f"Source samples: {len(src_loader.dataset)}, Target loaders: {len(tgt_loaders)}"
        if self.logger:
            self.logger.info(msg1)
            self.logger.info(msg2)
        elif self.verbose:
            print(msg1)
            print(msg2)
        
        for epoch in range(start_epoch + 1, adaptation_epochs):
            self.on_epoch_start(epoch)
            self.model.train()
            self.domain_classifier.train()
            if self.data_processor is not None:
                self.data_processor.train()
            
            total_reg, total_adv = 0.0, 0.0
            
            # Progress bar
            pbar = tqdm(
                range(base_batches),
                desc=f"DA Epoch {epoch}/{adaptation_epochs}",
                disable=not self.verbose,
                file=sys.stdout
            )
            
            for batch_idx in pbar:
                # Update GRL lambda (scheduled from 0 to lambda_max)
                # Formula: lambda_val = 2.0 / (1.0 + exp(-10 * p)) - 1.0
                # where p = (epoch * base_batches + batch_idx) / total_iters
                p = (epoch * base_batches + batch_idx) / total_iters
                lambda_val = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
                self.domain_classifier.grl.set_lambda(lambda_val)
                
                # Randomly select one target domain from the list for this training step
                chosen_tgt_iter = random.choice(tgt_iters)
                
                src_batch = next(src_iter)
                tgt_batch = next(chosen_tgt_iter)
                
                # Preprocess batches
                if self.data_processor is not None:
                    s = self.data_processor.preprocess(src_batch)
                    t = self.data_processor.preprocess(tgt_batch)
                else:
                    s = {k: v.to(self.device) for k, v in src_batch.items() if torch.is_tensor(v)}
                    t = {k: v.to(self.device) for k, v in tgt_batch.items() if torch.is_tensor(v)}
                
                # Forward pass with feature extraction
                # Extract features using return_features=True
                # Features should be in shape (batch, channels, H, W) for 2D
                try:
                    out_s, f_s = self.model(**s, return_features=True)
                    out_t, f_t = self.model(**t, return_features=True)
                except TypeError as e:
                    # Fallback if model doesn't support return_features
                    raise RuntimeError(
                        "Model must support return_features=True for domain adaptation. "
                        "Ensure model is wrapped with GINOWrapper."
                    ) from e
                
                # Postprocess outputs (after feature extraction, before loss)
                if self.data_processor is not None:
                    out_s, s = self.data_processor.postprocess(out_s, s)
                    out_t, t = self.data_processor.postprocess(out_t, t)
                
                # Regression loss on source and target
                # Note: training_loss expects (y_pred, **sample) where sample contains 'y'
                reg_loss = training_loss(out_s, **s) + training_loss(out_t, **t)
                
                # Prepare features for domain classifier
                # Features from GINOWrapper are already in shape (batch, channels, H, W)
                # Concatenate source and target features along batch dimension
                if f_s.dim() != 4 or f_t.dim() != 4:
                    raise ValueError(
                        f"Expected 4D features (B, C, H, W), got f_s.shape={f_s.shape}, f_t.shape={f_t.shape}. "
                        "Ensure GINOWrapper returns features in correct format."
                    )
                feats = torch.cat([f_s, f_t], dim=0)
                
                # Domain classification adversarial loss
                logits = self.domain_classifier(feats).squeeze(1)
                labels = torch.cat([
                    torch.ones(f_s.size(0), device=self.device),
                    torch.zeros(f_t.size(0), device=self.device)
                ], dim=0).float()
                adv_loss = adv_criterion(logits, labels)
                
                # Combined loss
                loss = reg_loss + class_loss_weight * adv_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_reg += reg_loss.item()
                total_adv += adv_loss.item()
                
                # Update progress bar
                if self.verbose:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'reg': f'{reg_loss.item():.4f}',
                        'adv': f'{adv_loss.item():.4f}',
                        'lambda': f'{lambda_val:.3f}'
                    })
            
            # Step scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_reg + total_adv)
            else:
                scheduler.step()
            
            avg_reg = total_reg / base_batches
            avg_adv = total_adv / base_batches
            msg = f"[DA Epoch {epoch}] reg={avg_reg:.4f}, adv={avg_adv:.4f}, lambda={lambda_val:.3f}"
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
            
            # Validation (if val_loaders provided)
            if val_loaders and (epoch % self.eval_interval == 0 or epoch == adaptation_epochs - 1):
                self._evaluate(val_loaders, training_loss, epoch)
            
            # Optional checkpointing using PhysicsNeMo checkpoint system
            if save_every is not None and (epoch % save_every == 0):
                # Only save on rank 0 in distributed training
                should_save = True
                if DistributedManager.is_initialized():
                    dist_manager = DistributedManager()
                    should_save = (dist_manager.rank == 0)
                elif _has_comm:
                    should_save = (comm.get_local_rank() == 0)
                
                if should_save:
                    sd = Path(save_dir)
                    sd.mkdir(parents=True, exist_ok=True)
                    
                    # Determine model parallel rank
                    model_parallel_rank = 0
                    if DistributedManager.is_initialized():
                        dist_manager = DistributedManager()
                        if "model_parallel" in dist_manager.group_names:
                            model_parallel_rank = dist_manager.group_rank("model_parallel")
                    
                    # Save model(s) - handle PyTorch submodules if needed
                    model_to_save = self.model
                    if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                        model_to_save = self.model.module
                    
                    model_saved_separately = save_model_checkpoint(
                        model=model_to_save,
                        save_dir=sd,
                        epoch=epoch,
                        model_parallel_rank=model_parallel_rank,
                    )
                    
                    # Prepare models list for PhysicsNeMo (includes domain classifier)
                    models_to_save = []
                    if not model_saved_separately:
                        models_to_save.append(model_to_save)
                    # Always add domain classifier as second model
                    models_to_save.append(self.domain_classifier)
                    
                    # Save checkpoint with both models and training state
                    save_checkpoint(
                        path=str(sd),
                        models=models_to_save if not model_saved_separately else [self.domain_classifier],
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=None,
                        epoch=epoch,
                        metadata={"stage": "domain_adaptation", "epoch": epoch},
                    )
                    
                    msg = f"Saved DA checkpoint at epoch {epoch}"
                    if self.logger:
                        self.logger.info(msg)
                    elif self.verbose:
                        print(msg)
        
        # Save final checkpoint (outside epoch loop) using PhysicsNeMo checkpoint system
        should_save = True
        if DistributedManager.is_initialized():
            dist_manager = DistributedManager()
            should_save = (dist_manager.rank == 0)
        elif _has_comm:
            should_save = (comm.get_local_rank() == 0)
        
        if should_save:
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            
            # Determine model parallel rank
            model_parallel_rank = 0
            if DistributedManager.is_initialized():
                dist_manager = DistributedManager()
                if "model_parallel" in dist_manager.group_names:
                    model_parallel_rank = dist_manager.group_rank("model_parallel")
            
            # Save model(s) - handle PyTorch submodules if needed
            model_to_save = self.model
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model_to_save = self.model.module
            
            final_epoch = adaptation_epochs - 1
            model_saved_separately = save_model_checkpoint(
                model=model_to_save,
                save_dir=sd,
                epoch=final_epoch,
                model_parallel_rank=model_parallel_rank,
            )
            
            # Prepare models list for PhysicsNeMo (includes domain classifier)
            models_to_save = []
            if not model_saved_separately:
                models_to_save.append(model_to_save)
            # Always add domain classifier as second model
            models_to_save.append(self.domain_classifier)
            
            # Save checkpoint with both models and training state
            save_checkpoint(
                path=str(sd),
                models=models_to_save if not model_saved_separately else [self.domain_classifier],
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=final_epoch,
                metadata={"stage": "domain_adaptation", "final_epoch": True, "epoch": final_epoch},
            )
            
            msg = "Saved final DA checkpoint using PhysicsNeMo format"
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
        
        msg = "Domain adaptation training completed!"
        if self.logger:
            self.logger.info(msg)
        elif self.verbose:
            print(msg)
        return self.model
    
    def on_epoch_start(self, epoch):
        r"""Stub called at the start of each epoch."""
        self.epoch = epoch
        return None
    
    @property
    def eval_interval(self):
        r"""Evaluation interval (default 1)."""
        return getattr(self, '_eval_interval', 1)
    
    @eval_interval.setter
    def eval_interval(self, value):
        self._eval_interval = value
    
    def _evaluate(
        self, 
        val_loaders: Dict[str, DataLoader], 
        loss_fn: Any, 
        epoch: int
    ) -> None:
        r"""
        Evaluate model on validation loaders.

        Parameters
        ----------
        val_loaders : Dict[str, DataLoader]
            Dictionary of validation dataloaders.
        loss_fn : callable
            Loss function to use for evaluation.
        epoch : int
            Current epoch number (for logging).
        """
        self.model.eval()
        if self.data_processor is not None:
            self.data_processor.eval()
        
        with torch.no_grad():
            for name, loader in val_loaders.items():
                total_loss = 0.0
                n_samples = 0
                
                for sample in loader:
                    try:
                        if self.data_processor is not None:
                            sample = self.data_processor.preprocess(sample)
                        else:
                            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}
                        
                        out = self.model(**sample)
                        
                        if self.data_processor is not None:
                            out, sample = self.data_processor.postprocess(out, sample)
                        
                        # Loss function expects (y_pred, **sample) where sample contains 'y'
                        loss = loss_fn(out, **sample)
                        total_loss += loss.item()
                        n_samples += sample.get("y", out).shape[0] if isinstance(sample.get("y"), torch.Tensor) else out.shape[0]
                    except Exception as e:
                        msg = f"Error evaluating on {name}: {e}"
                        if self.logger:
                            self.logger.error(msg)
                        elif self.verbose:
                            print(msg)
                        raise
                
                avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
                msg = f"  Eval {name}: loss={avg_loss:.6f}"
                if self.logger:
                    self.logger.info(msg)
                elif self.verbose:
                    print(msg)
    
    def _save_checkpoint(
        self, 
        save_dir: Union[str, Path], 
        optimizer: torch.optim.Optimizer, 
        scheduler: Any, 
        epoch: int, 
        save_classifier: bool = False
    ) -> None:
        r"""
        Save training checkpoint using PhysicsNeMo checkpoint system.
        
        This method saves both the main model and domain classifier (if requested)
        using PhysicsNeMo's checkpoint format.

        Parameters
        ----------
        save_dir : str or Path
            Directory to save checkpoint.
        optimizer : torch.optim.Optimizer
            Optimizer instance.
        scheduler : Any
            Scheduler instance.
        epoch : int
            Current epoch number.
        save_classifier : bool, optional, default=False
            Whether to save classifier state dict.
        """
        save_dir = Path(save_dir)
        
        # Only save on rank 0 in distributed training
        should_save = True
        if DistributedManager.is_initialized():
            dist_manager = DistributedManager()
            should_save = (dist_manager.rank == 0)
        elif _has_comm:
            should_save = (comm.get_local_rank() == 0)
        
        if not should_save:
            return
        
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine model parallel rank
            model_parallel_rank = 0
            if DistributedManager.is_initialized():
                dist_manager = DistributedManager()
                if "model_parallel" in dist_manager.group_names:
                    model_parallel_rank = dist_manager.group_rank("model_parallel")
            
            # Save model(s) - handle PyTorch submodules if needed
            model_to_save = self.model
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model_to_save = self.model.module
            
            model_saved_separately = save_model_checkpoint(
                model=model_to_save,
                save_dir=save_dir,
                epoch=epoch,
                model_parallel_rank=model_parallel_rank,
            )
            
            # Prepare models list for PhysicsNeMo
            models_to_save = []
            if not model_saved_separately:
                models_to_save.append(model_to_save)
            if save_classifier:
                models_to_save.append(self.domain_classifier)
            
            # Save checkpoint with models and training state
            save_checkpoint(
                path=str(save_dir),
                models=models_to_save if models_to_save else None,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=epoch,
                metadata={"stage": "domain_adaptation", "epoch": epoch},
            )
            
            msg = f"Saved checkpoint to {save_dir}"
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
        except Exception as e:
            msg = f"Error saving checkpoint to {save_dir}: {e}"
            if self.logger:
                self.logger.error(msg)
            elif self.verbose:
                print(msg)
            raise
    
    def _resume_from_checkpoint(
        self, 
        resume_dir: Union[str, Path], 
        optimizer: torch.optim.Optimizer, 
        scheduler: Any
    ) -> int:
        r"""
        Resume training from checkpoint using PhysicsNeMo checkpoint system.
        
        Supports both PhysicsNeMo format (new) and neuralop format (old) for backward compatibility.

        Parameters
        ----------
        resume_dir : str or Path
            Directory containing checkpoint.
        optimizer : torch.optim.Optimizer
            Optimizer instance (will be updated with checkpoint state).
        scheduler : Any
            Scheduler instance (will be updated with checkpoint state).

        Returns
        -------
        int
            Epoch number to resume from (0 if no checkpoint found).
        """
        resume_dir = Path(resume_dir)
        
        if not resume_dir.exists():
            msg = f"Resume directory does not exist: {resume_dir}"
            if self.logger:
                self.logger.warning(msg)
            elif self.verbose:
                print(msg)
            return 0

        try:
            # Try PhysicsNeMo format first (new format)
            checkpoint_loaded = False
            resume_epoch = 0
            metadata_dict = {}
            
            try:
                # Try to load using PhysicsNeMo format
                # Prepare models list (main model + domain classifier if available)
                models_to_load = [self.model]
                if hasattr(self, 'domain_classifier') and self.domain_classifier is not None:
                    models_to_load.append(self.domain_classifier)
                
                resume_epoch = load_checkpoint(
                    path=str(resume_dir),
                    models=models_to_load,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    epoch=None,  # Load latest
                    metadata_dict=metadata_dict,
                    device=self.device,
                )
                
                if self.logger:
                    self.logger.info("Loaded checkpoint using PhysicsNeMo format")
                checkpoint_loaded = True
            except (FileNotFoundError, KeyError, ValueError) as e:
                # Fall back to neuralop format (old format)
                if self.logger:
                    self.logger.info(f"PhysicsNeMo checkpoint not found, trying neuralop format: {e}")
                
                # Check for neuralop checkpoint files
                if (resume_dir / "best_model_state_dict.pt").exists():
                    save_name = "best_model"
                elif (resume_dir / "model_state_dict.pt").exists():
                    save_name = "model"
                else:
                    msg = f"No checkpoint found in {resume_dir} (tried both formats)"
                    if self.logger:
                        self.logger.warning(msg)
                    elif self.verbose:
                        print(msg)
                    return 0

                # Load using neuralop format
                from neuralop.training.training_state import load_training_state
                
                self.model, optimizer, scheduler, _, resume_epoch = load_training_state(
                    save_dir=resume_dir,
                    save_name=save_name,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                
                # Try to load domain classifier if it exists
                classifier_path = resume_dir / "classifier_state_dict.pt"
                if classifier_path.exists() and hasattr(self, 'domain_classifier') and self.domain_classifier is not None:
                    self.domain_classifier.load_state_dict(
                        torch.load(str(classifier_path), map_location=self.device)
                    )
                    if self.logger:
                        self.logger.info("Loaded domain classifier from neuralop checkpoint")
                
                if self.logger:
                    self.logger.info(f"Loaded checkpoint using neuralop format: {save_name}")
                checkpoint_loaded = True

            if checkpoint_loaded and resume_epoch is not None:
                msg = f"Resumed from epoch {resume_epoch}"
                if self.logger:
                    self.logger.info(msg)
                elif self.verbose:
                    print(msg)
                return resume_epoch
            else:
                return 0
                
        except Exception as e:
            msg = f"Error loading checkpoint from {resume_dir}: {e}"
            if self.logger:
                self.logger.error(msg)
            elif self.verbose:
                print(msg)
            raise


def adapt_model(
    model: nn.Module,
    normalizers: Dict[str, Any],
    data_processor: Optional[nn.Module],
    config: Any,
    device: Union[str, torch.device],
    is_logger: bool,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_data_config: Any,
    logger: Optional[Any] = None,
    wandb_step_offset: int = 0,
) -> Tuple[nn.Module, nn.Module, "DomainAdaptationTrainer"]:
    r"""
    Perform domain adaptation on target domain data.

    This function orchestrates the domain adaptation training process:
    1. Loads and normalizes target domain data
    2. Creates domain classifier
    3. Sets up optimizer and scheduler
    4. Trains model with adversarial domain adaptation

    Parameters
    ----------
    model : nn.Module
        Pretrained model (should be wrapped with GINOWrapper).
    normalizers : Dict[str, Any]
        Dictionary of normalizers from pretraining.
    data_processor : nn.Module, optional
        Data processor instance.
    config : Any
        Configuration object (OmegaConf DictConfig).
    device : str or torch.device
        Device to train on ('cuda', 'cpu', or torch.device).
    is_logger : bool
        Whether this process is the logger (for distributed training).
    source_train_loader : DataLoader
        Source domain training dataloader.
    source_val_loader : DataLoader
        Source domain validation dataloader.
    target_data_config : Any
        Target data configuration (OmegaConf DictConfig).
    logger : Any, optional
        Optional logger instance (physicsnemo PythonLogger or compatible).
    wandb_step_offset : int, optional, default=0
        Step offset for wandb logging to continue from pretraining step count.
        Currently not used but reserved for future wandb integration.

    Returns
    -------
    Tuple[nn.Module, nn.Module, DomainAdaptationTrainer]
        Tuple of (adapted_model, domain_classifier, trainer).

    Raises
    ------
    ValueError
        If required config keys are missing.
    RuntimeError
        If model doesn't support return_features.
    """
    if logger is None:
        # Fallback to print if no logger provided (for backward compatibility)
        def log_info(msg: str) -> None:
            print(msg)

        logger = type("Logger", (), {"info": lambda self, msg: log_info(msg)})()

    logger.info("Starting domain adaptation on source + target...")
    
    # Validate inputs
    if not hasattr(model, 'fno_hidden_channels'):
        raise AttributeError(
            "Model must have 'fno_hidden_channels' attribute. "
            "Ensure model is a GINO model or wrapped with GINOWrapper."
        )
    
    # Create target dataset
    logger.info(f"Loading target dataset from: {target_data_config.root}")
    try:
        target_full_dataset = FloodDatasetWithQueryPoints(
        data_root=target_data_config.root,
        n_history=target_data_config.n_history,
        xy_file=getattr(target_data_config, "xy_file", None),
        query_res=getattr(target_data_config, "query_res", [64, 64]),
        static_files=getattr(target_data_config, "static_files", []),
        dynamic_patterns=getattr(target_data_config, "dynamic_patterns", {}),
        boundary_patterns=getattr(target_data_config, "boundary_patterns", {}),
        raise_on_smaller=True,
        skip_before_timestep=getattr(target_data_config, "skip_before_timestep", 0),
        noise_type=getattr(target_data_config, "noise_type", "none"),
        noise_std=getattr(target_data_config, "noise_std", None)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load target dataset from {target_data_config.root}: {e}") from e
    
    # Split into train/val
    train_sz_target = int(0.9 * len(target_full_dataset))
    target_train_raw, target_val_raw = random_split(
        target_full_dataset,
        [train_sz_target, len(target_full_dataset) - train_sz_target]
    )
    logger.info(f"Target domain: total={len(target_full_dataset)}, train={train_sz_target}, val={len(target_val_raw)}")
    
    # Move normalizers to CPU temporarily
    for nm in normalizers.values():
        nm.to('cpu')
    
    # Collect and normalize target training data
    logger.info("Collecting and normalizing target training data...")
    geom_t_tr, static_t_tr, boundary_t_tr, dyn_t_tr, tgt_t_tr = collect_all_fields(target_train_raw, True)
    _, big_target_train = stack_and_fit_transform(
        geom_t_tr, static_t_tr, boundary_t_tr, dyn_t_tr, tgt_t_tr,
        normalizers=normalizers, fit_normalizers=False
    )
    target_train_ds = NormalizedDataset(
        geometry=big_target_train["geometry"],
        static=big_target_train["static"],
        boundary=big_target_train["boundary"],
        dynamic=big_target_train["dynamic"],
        target=big_target_train["target"],
        query_res=target_data_config.query_res
    )
    
    # Collect and normalize target validation data
    logger.info("Collecting and normalizing target validation data...")
    geom_t_val, static_t_val, boundary_t_val, dyn_t_val, tgt_t_val = collect_all_fields(target_val_raw, True)
    _, big_target_val = stack_and_fit_transform(
        geom_t_val, static_t_val, boundary_t_val, dyn_t_val, tgt_t_val,
        normalizers=normalizers, fit_normalizers=False
    )
    target_val_ds = NormalizedDataset(
        geometry=big_target_val["geometry"],
        static=big_target_val["static"],
        boundary=big_target_val["boundary"],
        dynamic=big_target_val["dynamic"],
        target=big_target_val["target"],
        query_res=target_data_config.query_res
    )
    target_val_loader = DataLoader(
        target_val_ds, batch_size=target_data_config.batch_size, shuffle=False
    )
    
    # Create domain classifier
    logger.info("Creating domain classifier...")
    try:
        da_cfg = config.training.get("da_classifier", {})
        if not da_cfg:
            raise ValueError("config.training.da_classifier is required for domain adaptation")
        domain_classifier = CNNDomainClassifier(
            model.fno_hidden_channels,
            config.training.get("da_lambda_max", 1.0),
            da_cfg,
        ).to(device)
    except (AttributeError, KeyError) as e:
        raise ValueError(
            f"Invalid domain adaptation configuration: {e}. "
            "Ensure config.training.da_classifier contains 'conv_layers' and 'fc_dim'."
        ) from e
    
    # Create optimizer and scheduler
    adapt_lr = config.training.get("adapt_learning_rate", config.training.get("learning_rate", 1e-4))
    weight_decay = config.training.get("weight_decay", 1e-4)
    optimizer_adapt = AdamW(
        list(model.parameters()) + list(domain_classifier.parameters()),
        lr=adapt_lr,
        weight_decay=weight_decay,
    )
    logger.info(f"Optimizer: AdamW (lr={adapt_lr}, weight_decay={weight_decay})")
    scheduler_adapt = create_scheduler(optimizer_adapt, config, logger)

    # Create loss - wrap with LpLossWrapper to filter out unexpected kwargs
    # Get loss type from config, default to 'l2'
    def create_loss(loss_type_str, default="l2"):
        """Helper function to create loss function from string."""
        loss_type_str = loss_type_str.lower()
        if loss_type_str == "l1":
            return LpLossWrapper(LpLoss(d=2, p=1)), "l1"
        elif loss_type_str == "l2":
            return LpLossWrapper(LpLoss(d=2, p=2)), "l2"
        else:
            if logger:
                logger.warning(f"Unknown loss type '{loss_type_str}', defaulting to '{default}'")
            return LpLossWrapper(LpLoss(d=2, p=2)), default
    
    training_loss_type = config.training.get("training_loss", "l2")
    training_loss_fn, training_loss_name = create_loss(training_loss_type)
    if logger:
        logger.info(f"Using {training_loss_name.upper()} loss for domain adaptation training")
    
    # Use testing_loss for evaluation if specified, otherwise use training_loss
    # Note: Currently domain adaptation uses the same loss for both training and evaluation
    # but we create it here for consistency and future extensibility
    testing_loss_type = config.training.get("testing_loss", training_loss_type)
    eval_loss_fn, eval_loss_name = create_loss(testing_loss_type, default=training_loss_name)
    if testing_loss_type.lower() != training_loss_type.lower() and logger:
        logger.info(f"Note: testing_loss specified but domain adaptation currently uses training_loss for evaluation")

    # Create custom domain adaptation trainer
    trainer_adapt = DomainAdaptationTrainer(
        model=model,
        data_processor=data_processor,
        domain_classifier=domain_classifier,
        device=device,
        verbose=is_logger,
        logger=logger,
        wandb_step_offset=wandb_step_offset,
    )
    trainer_adapt.eval_interval = 1  # Evaluate every epoch

    # Train with domain adaptation
    save_dir = os.path.join(config.checkpoint.get("save_dir", "./checkpoints"), "adapt")
    logger.info(f"Starting training... Checkpoints will be saved to: {save_dir}")
    logger.info(f"Starting domain adaptation training for {config.training.get('n_epochs_adapt', 50)} epochs")
    trainer_adapt.train_domain_adaptation(
        src_loader=source_train_loader,
        tgt_loader=DataLoader(target_train_ds, batch_size=target_data_config.batch_size, shuffle=True),
        optimizer=optimizer_adapt,
        scheduler=scheduler_adapt,
        training_loss=training_loss_fn,
        # da_class_loss_weight controls adversarial training strength
        # Default 0.0 disables adversarial training (standard fine-tuning)
        # Set to positive value (e.g., 0.1) to enable domain adaptation
        class_loss_weight=config.training.get("da_class_loss_weight", 0.0),
        adaptation_epochs=config.training.get("n_epochs_adapt", 50),
        save_every=None,  # Save at end only, or set to save interval
        save_dir=save_dir,
        resume_from_dir=config.checkpoint.get("resume_from_adapt", None),
        resume_classifier_from_dir=config.checkpoint.get("resume_from_adapt", None),
        val_loaders={"source_val": source_val_loader, "target_val": target_val_loader},
    )
    
    return model, domain_classifier, trainer_adapt

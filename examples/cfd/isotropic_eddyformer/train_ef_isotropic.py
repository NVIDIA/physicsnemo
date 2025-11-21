import hydra
from typing import Tuple
from torch import Tensor
from omegaconf import DictConfig

import os
import numpy as np

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from physicsnemo.models.eddyformer import EddyFormer, EddyFormerConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining
from physicsnemo.launch.logging import PythonLogger, LaunchLogger


class Re94(Dataset):

    root: str
    t: float

    n: int = 50
    dt: float = 0.1

    def __init__(self, root: str, split: str, *, t: float = 0.5) -> None:
        """
        """
        super().__init__()
        self.root = root
        self.t = t

        self.file = []
        for fname in sorted(os.listdir(root)):
            if fname.startswith(split):
                self.file.append(fname)

    @property
    def stride(self) -> int:
        k = int(self.t / self.dt)
        assert self.dt * k == self.t
        return k

    @property
    def samples_per_file(self) -> int:
        return self.n - self.stride + 1

    def __len__(self) -> int:
        return len(self.file) * self.samples_per_file

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        file_idx, time_idx = divmod(idx, self.samples_per_file)

        data = np.load(f"{self.root}/{self.file[file_idx]}", allow_pickle=True).item()
        return torch.from_numpy(data["u"][time_idx]), torch.from_numpy(data["u"][time_idx + self.stride])

@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def isotropic_trainer(cfg: DictConfig) -> None:
    """
    """
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="re94_ef")
    log.file_logging()
    LaunchLogger.initialize()  # PhysicsNeMo launch logger

    # define model, loss, optimizer
    model = EddyFormer(
        idim=cfg.model.idim,
        odim=cfg.model.odim,
        hdim=cfg.model.hdim,
        num_layers=cfg.model.num_layers,
        cfg=EddyFormerConfig(**cfg.model.layer_config),
    ).to(dist.device)
    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)

    # define dataset and dataloader
    dataset = Re94(root=cfg.training.dataset, split="train", t=cfg.training.t)
    dataloader = DataLoader(dataset, cfg.training.batch_size, shuffle=True)

    # define forward passes for training
    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        logger=log,
        use_amp=False,
        use_graphs=False
    )
    def training_step(input, target):
        pred = torch.vmap(model)(input)
        loss = loss_fun(pred, target)
        return loss

    for epoch in range(cfg.training.num_epochs):

        for input, target in dataloader:

            input = input.to(dist.device)
            target = target.to(dist.device)
            loss = training_step(input, target)

            with LaunchLogger("train", epoch=epoch) as logger:
                logger.log_minibatch({"Training loss": loss.item()})

    log.success("Training completed")


if __name__ == "__main__":
    isotropic_trainer()

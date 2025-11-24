import hydra
from typing import Tuple
from torch import Tensor
from omegaconf import DictConfig

import os
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.models.eddyformer import EddyFormer, EddyFormerConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining
from physicsnemo.launch.utils import save_checkpoint
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
    log.file_logging(f"{cfg.training.result_dir}/log.txt")
    LaunchLogger.initialize()  # PhysicsNeMo launch logger

    # define model and optimizer
    model = EddyFormer(
        idim=cfg.model.idim,
        odim=cfg.model.odim,
        hdim=cfg.model.hdim,
        num_layers=cfg.model.num_layers,
        use_scale=cfg.model.use_scale,
        cfg=EddyFormerConfig(**cfg.model.layer_config),
    ).to(dist.device)

    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)
        log.success("Initialized DDP training")

    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)

    # define dataset and dataloader
    dataset = Re94(root=cfg.training.dataset, split="train", t=cfg.training.t)
    dataloader = DataLoader(dataset, cfg.training.batch_size, shuffle=True)

    # define relative l2 error as the loss function
    def loss_fun(pred: Tensor, target: Tensor) -> Tensor:
        return torch.linalg.norm(pred - target) / torch.linalg.norm(target)

    # define training step
    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        logger=log,
        use_amp=False,
        use_graphs=False
    )
    def training_step(input: Tensor, target: Tensor) -> Tensor:
        pred = torch.vmap(model)(input)
        loss = torch.vmap(loss_fun)(pred, target)
        return torch.mean(loss)

    it = 0
    log.info("Training started")

    for epoch in range(cfg.training.num_epochs):
        for it, (input, target) in enumerate(dataloader, it):

            input = input.to(dist.device)
            target = target.to(dist.device)
            loss = training_step(input, target)

            with LaunchLogger("train", epoch=epoch) as logger:
                logger.log_minibatch({"Training loss": loss.item()})

            if it and it % cfg.training.ckpt_every == 0 and dist.rank == 0:
                save_checkpoint(f"{cfg.training.result_dir}/ckpt.pt", model, optimizer, epoch=it)

    log.success("Training completed")
    save_checkpoint(f"{cfg.training.result_dir}/ckpt.pt", model, optimizer)


if __name__ == "__main__":
    isotropic_trainer()

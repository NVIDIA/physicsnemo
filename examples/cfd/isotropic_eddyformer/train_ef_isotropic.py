import hydra
from tqdm import tqdm

from typing import Tuple
from torch import Tensor
from omegaconf import DictConfig

import os
import collections
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.models.eddyformer import EddyFormer, EddyFormerConfig
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.launch.utils import save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger


def rel_l2(pred: Tensor, target: Tensor) -> Tensor:
    return torch.linalg.norm(pred - target) / torch.linalg.norm(target)

class Re94(Dataset):

    root: str
    t: float

    n: int = 50
    dt: float = 0.1

    def __init__(self, root: str, split: str, *, t: float = 0.5,
                 n: int = 50, dt: float = 0.1) -> None:
        """
        """
        super().__init__()
        self.root = root
        self.t = t

        self.n = n
        self.dt = dt

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

    def metric(self, pred: Tensor, target: Tensor) -> dict[str, float]:
        """
        """
        l2 = [rel_l2(pred[..., i], target[..., i]).item() for i in range(3)]
        return { f"err_{ax}": value for ax, value in (zip("xyz", l2)) }

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
        cfg=EddyFormerConfig(
            basis=cfg.model.layer_config.basis,
            mesh=tuple(cfg.model.layer_config.mesh),
            mode=tuple(cfg.model.layer_config.mode),
            mode_les=tuple(cfg.model.layer_config.mode_les),
            kernel_size=tuple(cfg.model.layer_config.kernel_size),
            kernel_size_les=tuple(cfg.model.layer_config.kernel_size_les),
            ffn_dim=cfg.model.layer_config.ffn_dim,
            activation=cfg.model.layer_config.activation,
            num_heads=cfg.model.layer_config.num_heads,
            heads_dim=cfg.model.layer_config.heads_dim,
        ),
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

    testset = Re94(root=cfg.training.dataset, split="test", t=cfg.training.t, n=40, dt=0.5)
    testloader = DataLoader(testset, batch_size=None)

    # define training step
    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        logger=log,
        use_graphs=False,
        use_amp=cfg.training.amp,
        compile=cfg.training.compile
    )
    def training_step(input: Tensor, target: Tensor) -> Tensor:
        pred = torch.vmap(model)(input)
        loss = torch.vmap(rel_l2)(pred, target)
        return torch.mean(loss)

    # define evaluation step
    @StaticCaptureEvaluateNoGrad(
        model=model,
        logger=log,
        use_graphs=False,
        use_amp=cfg.training.amp,
        compile=cfg.training.compile
    )
    def forward_eval(input):
        return model(input)

    it = 0

    model.train()
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

            if it and it % cfg.training.test_every == 0:

                model.eval()
                metrics = collections.defaultdict(float)

                for input, target in tqdm(testloader, desc="Test"):

                    input = input.to(dist.device)
                    target = target.to(dist.device)

                    pred = forward_eval(input)
                    metric = testset.metric(pred, target)

                    for key, value in metric.items():
                        metrics[key] += value / len(testset)

                with LaunchLogger("test", epoch=epoch) as logger:
                    logger.log_minibatch(metrics)

                model.train()

    log.success("Training completed")
    save_checkpoint(f"{cfg.training.result_dir}/ckpt.pt", model, optimizer)


if __name__ == "__main__":
    isotropic_trainer()

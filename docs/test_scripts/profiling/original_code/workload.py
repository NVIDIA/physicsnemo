import time

import torch

# For hydra:
import hydra
from omegaconf import DictConfig

# For dataloader
from torch.utils.data import DataLoader

# Import the dataset:
from dataset import RandomNoiseDataset

#  For the model code:
from attn import Block

def loss_fn(output_data):
    # All except the first dim:
    dims = tuple(range(len(output_data.shape)))
    # Just a silly loss function:
    output_data = output_data**2.
    loss = torch.sum(output_data, dims[1:])
    return loss.mean()

def workload(cfg):
    ds = RandomNoiseDataset(cfg["shape"])
    
    loader = DataLoader(
        ds, 
        batch_size=cfg["batch_size"], 
        shuffle = True,
    )
    
    
    # Initialize the model:
    model = Block(
        dim = cfg["shape"][-1],
        num_heads = cfg.model["num_heads"],
        qkv_bias  = cfg.model["qkv_bias"] ,
        attn_drop = cfg.model["attn_drop"],
        proj_drop = cfg.model["proj_drop"],
    ).to("cuda")
    
    if cfg["train"]:
        opt = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
    times = []
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        image = batch["image"]
        image = image.to("cuda")
        output = model(image)
        if cfg["train"]:
            opt.zero_grad()
            # Compute the loss:
            loss = loss_fn(output)
            # Do the gradient calculation:
            loss.backward()
            # Apply the gradients
            opt.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"Finished step {i} in {end - start:.4f} seconds")
        times.append(end - start)
        start = time.perf_counter()

    times = torch.tensor(times)
    # Drop first and last:
    avg_time = times[1:-1].mean()
    # compute throughput too:
    throughput = cfg["batch_size"] / avg_time
    print(f"Average time per iteration: {avg_time:.3f} ({throughput:.3f} examples / s)")

@hydra.main(version_base="1.3", config_path="../", config_name="cfg")
def main(config: DictConfig):
    


    workload(config)



if __name__ == "__main__":

    main()
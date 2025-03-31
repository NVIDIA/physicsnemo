import time

import torch
torch.backends.cudnn.benchmark = True

# For hydra:
import hydra
from omegaconf import DictConfig
from pathlib import Path

# For dataloader
from torch.utils.data import DataLoader

# Import the dataset:
from dataset import RandomNoiseDataset

#  For the model code:
from attn import Block

# Import profiling hooks from physicsnemo:
from physicsnemo.utils.profiling import Profiler, profile, annotate

def loss_fn(output_data):
    # All except the first dim:
    dims = tuple(range(len(output_data.shape)))
    # Just a silly loss function:
    output_data = output_data**2.
    loss = torch.sum(output_data, dims[1:])
    return loss.mean()

@profile
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
    
    model = torch.compile(model)

    if cfg["train"]:
        opt = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
    times = []    
    with Profiler() as p:
        start = time.perf_counter()
        for i, batch in enumerate(loader):
            image = batch["image"]
            image = image.to("cuda")
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                with annotate(domain="forward", color="blue"):
                    output = model(image)
            if cfg["train"]:
                opt.zero_grad()
                # Compute the loss:
                loss = loss_fn(output)
                # Do the gradient calculation:
                with annotate(domain="backward", color="green"):
                    loss.backward()
                    # Apply the gradients
                    opt.step()
            p.step()
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
    
    # configure the profiling tools:
    p = Profiler()
    
    for key, val in config.profile.items():
        # This is not the mandatory way to enable tools
        # I've set up the config to have the keys match
        # the registered profilers.  You can do it manually
        # too such as `p.enable("torch")`
        if val: p.enable(key)
            
    # The profiler has to be initilized before use.  Using it in a context
    # will do it automatically, but to use it as a decorator we should do
    # it manually here:


    p.initialize()
    print(p)

    workload(config)



if __name__ == "__main__":

    main()  
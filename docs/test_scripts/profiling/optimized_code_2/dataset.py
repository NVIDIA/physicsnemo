import torch
import numpy as np
from torch.utils.data import Dataset


class RandomNoiseDataset(Dataset):
    """
    Random normal distribution dataset.
    
    Mean AND STD of the distribution is set to the index
    of the sample requested.
    
    Length is hardcoded to 64.
    
    (Don't use this anywhere that isn't an example of how to write non-performant python code!)
    
    """

    def __init__(self, image_shape, ):
        """
        Arguments:
            image_shape (string): Shape of a single example to generate
        """
        self.shape = list(image_shape)


    def __len__(self):
        return 256

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Generate the raw data:
        raw = self.gen_single_image(idx)

        sample = {
            'image' : raw
        }
        
        return sample

    def gen_single_image(self, idx):
        
        return torch.normal(idx, idx, self.shape, device="cuda" )

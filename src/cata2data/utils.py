import torch
import numpy as np

def collate_fn_irregular_cutouts(batch):
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    # Convert tuple of (x,y) tuples to tuple of x and y lists.
    x, y = list(zip(*batch))
    # if y is arraylike
    if isinstance(y[0], np.ndarray) or isinstance(y[0], list):
        y = torch.as_tensor(np.concatenate([target.astype(np.float32) for target in y]))
    else:
        y = torch.as_tensor(y)
    x = [torch.as_tensor(sample) for sample in x]
    return x, y

def collate_fn_regular_cutouts(batch):
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    # Convert tuple of (x,y) tuples to tuple of x and y lists.
    x, y = list(zip(*batch))
    # if y is arraylike
    if isinstance(y[0], np.ndarray) or isinstance(y[0], list):
        y = torch.as_tensor(np.concatenate([target.astype(np.float32) for target in y]))
    else:
        y = torch.as_tensor(y)
    x = torch.stack([sample for sample in x])
    return x, y

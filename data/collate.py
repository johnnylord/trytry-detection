import torch
from torch.utils.data._utils.collate import default_collate

def maskv3_collate_fn(data):
    imgs, masks, targets = zip(*data)

    imgs = default_collate(imgs)
    targets = default_collate(targets)

    return imgs, masks, targets

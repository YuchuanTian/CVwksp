import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn


def image_transform(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def load_data(data_dir, batch_size=8, seed=42):
    dset_loaders = {}
    train_bs = batch_size
    test_bs = batch_size

    all_data = datasets.ImageFolder(root=data_dir, transform=image_transform())
    # assign size of train set & test set
    data_num = len(all_data.imgs)
    train_size = int(0.8 * len(all_data))
    test_size = data_num - train_size

    torch.manual_seed(seed)
    train_data, test_data = torch.utils.data.random_split(all_data, [train_size, test_size])


    dset_loaders["tr"] = DataLoader(train_data, batch_size=train_bs, shuffle=True,
                                           num_workers=4, drop_last=False)
    dset_loaders["te"] = DataLoader(test_data, batch_size=test_bs, shuffle=False,
                                           num_workers=4, drop_last=False)
    return data_num, dset_loaders
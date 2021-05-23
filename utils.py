
import numpy as np
import torch
from torch.utils.data import random_split


# Grab a GPU if there is one
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} device: {}".format(device, torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using {}".format(device))


class Start(object):
    """Return beginning of ECG"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ecg = sample["ecg"]
        sample["ecg"] = ecg[:, :self.output_size]

        return sample


class RandomCrop(object):
    """Randomly crop ECG"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ecg = sample["ecg"]

        size = ecg.shape[-1]
        start = np.random.randint(0, size - self.output_size)
        end = start + self.output_size
        sample["ecg"] = ecg[:, start:end]

        return sample


def split_dataset(dataset, val_split=0.3):
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def sort_batch(batch):
    """Sort batch by follow up time (descending)"""
    fu_time = batch["fu_time"]
    ind = np.argsort(fu_time)
    ind = torch.flip(ind, dims=[0])
    return {k: v[ind] for k, v in batch.items()}


import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


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
        ecg, measures = sample["ecg"], sample["measures"]

        return {"ecg": ecg[:, :self.output_size], "measures": measures}


class RandomCrop(object):
    """Randomly crop ECG"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ecg, measures = sample["ecg"], sample["measures"]

        size = ecg.shape[-1]
        start = np.random.randint(0, size - self.output_size)
        end = start + self.output_size

        return {"ecg": ecg[:, start:end], "measures": measures}


def compute_means(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    means = torch.zeros(8)
    counts = torch.zeros(8)
    for sample in dataloader:
        measures = sample["measures"]
        counts += (~measures.isnan()).sum(0)
        measures[measures.isnan()] = 0
        means += measures.sum(0)

    means = means / counts
    return means.numpy().round()


def split_dataset(dataset, val_split=0.3):
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def sort_batch(batch):
    """Sort batch by follow up time (descending)"""
    fu_time = batch["fu_time"]
    ind = np.argsort(fu_time)[::-1]
    return {k: v[ind] for k, v in batch.items()}

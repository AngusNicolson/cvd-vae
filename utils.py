
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

    train_dataset.dataset.load_ecg = False
    # Calculate std and mean of training set to scale data
    data = np.zeros((len(train_dataset), 8))
    for i in range(len(train_dataset)):
        data[i] = train_dataset[i]["measures"]
    train_dataset.dataset.load_ecg = True

    std = np.nanstd(data, axis=0)
    mean = np.nanmean(data, axis=0)

    train_dataset.dataset.means = mean
    val_dataset.dataset.means = mean

    train_dataset.dataset.std = std
    val_dataset.dataset.std = std

    train_dataset.dataset.replace_missing = True
    val_dataset.dataset.replace_missing = True
    return train_dataset, val_dataset

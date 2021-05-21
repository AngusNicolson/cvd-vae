
import numpy as np
import torch
from torch.utils.data import DataLoader


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

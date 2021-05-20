
import numpy as np
import torch
from scipy.signal import firwin, lfilter


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

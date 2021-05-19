
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


def clean_ecg(data, filter_size=251, thresh=800):
    """Bandpass FIR filter then remove ECGs with |values| > thresh"""
    filtered_data = filter_ecg(data, filter_size=filter_size)

    # Find ECG containing high values after filter
    maxes = filtered_data.max(axis=1).max(axis=1)
    locs = np.where(maxes > thresh)[0]

    # Remove
    filtered_data = np.delete(filtered_data, locs, axis=0)

    return filtered_data


def filter_ecg(data, filter_size=251, low=3, high=45):
    """Bandpass FIR filter input data"""
    nyq_rate = 500 / 2  # 500 Hz data

    fir_filter = firwin(filter_size, [low / nyq_rate, high / nyq_rate], pass_zero="bandpass")
    filtered_data = lfilter(fir_filter, 1, data, axis=1)

    # Filter corrupts beginning signal with initial conditions
    return filtered_data[:, filter_size:, :]

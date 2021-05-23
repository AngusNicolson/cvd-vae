
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """ECG dataset"""

    def __init__(self, json_file: str, prefix: str = "", transform=None, load_ecg=True):
        """
        Args:
            json_file (string): Path to the json metadata file.
            prefix (string): Add a prefix to ECG file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(json_file, "r") as fp:
            self.metadata = json.load(fp)

        self.prefix = prefix
        self.transform = transform
        self.pids = list(np.sort(list(self.metadata.keys())))
        self.ecg_type = "strip"
        self.load_ecg = load_ecg

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid_metadata = self.metadata[self.pids[idx]]

        if self.load_ecg:
            ecg_name = f"{self.prefix}/{pid_metadata['data'][self.ecg_type]}"
            ecg = np.load(ecg_name).astype("f4")
        else:
            ecg = None

        survival = pid_metadata["cvd"]
        incident = np.array(survival["incident"]).astype("int32")
        fu_time = np.array(survival["fu_time"]).astype("float32")

        sample = {'ecg': ecg, "incident": incident, "fu_time": fu_time}

        if self.load_ecg and self.transform:
            sample = self.transform(sample)

        return sample

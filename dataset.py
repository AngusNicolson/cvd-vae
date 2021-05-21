
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from utils import Start, RandomCrop
from torchvision.transforms import Compose


class ECGDataset(Dataset):
    """ECG dataset"""

    def __init__(self, json_file: str, prefix: str = "", transform=None, replace_missing=False):
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
        self.means = np.zeros(8)
        self.replace_missing = replace_missing

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid_metadata = self.metadata[self.pids[idx]]

        ecg_name = f"{self.prefix}/{pid_metadata['data'][self.ecg_type]}"
        ecg = np.load(ecg_name).astype("f4")

        measures = list(pid_metadata["measures"].values())
        measures = np.array(measures)
        measures = measures.astype('f4')

        if self.replace_missing:
            missing = np.where(np.isnan(measures))
            measures[missing] = np.take(self.means, missing[0])

        sample = {'ecg': ecg, 'measures': measures}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def compute_means(self, batch_size):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=False)
        means = torch.zeros(8)
        counts = torch.zeros(8)
        for sample in dataloader:
            measures = sample["measures"]
            counts += (~measures.isnan()).sum(0)
            measures[measures.isnan()] = 0
            means += measures.sum(0)

        means = means / counts
        self.means = means.numpy().round()


if __name__ == "__main__":
    dataset = ECGDataset(
        "/home/angus/labs/vicente/ecg/data/processed/metadata.json",
        "/home/angus/labs/vicente/ecg/data/processed/"
    )

    fig = plt.figure(figsize=(12, 8))
    n = 4
    for i in range(n):
        sample = dataset[i]

        print(i, sample['ecg'].shape, sample['measures'].shape)

        ax = plt.subplot(n, 1, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.plot(sample["ecg"][0, :])

    plt.savefig("/home/angus/dataset_test.png")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    for i_batch, batch in enumerate(dataloader):
        print(i_batch, batch['ecg'].shape, batch['measures'].shape)

        if i_batch == 2:
            break

    transform = Compose([
        Start(output_size=1024)
    ])

    dataset = ECGDataset(
        "/home/angus/labs/vicente/ecg/data/processed/metadata.json",
        "/home/angus/labs/vicente/ecg/data/processed/",
        transform
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    for i_batch, batch in enumerate(dataloader):
        print(i_batch, batch['ecg'].shape, batch['measures'].shape)

        fig = plt.figure(figsize=(12, 8))
        for i in range(n):
            ax = plt.subplot(n, 1, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.plot(batch["ecg"][i, 0, :])

        plt.savefig(f"/home/angus/dataset_test_transform{i_batch}.png")
        if i_batch == 2:
            break

    dataset.compute_means(64)
    dataset.replace_missing = True
    print(dataset[9])

    transform = Compose([
        RandomCrop(output_size=1024)
    ])

    dataset = ECGDataset(
        "/home/angus/labs/vicente/ecg/data/processed/metadata.json",
        "/home/angus/labs/vicente/ecg/data/processed/",
        transform
    )

    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.set_title('Sample #{}'.format(i))
        ax.plot(dataset[0]["ecg"][0, :])
        ax.set_ylim([-80, 220])
    plt.tight_layout()
    plt.savefig(f"/home/angus/dataset_test_crop.png")

    print("Done!")

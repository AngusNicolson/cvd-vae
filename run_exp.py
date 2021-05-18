
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from scipy.signal import firwin, lfilter

import torch
import torch.nn as nn

from utils import device
from pytorch_models import ResNet, ResNetDecoder, BasicBlock, DecoderBlock
from trainer import Trainer
from models import VAE, Encoder

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)


def main(args):
    directory = Path(args.in_dir)
    strip_data = np.load(str(directory / "rest_ECG_strip.npy"))

    nyq_rate = 500/2  # 500 Hz data
    filter_size = 251
    ecg_size = 2048
    latent_size = 16

    fir_filter = firwin(filter_size, [3/nyq_rate, 45/nyq_rate], pass_zero="bandpass")
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    filtered_strip_data = lfilter(fir_filter, 1, strip_data, axis=1)

    strip_data_shortened = filtered_strip_data[:, filter_size:filter_size+ecg_size, :]

    X_train, X_test, _, _ = train_test_split(
        strip_data_shortened, strip_data_shortened, test_size=0.3, shuffle=False)

    encoder_resnet = ResNet(BasicBlock, [2, 2, 2, 2], do_fc=False, in_channels=12, inner_kernel=3, first_kernel=7)
    # Bottleneck: 2048, BasicBlock: 512 (512*block.expansion)
    encoder = Encoder(encoder_resnet, nn.Linear(512, latent_size), nn.Linear(512, latent_size))
    decoder = ResNetDecoder(DecoderBlock, [2, 2, 2, 2], latent_size, out_channels=12)
    vae = VAE(encoder, decoder).to(device)

    savedir = Path(args.out_dir)
    savedir.mkdir(exist_ok=True)

    trainer = Trainer(vae, "vae", 32, 1e-4, 10, patience=50, reduce_lr=False)

    trainer.train(X_train, X_test, 120, kld_lag=20, save_prefix=str(savedir) + "/")

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-dir", type=str, help="Directory containing 12 lead ECG data")
    parser.add_argument("--out-dir", type=str, help="Output directory", default="./")
    args = parser.parse_args()
    main(args)

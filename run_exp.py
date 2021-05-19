
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from utils import device, clean_ecg
from pytorch_models import ResNet, ResNetDecoder, BasicBlock, DecoderBlock
from trainer import Trainer
from models import VAE, Encoder

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)


def main(args):
    directory = Path(args.in_dir)
    strip_data = np.load(str(directory / "rest_ECG_strip.npy"))

    with open(args.config, "r") as fp:
        config = json.load(fp)

    filtered_strip_data = clean_ecg(strip_data, filter_size=251, thresh=800)
    strip_data_shortened = filtered_strip_data[:, :config["ecg_size"], :]

    X_train, X_test, _, _ = train_test_split(
        strip_data_shortened, strip_data_shortened, test_size=0.3, shuffle=False)

    encoder_resnet = ResNet(BasicBlock, [2, 2, 2, 2], do_fc=False, in_channels=12, inner_kernel=3, first_kernel=7)
    latent_size = config["latent_size"]
    # Bottleneck: 2048, BasicBlock: 512 (512*block.expansion)
    encoder = Encoder(encoder_resnet, nn.Linear(512, latent_size), nn.Linear(512, latent_size))
    decoder = ResNetDecoder(
        DecoderBlock,
        [2, 2, 2, 2],
        latent_size,
        out_channels=12,
        conv1_scale=config["conv1_scale"],
        conv2_scale=config["conv2_scale"],
        initial_size=config["initial_size"]
    )
    vae = VAE(encoder, decoder).to(device)

    savedir = Path(args.out_dir)
    savedir.mkdir(exist_ok=True)

    trainer = Trainer(vae, "vae", 32, 1e-4, config["kld_importance"], patience=50, reduce_lr=False)

    with open(f"{str(savedir)}/{trainer.savedir}/config.json", "w") as fp:
        json.dump(config, fp, indent=2)

    trainer.train(X_train, X_test, config["epochs"], kld_lag=config["kld_lag"], kld_warmup=config["kld_warmup"], save_prefix=str(savedir) + "/")

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-dir", type=str, help="Directory containing 12 lead ECG data")
    parser.add_argument("--out-dir", type=str, help="Output directory", default="./")
    parser.add_argument("--config", type=str, help="Config .json for training", default="./config.json")
    args = parser.parse_args()
    main(args)

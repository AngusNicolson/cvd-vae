
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np

import torch
import torch.nn as nn

from utils import device, Start
from pytorch_models import ResNet, ResNetDecoder, BasicBlock, DecoderBlock
from trainer import Trainer
from models import VAE, Encoder, SupervisedVAE
from dataset import ECGDataset
from torchvision.transforms import Compose

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)


def main(args):

    with open(args.config, "r") as fp:
        config = json.load(fp)

    transform = Compose([
        Start(output_size=config["ecg_size"])
    ])
    dataset = ECGDataset(args.dataset, args.prefix, transform=transform)

    encoder_resnet = ResNet(BasicBlock, [2, 2, 2, 2], do_fc=False, in_channels=12, inner_kernel=3, first_kernel=7)
    latent_size = config["latent_size"]
    # Bottleneck: 2048, BasicBlock: 512 (512*block.expansion)
    encoder = Encoder(encoder_resnet, nn.Linear(512, latent_size), nn.Linear(512, latent_size))
    decoder = ResNetDecoder(
        DecoderBlock,
        [2, 2, 2, 2],
        latent_size,
        out_channels=12,
        **config["decoder"]
    )
    measurement_names = [
        'VentricularRate',
        'PQInterval',
        'PDuration',
        'QRSDuration',
        'QTInterval',
        'QTCInterval',
        'RRInterval',
        'PPInterval',
    ]
    predictor = nn.Linear(latent_size, 8)
    vae = SupervisedVAE(encoder, decoder, predictor).to(device)

    savedir = Path(args.out_dir)
    savedir.mkdir(exist_ok=True)

    trainer = Trainer(vae, "vae", **config["trainer"])

    train_dir = f"{str(savedir)}/{trainer.savedir}"
    Path(train_dir).mkdir(exist_ok=True)
    with open(f"{train_dir}/config.json", "w") as fp:
        json.dump(config, fp, indent=2)

    trainer.train(dataset, config["epochs"], save_prefix=str(savedir) + "/", **config["train"])

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset .json")
    parser.add_argument("--prefix", type=str, help="Prefix for ECG .npy paths", default="")
    parser.add_argument("--out-dir", type=str, help="Output directory", default="./")
    parser.add_argument("--config", type=str, help="Config .json for training", default="./config.json")
    args = parser.parse_args()
    main(args)

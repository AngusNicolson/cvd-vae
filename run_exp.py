
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose

from utils import device, Start, split_dataset
from pytorch_models import ResNet, ResNetDecoder, BasicBlock, DecoderBlock
from trainer import Trainer
from models import VAE, Encoder, SupervisedVAE
from dataset import ECGDataset

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)


def main(args):

    with open(args.config, "r") as fp:
        config = json.load(fp)

    train_dataset, val_dataset = load_data(config)

    vae = create_supervised_vae(config)

    if args.load is not None:
        vae = load_pretrained(vae, args)

    savedir = Path(args.out_dir)
    savedir.mkdir(exist_ok=True)

    trainer = Trainer(vae, "vae", **config["trainer"])

    train_dir = f"{str(savedir)}/{trainer.savedir}"
    Path(train_dir).mkdir(exist_ok=True)
    with open(f"{train_dir}/config.json", "w") as fp:
        json.dump(config, fp, indent=2)

    trainer.train(train_dataset, val_dataset, config["epochs"], save_prefix=str(savedir) + "/", **config["train"])

    print("Done!")


def load_data(config):
    transform = Compose([
        Start(output_size=config["ecg_size"])
    ])
    dataset = ECGDataset(args.dataset, args.prefix, transform=transform)
    train_dataset, val_dataset = split_dataset(dataset)
    return train_dataset, val_dataset


def create_supervised_vae(config):
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
    predictor = nn.Linear(latent_size, 8)
    vae = SupervisedVAE(encoder, decoder, predictor).to(device)
    return vae


def create_vae(config):
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
    vae = VAE(encoder, decoder).to(device)
    return vae


def load_pretrained(vae, args):
    load_path = Path(args.load)
    state_dict = torch.load(load_path)

    if len([k for k in state_dict.keys() if "predictor" in k]) > 0:
        vae.load_state_dict(state_dict)
    else:
        with open(load_path.parent / "config.json", "r") as fp:
            old_config = json.load(fp)

        initial_vae = create_vae(old_config)
        vae = vae.from_vae(initial_vae)
        del initial_vae
    return vae


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset .json")
    parser.add_argument("--prefix", type=str, help="Prefix for ECG .npy paths", default="")
    parser.add_argument("--out-dir", type=str, help="Output directory", default="./")
    parser.add_argument("--config", type=str, help="Config .json for training", default="./config.json")
    parser.add_argument("--load", type=str, help="Optional path to a pre-trained model", default=None)
    args = parser.parse_args()
    main(args)

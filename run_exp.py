
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

    if config["load"]["path"] is not None:
        vae = load_pretrained(vae, **config["load"])

    savedir = Path(args.out_dir)
    savedir.mkdir(exist_ok=True)

    trainer = Trainer(vae, "vae", **config["trainer"])

    if config["load"]["path"] is not None:
        optimizer_state_dict = torch.load(config["load"]["path"])["optimizer"]
        trainer.load_optimizer(optimizer_state_dict)

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
    dataset = ECGDataset(args.dataset, args.prefix, transform=transform, positive_ratio=config["positive_ratio"])
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
    predictor = create_predictor(config)
    vae = SupervisedVAE(encoder, decoder, predictor).to(device)
    return vae


def create_predictor(config):
    if len(config["predictor"]) > 0:
        n_features = [config["latent_size"]] + config["predictor"]["n"] + [1]
        n_layers = len(n_features) - 1
        linears = [nn.Linear(n_features[i], n_features[i+1]) for i in range(n_layers)]
        activations = []
        for act_label in config["predictor"]["activations"]:
            if act_label == "sigmoid":
                act = nn.Sigmoid()
            elif act_label == "relu":
                act = nn.Tanh()
            elif act_label == "tanh":
                act = nn.Tanh()
            else:
                raise NotImplementedError(
                    f"{act_label} activation not implemented. Only tanh, relu and sigmoid available."
                )
            activations.append(act)

        layers = []
        for i in range(n_layers - 1):
            layers += [linears[i], activations[i]]
        layers.append(linears[-1])

        predictor = nn.Sequential(
            *layers
        )

        return predictor


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


def load_pretrained(vae, path, load_predictor):
    """Load a previously trained model, and optionally ignore weights/bias for predictor"""
    load_path = Path(path)
    state = torch.load(load_path)
    print(f"Loading model from epoch {state['epoch']}")
    state_dict = state["state_dict"]

    if not load_predictor:
        state_dict = {k: v for k, v in state_dict.items() if "predictor" not in k}

    mismatch = vae.load_state_dict(state_dict, strict=False)
    print("Missing keys:", mismatch)
    return vae


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset .json")
    parser.add_argument("--prefix", type=str, help="Prefix for ECG .npy paths", default="")
    parser.add_argument("--out-dir", type=str, help="Output directory", default="./")
    parser.add_argument("--config", type=str, help="Config .json for training", default="./config.json")
    args = parser.parse_args()
    main(args)

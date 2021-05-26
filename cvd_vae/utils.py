
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision.transforms import Compose

from cvd_vae.pytorch_models import ResNet, ResNetDecoder, BasicBlock, DecoderBlock
from cvd_vae.models import VAE, Encoder, SupervisedVAE
from cvd_vae.dataset import ECGDataset

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
        ecg = sample["ecg"]
        sample["ecg"] = ecg[:, :self.output_size]

        return sample


class RandomCrop(object):
    """Randomly crop ECG"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        ecg = sample["ecg"]

        size = ecg.shape[-1]
        start = np.random.randint(0, size - self.output_size)
        end = start + self.output_size
        sample["ecg"] = ecg[:, start:end]

        return sample


def split_dataset(dataset, val_split=0.3):
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def sort_batch(batch, return_ind=False):
    """Sort batch by follow up time (descending)"""
    fu_time = batch["fu_time"]
    ind = np.argsort(fu_time.cpu())
    ind = torch.flip(ind, dims=[0])
    out = {k: v[ind] for k, v in batch.items()}
    if return_ind:
        return out, ind
    else:
        return out


def load_data(config, dataset_path, prefix=""):
    transform = Compose([
        Start(output_size=config["ecg_size"])
    ])
    dataset = ECGDataset(dataset_path, prefix, transform=transform, positive_ratio=config["positive_ratio"])
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
                act = nn.ReLU()
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
            if config["predictor"]["activations"][i] == "relu":
                nn.init.kaiming_normal_(layers[-2].weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(layers[-2].weight)
        layers.append(linears[-1])
        nn.init.xavier_uniform_(layers[-1].weight)

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
    if "epoch" in state.keys():
        print(f"Loading model from epoch {state['epoch']}")
        state_dict = state["state_dict"]
    else:
        state_dict = state

    if not load_predictor:
        state_dict = {k: v for k, v in state_dict.items() if "predictor" not in k}

    mismatch = vae.load_state_dict(state_dict, strict=False)
    print("Missing keys:", mismatch)
    return vae


from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
import torch

from cvd_vae.utils import load_data, create_supervised_vae, load_pretrained
from cvd_vae.trainer import Trainer

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)


def main(args):

    with open(args.config, "r") as fp:
        config = json.load(fp)

    train_dataset, val_dataset = load_data(config, args.dataset, args.prefix)

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset .json")
    parser.add_argument("--prefix", type=str, help="Prefix for ECG .npy paths", default="")
    parser.add_argument("--out-dir", type=str, help="Output directory", default="./")
    parser.add_argument("--config", type=str, help="Config .json for training", default="./config.json")
    args = parser.parse_args()
    main(args)

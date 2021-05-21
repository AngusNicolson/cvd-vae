
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

from utils import device
from dataset import DataLoader


class Trainer:
    def __init__(self, model, model_name, batch_size, lr, c, reduce_lr=True, patience=10):
        self.model = model
        self.model_name = model_name
        self.savedir = f"{model_name}-{datetime.now().strftime('%d-%H%M%S')}"
        self.batch_size = batch_size
        self.lr = lr
        self.c = c
        self.reduce_lr = reduce_lr
        self.patience = patience

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=0.1)

        self.loss_history = []
        self.loss_history_train = []
        self.recon_loss_history_train = []
        self.kld_loss_history_train = []
        self.weighted_kld_loss_history_train = []

    def train(self, dataset, num_epoch, kld_lag=0, kld_warmup=0, verbose=1, val_split=0.3, save_prefix=""):
        savedir = f"{save_prefix}{self.savedir}"
        writer = SummaryWriter(savedir)
        iters = 0

        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        for epoch in range(num_epoch):

            t0 = time.time()
            dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            losses = []
            recon_losses = []
            kld_losses = []
            weighted_kld_losses = []

            if epoch < kld_lag:
                kld_weight = 0.0
            else:
                if kld_warmup == 0:
                    kld_weight = 1.0
                else:
                    kld_weight = min(((1 + epoch - kld_lag)/kld_warmup)**4, 1.0)

            for sample in dataloader:
                x = sample["ecg"]
                x = x.to(device)
                self.model.zero_grad()
                output, latent, mean, log_var = self.model(x)
                loss, reconstruction_loss, kld, weighted_kld = self.loss_fn(x, output, latent, mean, log_var, kld_weight)
                loss.backward()
                self.optimizer.step()

                # Logging -- track train loss
                losses.append(loss.item())
                recon_losses.append(reconstruction_loss.item())
                kld_losses.append(kld.item())
                weighted_kld_losses.append(weighted_kld.item())

                iters += x.shape[0]
                writer.add_scalar("iter_loss/all/train", losses[-1], iters)
                writer.add_scalar("iter_loss/recon/train", recon_losses[-1], iters)
                writer.add_scalar("iter_loss/kld/train", kld_losses[-1], iters)

            # --------------------------------------------------------
            #       Evaluate performance at the end of each epoch
            # --------------------------------------------------------

            test_metrics = self.evaluate_model(val_dataset, kld_weight)
            self.loss_history.append(test_metrics["loss"])
            if self.reduce_lr:
                self.scheduler.step(test_metrics["loss"])

            # Logging -- average train loss in this epoch
            self.loss_history_train.append(np.mean(losses))
            self.recon_loss_history_train.append(np.mean(recon_losses))
            self.kld_loss_history_train.append(np.mean(kld_losses))
            self.weighted_kld_loss_history_train.append(np.mean(weighted_kld_losses))

            param_grp = next(iter(self.optimizer.param_groups))
            writer.add_scalar("lr", param_grp["lr"], epoch)

            writer.add_scalar("epoch_loss/all/train", self.loss_history_train[-1], epoch)
            writer.add_scalar("epoch_loss/recon/train", self.recon_loss_history_train[-1], epoch)
            writer.add_scalar("epoch_loss/kld/train", self.kld_loss_history_train[-1], epoch)
            writer.add_scalar("epoch_loss/weighted_kld/train", self.weighted_kld_loss_history_train[-1], epoch)

            writer.add_scalar("epoch_loss/all/val", test_metrics['loss'], epoch)
            writer.add_scalar("epoch_loss/recon/val", test_metrics['recon_loss'], epoch)
            writer.add_scalar("epoch_loss/kld/val", test_metrics['kld_loss'], epoch)
            writer.add_scalar("epoch_loss/weighted_kld/val", test_metrics['weighted_kld_loss'], epoch)
            writer.add_scalar("mae/val", test_metrics['mae'], epoch)

            writer.add_histogram("activity", test_metrics["activity"], epoch)
            writer.add_scalar("activity/n", test_metrics["n_active"], epoch)
            writer.add_scalar("activity/proportion",
                              test_metrics["n_active"]/self.model.encoder.mean_model.weight.shape[0],
                              epoch)

            writer.add_histogram("means/bias", self.model.encoder.mean_model.bias, epoch)
            writer.add_histogram("means/weight", self.model.encoder.mean_model.weight, epoch)
            writer.add_histogram("logvar/bias", self.model.encoder.var_model.bias, epoch)
            writer.add_histogram("logvar/weight", self.model.encoder.var_model.weight, epoch)
            writer.add_histogram("decoder.linear/bias", self.model.decoder.linear.bias, epoch)
            writer.add_histogram("decoder.linear/weight", self.model.decoder.linear.weight, epoch)
            #writer.add_histogram("decoder.conv1/weight", self.model.decoder.conv1[1].weight, epoch)
            writer.add_histogram("decoder.conv2/weight", self.model.decoder.conv2[1].weight, epoch)

            fig = self.plot_example(val_dataset, 2)
            writer.add_figure("example/test", fig, epoch)

            fig = self.plot_example(val_dataset, 2, True)
            writer.add_figure("example/test_mean", fig, epoch)

            fig = self.plot_example(train_dataset, 2)
            writer.add_figure("example/train", fig, epoch)

            fig = self.plot_example(train_dataset, 2, True)
            writer.add_figure("example/train_mean", fig, epoch)

            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f"{savedir}/{self.model_name}_e{epoch}.pt")

            t1 = time.time()
            if verbose == 1:
                print(f"Epoch {epoch}: Train Loss {self.loss_history_train[-1]}  Test Loss {test_metrics['loss']}  {(t1 - t0):.0f} seconds")

    def forward_by_batches(self, dataset):
        """ Forward pass model on a dataset.
        Do this by batches so that we don't blow up the memory. """
        outputs = [[] for i in range(4)]
        xs = []
        dataloader = DataLoader(dataset, 64, False)
        self.model.eval()
        with torch.no_grad():
            for sample in dataloader:  # do not shuffle here!
                x = sample["ecg"]
                x = x.to(device)
                out = self.model(x)
                for i in range(4):
                    outputs[i].append(out[i])
                xs.append(x)
        self.model.train()

        outputs = [torch.cat(out_list) for out_list in outputs]
        x = torch.cat(xs)
        return [x, *outputs]

    def evaluate_model(self, dataset, kld_weight=1.0):

        x, output, latent, mean, log_var = self.forward_by_batches(dataset)

        activity = np.diag(np.cov(mean.cpu().numpy().T))
        n_active = (activity > 0.01).sum()

        # scores
        loss, recon_loss, kld_loss, weighted_kld = self.loss_fn(x, output, latent, mean, log_var, kld_weight)

        mae = F.l1_loss(x, output).item()
        mse = F.mse_loss(x, output).item()

        out_metrics = {
            'loss': loss.item(),
            "recon_loss": recon_loss.item(),
            "kld_loss": kld_loss.item(),
            "weighted_kld_loss": weighted_kld.item(),
            "output": output.cpu().numpy(),
            "mae": mae,
            "mse": mse,
            "activity": activity,
            "n_active": n_active
        }
        return out_metrics

    def loss_fn(self, x, x_hat, latent, mean, log_var, kld_weight=1.0):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
        kld = - self.c * 0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        weighted_kld = kld * kld_weight

        loss = reconstruction_loss + weighted_kld

        return loss, reconstruction_loss, kld, weighted_kld

    def plot_example(self, dataset, idx, mean=False):
        sample = dataset[idx]
        x = sample["ecg"]
        x_torch = torch.from_numpy(np.expand_dims(x, 0)).to(device)
        self.model.eval()

        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        X_examples = []
        with torch.no_grad():
            for j in range(5):
                X_examples.append(self.model(x_torch)[0].cpu().numpy())

            X_means = self.model.decoder(self.model(x_torch)[1]).cpu().numpy()
        for i, ax in enumerate(axes.flatten()):
            for j in range(5):
                ax.plot(X_examples[j][0, i], color="grey", alpha=0.3, linewidth=2)
            ax.plot(x[i, :], color="red", label="gt", alpha=0.9)
            if mean:
                ax.plot(X_means[0, i, :], color="black", label="recon")

        plt.tight_layout()
        self.model.train()
        return fig


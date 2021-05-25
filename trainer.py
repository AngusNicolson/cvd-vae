
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils import device, sort_batch


class Trainer:
    def __init__(self, model, model_name, batch_size, lr, c, supervised_importance, reduce_lr=True, patience=10):
        self.model = model
        self.model_name = model_name
        self.savedir = f"{model_name}-{datetime.now().strftime('%d-%H%M%S')}"
        self.batch_size = batch_size
        self.lr = lr
        self.c = c
        self.supervised_importance = supervised_importance
        self.reduce_lr = reduce_lr
        self.patience = patience

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=0.1)

    def train(self, train_dataset, val_dataset, num_epoch, lag=None, warmup=None, verbose=1, save_prefix=""):
        savedir = f"{save_prefix}{self.savedir}"
        writer = SummaryWriter(savedir)
        iters = 0

        if lag is None:
            lag = {"kld": 0, "supervised": 0}
        if warmup is None:
            warmup = {"kld": 0, "supervised": 0}

        for epoch in range(num_epoch):
            t0 = time.time()
            sampler = WeightedRandomSampler(np.array(train_dataset.dataset.weight)[train_dataset.indices], len(train_dataset))
            dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=3)
            losses = []
            recon_losses = []
            kld_losses = []
            weighted_kld_losses = []
            supervised_losses = []
            weighted_supervised_losses = []

            kld_weight = self.get_weight(epoch, lag["kld"], warmup["kld"])
            supervised_weight = self.get_weight(epoch, lag["supervised"], warmup["supervised"])

            for sample in dataloader:
                sample = sort_batch(sample)
                x = sample["ecg"]
                x = x.to(device)
                y = sample["incident"]
                y = y.to(device)
                self.model.zero_grad()
                output, y_output, latent, mean, log_var = self.model(x)
                loss, reconstruction_loss, kld, weighted_kld, supervised_loss, weighted_supervised_loss = self.loss_fn(x, output, y, y_output, mean, log_var, kld_weight, supervised_weight)
                loss.backward()
                self.optimizer.step()

                # Logging -- track train loss
                losses.append(loss.item())
                recon_losses.append(reconstruction_loss.item())
                kld_losses.append(kld.item())
                weighted_kld_losses.append(weighted_kld.item())
                supervised_losses.append(supervised_loss.item())
                weighted_supervised_losses.append(weighted_supervised_loss.item())

                iters += x.shape[0]
                writer.add_scalar("iter_loss/all/train", losses[-1], iters)
                writer.add_scalar("iter_loss/recon/train", recon_losses[-1], iters)
                writer.add_scalar("iter_loss/kld/train", kld_losses[-1], iters)
                writer.add_scalar("iter_loss/supervised/train", supervised_losses[-1], iters)

            # --------------------------------------------------------
            #       Evaluate performance at the end of each epoch
            # --------------------------------------------------------

            test_metrics = self.evaluate_model(val_dataset, kld_weight, supervised_weight)
            if self.reduce_lr:
                self.scheduler.step(test_metrics["loss"])

            param_grp = next(iter(self.optimizer.param_groups))
            writer.add_scalar("lr", param_grp["lr"], epoch)

            writer.add_scalar("epoch_loss/all/train", np.mean(losses), epoch)
            writer.add_scalar("epoch_loss/recon/train", np.mean(recon_losses), epoch)
            writer.add_scalar("epoch_loss/kld/train", np.mean(kld_losses), epoch)
            writer.add_scalar("epoch_loss/weighted_kld/train", np.mean(weighted_kld_losses), epoch)
            writer.add_scalar("epoch_loss/supervised/train", np.mean(supervised_losses), epoch)
            writer.add_scalar("epoch_loss/weighted_supervised/train", np.mean(weighted_supervised_losses), epoch)

            loss_names = {
                "loss": "all",
                "recon_loss": "recon",
                "kld_loss": "kld",
                "weighted_kld_loss": "weighted_kld",
                "supervised_loss": "supervised",
                "weighted_supervised_loss": "weighted_supervised"
            }
            for k, v in loss_names.items():
                writer.add_scalar(f"epoch_loss/{v}/val", test_metrics[k], epoch)
            writer.add_scalar("mae/val", test_metrics['mae'], epoch)
            writer.add_scalar("c_index/val", test_metrics['c_index'], epoch)

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
            if len(self.model.predictor) == 1:
                writer.add_histogram("predictor/bias", self.model.predictor.bias, epoch)
                writer.add_histogram("predictor/weight", self.model.predictor.weight, epoch)
            else:
                writer.add_histogram("predictor/bias", self.model.predictor[-1].bias, epoch)
                writer.add_histogram("predictor/weight", self.model.predictor[-1].weight, epoch)

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
                print(f"Epoch {epoch}: Train Loss {np.mean(losses)}  Test Loss {test_metrics['loss']}  {(t1 - t0):.0f} seconds")

    def forward_by_batches(self, dataset):
        """ Forward pass model on a dataset.
        Do this by batches so that we don't blow up the memory. """
        n_outputs = 5
        outputs = [[] for i in range(n_outputs)]
        inputs = {"ecg": [], "incident": [], "fu_time": []}
        dataloader = DataLoader(dataset, 64, False, num_workers=3)
        self.model.eval()
        with torch.no_grad():
            for sample in dataloader:  # do not shuffle here!
                for k in inputs.keys():
                    inputs[k].append(sample[k].to(device))
                out = self.model(inputs["ecg"][-1])
                for i in range(n_outputs):
                    outputs[i].append(out[i])
        self.model.train()

        outputs = [torch.cat(out_list) for out_list in outputs]
        inputs = {k: torch.cat(v) for k, v in inputs.items()}
        return inputs, outputs

    def evaluate_model(self, dataset, kld_weight=1.0, supervised_weight=1.0):
        inputs, outputs = self.forward_by_batches(dataset)
        # Sort tensors by fu_time
        inputs, ind = sort_batch(inputs, return_ind=True)
        outputs = [o[ind] for o in outputs]
        output, pred, latent, mean, log_var = outputs
        x, censor_status, fu_time = list(inputs.values())

        activity = np.diag(np.cov(mean.cpu().numpy().T))
        n_active = (activity > 0.01).sum()

        # scores
        loss, recon_loss, kld_loss, weighted_kld, supervised_loss, weighted_supervised_loss = self.loss_fn(x, output, censor_status, pred, mean, log_var, kld_weight, supervised_weight)

        mae = F.l1_loss(x, output).item()
        mse = F.mse_loss(x, output).item()

        c_index = concordance_index(fu_time.cpu().numpy(), -pred.squeeze().cpu().numpy(), censor_status.cpu().numpy())

        out_metrics = {
            'loss': loss.item(),
            "recon_loss": recon_loss.item(),
            "kld_loss": kld_loss.item(),
            "weighted_kld_loss": weighted_kld.item(),
            "supervised_loss": supervised_loss.item(),
            "weighted_supervised_loss": weighted_supervised_loss.item(),
            "output": output.cpu().numpy(),
            "mae": mae,
            "mse": mse,
            "activity": activity,
            "n_active": n_active,
            "c_index": c_index
        }
        return out_metrics

    def loss_fn(self, x, x_hat, censor_status, y_hat, mean, log_var, kld_weight=1.0, supervised_weight=1.0):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
        kld = - self.c * 0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        weighted_kld = kld * kld_weight
        supervised_loss = self._negative_log_likelihood(censor_status, y_hat)
        weighted_supervised_loss = supervised_loss * supervised_weight * self.supervised_importance

        loss = reconstruction_loss
        if kld_weight != 0:
            loss += weighted_kld
        if supervised_weight != 0:
            loss += weighted_supervised_loss

        return loss, reconstruction_loss, kld, weighted_kld, supervised_loss, weighted_supervised_loss

    @staticmethod
    def _negative_log_likelihood(censor_status, risk):
        """
        Define Cox PH partial likelihood function loss.

        Taken from "https://github.com/UK-Digital-Heart-Project/4Dsurvival/blob/master/survival4D/nn.py"
        Arguments: censor_status (censoring status) bool, risk (risk [log hazard ratio] predicted by network) for batch of input subjects
        As defined, this function requires that all subjects in input batch must be sorted in descending order of
        followup time
        """
        risk = risk.squeeze()
        risk = torch.sigmoid(risk)
        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=-1))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * censor_status
        neg_likelihood = -torch.sum(censored_likelihood)
        return neg_likelihood

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

            X_means = self.model.decoder(self.model(x_torch)[3]).cpu().numpy()
        for i, ax in enumerate(axes.flatten()):
            for j in range(5):
                ax.plot(X_examples[j][0, i], color="grey", alpha=0.3, linewidth=2)
            ax.plot(x[i, :], color="red", label="gt", alpha=0.9)
            if mean:
                ax.plot(X_means[0, i, :], color="black", label="recon")

        plt.tight_layout()
        self.model.train()
        return fig

    @staticmethod
    def get_weight(epoch, lag, warmup):
        if epoch < lag:
            weight = 0.0
        else:
            if warmup == 0:
                weight = 1.0
            else:
                weight = min(((1 + epoch - lag)/warmup)**4, 1.0)
        return weight


import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate"""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


class Encoder(nn.Module):
    """ Encoder for VAE """
    def __init__(self, shared_model, mean_model, var_model):
        super(Encoder, self).__init__()

        self.shared_model = shared_model
        self.mean_model = mean_model
        self.var_model = var_model

    def forward(self, x):
        hidden = self.shared_model(x)
        mean = self.mean_model(hidden)
        log_var = self.var_model(hidden)

        var = torch.exp(0.5 * log_var)
        z = self.reparametrisation(mean, var)

        return z, mean, log_var

    @staticmethod
    def reparametrisation(mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent, mean, log_var = self.encoder(x)
        out = self.decoder(latent)

        return out, latent, mean, log_var


class SupervisedVAE(nn.Module):
    def __init__(self, encoder, decoder, predictor):
        super(SupervisedVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor

    def forward(self, x):
        latent, mean, log_var = self.encoder(x)
        out = self.decoder(latent)
        pred = self.predictor(latent)

        return out, pred, latent, mean, log_var

    def from_vae(self, vae, predictor):
        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.predictor = predictor

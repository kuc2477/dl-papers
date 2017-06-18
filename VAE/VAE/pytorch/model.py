import torch
from torch import autograd
from torch import nn


def _pad_same(kernel_size):
    return (kernel_size - 1) // 2


class VAE(nn.Module):
    def __init__(self, dataset,
                 image_size, channel_num,
                 kernel_num, kernel_size,
                 z_size, use_cuda):
        super().__init__()
        self.dataset = dataset
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.z_size = z_size
        self.use_cuda = use_cuda

        # encoder
        self.encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mu = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q|x
        mu, logvar = self.q(encoded)
        z = self.sample_z(mu, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z and calculate the loss
        x_reconstructed = self.decoder(z_projected)
        loss = self.loss(x, x_reconstructed, mu, logvar)
        return x_reconstructed, loss

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mu(unrolled), self.q_logvar(unrolled)

    def sample_z(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = autograd.Variable(
            (torch.cuda if self.use_cuda else torch)
            .FloatTensor(std.size())
            .normal_()
        )
        return eps.mul(std).add_(mu)

    def loss(self, x, x_reconstructed, mu, logvar):
        # reconstruction loss
        bce = nn.BCELoss()
        bce.size_average = False
        bce_loss = bce(x_reconstructed, x)
        # kl divergence loss between q and z
        kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld_loss = torch.sum(kld).mul_(-0.5)
        return bce_loss + kld_loss

    # =====
    # Utils
    # =====

    @property
    def name(self):
        return '{dataset}-{channel_num}x{image_size}x{image_size}'.format(
            dataset=self.dataset,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self):
        # TODO: NOT IMPLEMENTED Y
        pass

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num, kernel_size=self.kernel_size,
                stride=2, padding=_pad_same(self.kernel_size),
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num, kernel_size=self.kernel_size,
                stride=2, padding=_pad_same(self.kernel_size),
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)

import torch
from torch import nn


def _pad_conv_same(kernel_size):
    return (kernel_size - 1) // 2


def _pad_transposed_conv_same(kernel_size):
    pass


class VAE(nn.Module):
    def __init__(self, image_size, channel_num,
                 kernel_num, kernel_size, latent_code_size, use_cuda):
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.latent_code_size = latent_code_size
        self.use_cuda = use_cuda

        # encoder
        self.encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num),
        )

        # q
        self.q_mu = self._linear(
            kernel_num * (image_size // 8) ** 2,
            latent_code_size
        )
        self.q_logvar = self._linear(
            kernel_num * (image_size // 8) ** 2,
            latent_code_size,
        )

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = self.q_mu(encoded), self.q_logvar(encoded)
        z = self._sample(mu, logvar)
        z_projected = self._project(z)
        return self.decoder(z_projected)

    def _sample(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.Variable(
            (torch.cuda if self.use_cuda else torch)
            .FloatTensor(std.size())
            .normal_()
        )
        return eps.mul(std).add_(mu)

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=self.kernel_size,
                stride=2, padding=_pad_conv_same(self.kernel_size),
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        # TODO: NOT IMPLEMENTED YET
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=self.kernel_size,
                stride=2, padding=_pad_transposed_conv_same(self.kernel_size),
            )
        )

    def _linear(self, in_size, out_size):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        )

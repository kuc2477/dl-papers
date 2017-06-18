from torch import nn


def _pad_conv_same(kernel_size):
    return (kernel_size - 1) // 2


def _pad_transposed_conv_same(kernel_size):
    pass


class VAE(nn.Module):
    def __init__(self, image_size, channel_num, kernel_num, kernel_size):
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size

        # encoder
        self.encoder = nn.Sequential(
            self._conv(self.channel_num, kernel_num // 8),
            self._conv(kernel_num // 8, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num),
        )

        # q
        self.q_mu = self._linear()
        self.q_var = self._linear()

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(self.channel_num, kernel_num),
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, kernel_num // 8),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        mu, var = self.q_mu(encoded), self.q_var(encoded)
        z = self._sample(mu, var)
        return self.decoder(z)

    def _sample(self, mu, var):
        pass

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
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=self.kernel_size,
                stride=2, padding=_pad_transposed_conv_same(self.kernel_size),
            )
        )

    def _linear(self):
        return nn.Sequential(
        )

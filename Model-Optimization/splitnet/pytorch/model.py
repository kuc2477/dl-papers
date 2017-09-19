import abc
from functools import reduce, partial
import operator
from torch import nn
import splits


# =====================================
# Building Blocks (Split Regularizable)
# =====================================

class WeightRegularized(nn.Module):
    @abc.abstractproperty
    def split_loss(self):
        raise NotImplementedError


class RegularizedLinear(WeightRegularized):
    def __init__(self, in_channels, out_channels,
                 split_groups=None, split_q=None):
        super().__init__()

        # Linear layer.
        self.linear = nn.Linear(in_channels, out_channels)

        # Split indicators.
        if split_groups:
            self.p = splits.get_split_indicator(split_groups, in_channels)
            self.q = (
                split_q or
                splits.get_split_indicator(split_groups, out_channels)
            )
        else:
            self.p = None
            self.q = None

    def forward(self, x):
        return self.linear(x)

    @property
    def split_loss(self):
        return splits.split_loss(self.linear.weight, self.p, self.q)


class ResidualBlock(WeightRegularized):
    def __init__(self, in_channels, out_channels, stride,
                 split_group_number=None, split_q=None):
        super().__init__()

        # 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        # 2
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # transformation
        self.need_transform = in_channels != out_channels
        self.conv_transform = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=stride, padding=0, bias=False
        ) if self.need_transform else None

        # weight
        self.w1 = self.conv1.weight
        self.w2 = self.conv2.weight
        self.w3 = self.conv_transform.weight

        # split indicators
        if split_group_number:
            self.p = splits.split_indicator(split_group_number, in_channels)
            self.r = splits.split_indicator(split_group_number, out_channels)
            self.q = (
                split_q or
                splits.split_indicator(split_group_number, out_channels)
            )
        else:
            self.p = None
            self.r = None
            self.q = None

    def forward(self, x):
        x_nonlinearity_applied = self.relu1(self.bn1(x))
        y = self.conv1(x_nonlinearity_applied)
        y = self.conv2(self.relu2(self.bn2(y)))
        return y.add_(self.conv_transform(x) if self.need_transform else x)

    @property
    def split_loss(self):
        weight_and_split_indicators = filter(partial(operator.is_not, None), [
            (self.w1, self.p, self.r),
            (self.w2, self.r, self.q),
            (self.w3, self.p, self.q) if self.need_transform else None
        ])
        return sum([
            splits.split_loss(w, p, q)
            if (p is not None and q is not None) else 0
            for w, p, q in weight_and_split_indicators
        ])


class ResidualBlockGroup(WeightRegularized):
    def __init__(self, block_number, in_channels, out_channels, stride,
                 split_group_number=None, split_q_last=None):
        super().__init__()

        # Residual block group's hyperparameters.
        self.block_number = block_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.split_groups = split_group_number
        self.split_q_last = split_q_last

        # Define residual blocks in a reversed order. This is to define
        # feature groups in hierarchical manner - from subgroups to
        # supergroups.
        self.residual_blocks = []
        for i in reversed(range(self.block_number)):
            is_first = (i == 0)
            is_last = (i == len(self.block_number) - 1)

            # Deep Split & Hierarchical Grouping.
            if is_last:
                q = self.split_q_last
            else:
                q = splits.merge_split_indicator(
                    self.residual_blocks[i+1].p, self.split_groups
                )

            block = ResidualBlock(
                self.in_channels if is_first else self.out_channels,
                self.out_channels,
                self.stride if is_first else 1,
                split_group_number=split_group_number,
                split_q=q,
            )
            self.residual_blocks.insert(0, block)

    def forward(self, x):
        return reduce(lambda x, f: f(x), self.residual_blocks, x)

    @property
    def split_loss(self):
        return sum([b.split_loss for b in self.residual_blocks])


# =======================================
# Wide Residual Net (Split Regularizable)
# =======================================

class WideResNet(WeightRegularized):
    def __init__(self, label, input_size, input_channels, classes,
                 total_block_number, widen_factor=1,
                 baseline_strides=None,
                 baseline_channels=None,
                 split_groups=None):
        super().__init__()

        # Model name label.
        self.label = label

        # Data specific hyperparameters.
        self.input_size = input_size
        self.input_channels = input_channels
        self.classes = classes

        # Model hyperparameters.
        self.total_block_number = total_block_number
        self.widen_factor = widen_factor
        self.split_groups = split_groups or [2, 2, 2]
        self.baseline_strides = baseline_strides or [1, 1, 2, 2]
        self.baseline_channels = baseline_channels or [16, 16, 32, 64]
        self.widened_channels = [
            w*widen_factor if i != 0 else w for i, w in
            enumerate(self.baseline_channels)
        ]
        self.group_number = len(self.widened_channels) - 1

        # Validate hyperparameters.
        if self.split_groups is not None:
            assert len(self.split_groups) == len(self.baseline_channels)
            assert len(self.split_groups) == len(self.baseline_strides)
        assert len(self.baseline_channels) == len(self.baseline_strides)
        assert (
            self.total_block_number % (2*self.group_number) == 0 and
            self.total_block_number // (2*self.group_number) >= 1
        ), 'Total number of residual blocks should be multiples of 2 x N.'

        # Residual block group configurations.
        blocks_per_group = self.total_block_number // self.group_number
        zipped_channels_and_strides = zip(
            self.widened_channels[:-1],
            self.widened_channels[1:],
            self.baseline_strides[1:]
        )

        # 4. Affine layer.
        self.fc = RegularizedLinear(
            self.widened_channels[self.group_number], self.classes,
            split_group_number=self.split_groups[-1]
        )

        # 3. Batchnorm & nonlinearity & pooling.
        self.bn = nn.BatchNorm2d(self.widened_channels[self.group_number])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(
            self.input_size //
            reduce(operator.mul, self.baseline_strides)
        )

        # 2. Residual block groups.
        self.residual_block_groups = []
        for k, (i, o, s) in reversed(enumerate(zipped_channels_and_strides)):
            # TODO: NOT IMPLEMENTED YET
            block_group = ResidualBlockGroup(
                blocks_per_group, i, o, s,
            )
            self.residual_block_groups.insert(0, block_group)

        # 1. Convolution layer.
        self.conv = nn.Conv2d(
            self.input_channels, self.widened_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        return reduce(lambda x, f: f(x), [
            self.conv,
            *self.residual_block_groups,
            self.pool,
            self.bn,
            self.relu,
            (lambda x: x.view(-1, self.widened_channels[-1])),
            self.fc
        ], initial=x)

    @property
    def split_loss(self):
        return sum([g.split_loss for g in self.residual_block_groups])

    @property
    def name(self):
        # Label for the split group configurations.
        if self.split_groups:
            split_label = 'split({})-'.format('-'.join(self.split_groups))
        else:
            split_label = ''

        # Name of the model.
        return (
            'WRN-{split_label}{depth}-{widen_factor}-'
            '{label}-{size}x{size}x{channels}'
        ).format(
            split_label=split_label,
            depth=(self.total_block_number+4),
            widen_factor=self.widen_factor,
            size=self.input_size,
            channels=self.input_channels,
            label=self.label,
        )

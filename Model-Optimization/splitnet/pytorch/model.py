import copy
import operator
from functools import reduce, partial
from torch import nn
import splits


# =====================================
# Building Blocks (Split Regularizable)
# =====================================

class WeightRegularized(nn.Module):
    def split_loss(self):
        raise NotImplementedError

    @property
    def is_cuda(self):
        if hasattr(self, '__cuda_flag_cache'):
            return self.__cuda_flag_cache
        self.__cuda_flag_cache = next(self.parameters()).is_cuda
        return self.__cuda_flag_cache


class RegularizedLinear(WeightRegularized):
    def __init__(self, in_channels, out_channels,
                 split_size=None, split_q=None):
        super().__init__()

        # Linear layer.
        self.linear = nn.Linear(in_channels, out_channels)

        # Split indicators.
        if split_size:
            self.p = splits.split_indicator(split_size, in_channels)
            self.q = (
                split_q or
                splits.split_indicator(split_size, out_channels)
            )
        else:
            self.p = None
            self.q = None

    def forward(self, x):
        return self.linear(x)

    def split_loss(self):
        return splits.split_loss(
            self.linear.weight, self.p, self.q,
            cuda=self.is_cuda
        )


class ResidualBlock(WeightRegularized):
    def __init__(self, in_channels, out_channels, stride,
                 split_size=None, split_q=None):
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
        self.w3 = self.conv_transform.weight if self.need_transform else None

        # split indicators
        if split_size:
            self.p = splits.split_indicator(split_size, in_channels)
            self.r = splits.split_indicator(split_size, out_channels)
            self.q = (
                split_q if split_q is not None else
                splits.split_indicator(split_size, out_channels)
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

    def split_loss(self):
        weight_and_split_indicators = filter(partial(operator.is_not, None), [
            (self.w1, self.p, self.r),
            (self.w2, self.r, self.q),
            (self.w3, self.p, self.q) if self.need_transform else None
        ])
        return sum([
            splits.split_loss(w, p, q, cuda=self.is_cuda)
            if (p is not None and q is not None) else 0
            for w, p, q in weight_and_split_indicators
        ])


class ResidualBlockGroup(WeightRegularized):
    def __init__(self, block_number, in_channels, out_channels, stride,
                 split_size=None, split_q_last=None):
        super().__init__()
        assert (split_size is None) == (split_q_last is None)

        # Residual block group's hyperparameters.
        self.block_number = block_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.split_size = split_size
        self.split_q_last = split_q_last
        self.splitted = split_size is not None and split_q_last is not None

        # Define residual blocks in a reversed order. This is to define
        # feature groups in hierarchical manner - from subgroups to
        # supergroups.
        residual_blocks = []
        for i in reversed(range(self.block_number)):
            is_first = (i == 0)
            is_last = (i == self.block_number - 1)

            if self.splitted:
                # Deep Split & Hierarchical Grouping.
                q = (
                    self.split_q_last if is_last else
                    splits.merge_split_indicator(
                        residual_blocks[0].p, self.split_size
                    )
                )
            else:
                q = None

            block = ResidualBlock(
                self.in_channels if is_first else self.out_channels,
                self.out_channels,
                self.stride if is_first else 1,
                split_size=split_size,
                split_q=q,
            )
            residual_blocks.insert(0, block)
        # Register the residual block modules.
        self.residual_blocks = nn.ModuleList(residual_blocks)

    def forward(self, x):
        return reduce(lambda x, f: f(x), self.residual_blocks, x)

    def split_loss(self):
        return sum([b.split_loss() for b in self.residual_blocks])


# =======================================
# Wide Residual Net (Split Regularizable)
# =======================================

class WideResNet(WeightRegularized):
    def __init__(self, label, input_size, input_channels, classes,
                 total_block_number, widen_factor=1,
                 baseline_strides=None,
                 baseline_channels=None,
                 split_sizes=None):
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
        self.split_sizes = split_sizes or [2, 2, 2]
        self.baseline_strides = baseline_strides or [1, 1, 2, 2]
        self.baseline_channels = baseline_channels or [16, 16, 32, 64]
        self.widened_channels = [
            w*widen_factor if i != 0 else w for i, w in
            enumerate(self.baseline_channels)
        ]
        self.group_number = len(self.widened_channels) - 1

        # Validate hyperparameters.
        if self.split_sizes is not None:
            assert len(self.split_sizes) <= len(self.baseline_channels) - 1
            assert len(self.split_sizes) <= len(self.baseline_strides) - 1
        assert len(self.baseline_channels) == len(self.baseline_strides)
        assert (
            self.total_block_number % (2*self.group_number) == 0 and
            self.total_block_number // (2*self.group_number) >= 1
        ), 'Total number of residual blocks should be multiples of 2 x N.'

        # Residual block group configurations.
        split_sizes_stack = copy.deepcopy(self.split_sizes)
        blocks_per_group = self.total_block_number // self.group_number
        zipped_channels_and_strides = list(zip(
            self.widened_channels[:-1],
            self.widened_channels[1:],
            self.baseline_strides[1:]
        ))

        # 4. Affine layer.
        self.fc = RegularizedLinear(
            self.widened_channels[self.group_number], self.classes,
            split_size=split_sizes_stack.pop()
        )

        # 3. Batchnorm & nonlinearity & pooling.
        self.bn = nn.BatchNorm2d(self.widened_channels[self.group_number])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(
            self.input_size //
            reduce(operator.mul, self.baseline_strides)
        )

        # 2. Residual block groups.
        residual_block_groups = []
        for k, (i, o, s) in reversed(list(
                enumerate(zipped_channels_and_strides)
        )):
            is_last = (k == len(zipped_channels_and_strides) - 1)
            try:
                # Case of splitting a residual block group.
                split_size = split_sizes_stack.pop()
                split_q_last = splits.merge_split_indicator(
                    self.fc.p if is_last else
                    residual_block_groups[0].residual_blocks[0].p,
                    split_size
                )
            except IndexError:
                # Case of not splitting a residual block group.
                split_size = None
                split_q_last = None

            # Push the residual block groups from upside down.
            residual_block_groups.insert(0, ResidualBlockGroup(
                blocks_per_group, i, o, s,
                split_size=split_size,
                split_q_last=split_q_last,
            ))
        # Register the residual block group modules.
        self.residual_block_groups = nn.ModuleList(residual_block_groups)

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
        ], x)

    def split_loss(self):
        return sum([g.split_loss() for g in self.residual_block_groups])

    @property
    def name(self):
        # Label for the split group configurations.
        if self.split_sizes:
            split_label = 'split({})-'.format('-'.join(
                str(s) for s in self.split_sizes
            ))
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
            label=self.label,
            size=self.input_size,
            channels=self.input_channels,
        )

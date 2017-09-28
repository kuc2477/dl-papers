import numpy as np
from torch.cuda import FloatTensor as CUDATensor
from visdom import Visdom


_WINDOW_CASH = {}


def _vis(env='main'):
    return Visdom(env=env)


def visualize_image(tensor, name, label=None, env='main'):
    tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor
    title = name + ('-{}'.format(label) if label is not None else '')
    _vis(env).image(tensor.numpy(), title=title)


def visualize_images(tensor, name, label=None, env='main'):
    tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor
    title = name + ('-{}'.format(label) if label is not None else '')
    _vis(env).images(tensor.numpy(), opts=dict(title=title))


def visualize_kernel(kernel, name, label=None, env='main'):
    # Do not visualize kernels that does not exists.
    if kernel is None:
        return

    assert len(kernel.size()) in (2, 4)
    kernel = kernel.cpu() if isinstance(kernel, CUDATensor) else kernel
    kernel_norm = kernel if len(kernel.size()) == 2 else (
        (kernel**2).mean(-1).mean(-1)
    )
    kernel_normalized = (
        (kernel_norm - kernel_norm.min()) /
        (kernel_norm.max() - kernel_norm.min())
    ) * 255
    title = name + ('-{}'.format(label) if label is not None else '')
    _vis(env).image(kernel_normalized.numpy(), opts=dict(title=title))


def visualize_scalar(scalar, name, iteration, env='main'):
    visualize_scalars(
        [scalar] if isinstance(scalar, float) or len(scalar) == 1 else scalar,
        [name], iteration, name, env=env
    )


def visualize_scalars(scalars, names, iteration, title, env='main'):
    assert len(scalars) == len(names)

    # Convert scalar tensors to numpy arrays.
    scalars = [s.cpu() if isinstance(s, CUDATensor) else s for s in scalars]
    scalars = [s.numpy() for s in scalars]

    options = dict(
        fillarea=True,
        legend=names,
        width=400,
        height=400,
        xlabel='Iterations',
        ylabel=title,
        title=title,
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
    )

    X = np.array([iteration] * len(scalars))
    Y = np.column_stack(scalars) if len(scalars) > 1 else scalars[0]

    if title in _WINDOW_CASH:
        _vis(env).updateTrace(X=X, Y=Y, win=_WINDOW_CASH[title])
    else:
        _WINDOW_CASH[title] = _vis(env).line(X=X, Y=Y, opts=options)

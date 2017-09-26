import numpy as np
from visdom import Visdom


_WINDOW_CASH = {}


def _vis(env='main'):
    return Visdom(env=env)


def visualize_image(tensor, name, label=None, env='main'):
    title = name + ('-{}'.format(label) if label is not None else '')
    _vis(env).image(tensor.numpy(), title=title)


def visualize_images(tensor, name, label=None, env='main'):
    title = name + ('-{}'.format(label) if label is not None else '')
    # TODO: NOT IMPLEMENTED YET


def visualize_scalars(scalars, names, iteration, title, env='main'):
    assert len(scalars) == len(names)
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

    if title in _WINDOW_CASH:
        _vis(env).updateTrace(
            X=np.column_stack([iteration] * len(scalars)),
            Y=np.column_stack(scalars),
            win=_WINDOW_CASH[title],
        )
    else:
        _WINDOW_CASH[title] = _vis(env).line(
            X=np.column_stack(scalars),
            Y=np.column_stack([iteration] * len(scalars)),
            opts=options
        )

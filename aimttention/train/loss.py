import torch
from torch import Tensor
from typing import Dict, Any
from aimttention.config import get_module
from functools import partial


class MTLoss:
    """ Multi-target loss function with fixed weights.

    Loss functions definition must contain keys:
        fn (str): The loss function (e.g. `aimttention.train.loss.energy_loss_fn`).
        weight (float): The weight of the loss function.
        kwargs (Dict): Optional, additional keyword arguments for the loss function.
    """

    def __init__(self, components: Dict[str, Any]):
        w_sum = sum(c['weight'] for c in components.values())
        self.components = dict()
        for name, c in components.items():
            kwargs = c.get('kwargs', dict())
            fn = partial(get_module(c['fn']), **kwargs)
            self.components[name] = (fn, c['weight'] / w_sum)

    def __call__(self, y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss = dict()
        for name, (fn, w) in self.components.items():
            l = fn(y_pred=y_pred, y_true=y_true)
            loss[name] = l * w
        loss['loss'] = sum(loss.values())
        return loss


def peratom_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str, key_true: str) -> Tensor:
    """ MSE loss function with per-atom normalization correction.
    Suitable when some of the values are zero both in y_pred and y_true due to padding of inputs.
    """
    x = y_true[key_true]
    y = y_pred[key_pred]

    if y_pred['_natom'].numel() == 1:
        l = torch.nn.functional.mse_loss(x, y)
    else:
        diff2 = (x - y).pow(2).view(x.shape[0], -1)
        dim = diff2.shape[-1]
        l = (diff2 * (y_pred['_natom'].unsqueeze(-1) / dim)).mean()
    return l


def energy_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'energy', key_true: str = 'energy') -> Tensor:
    """MSE loss normalized by the number of atoms."""
    x = y_true[key_true]
    y = y_pred[key_pred]
    s = y_pred['_natom'].sqrt()
    if y_pred['_natom'].numel() > 1:
        l = ((x - y).pow(2) / s).mean()
    else:
        l = torch.nn.functional.mse_loss(x, y) / s
    return l

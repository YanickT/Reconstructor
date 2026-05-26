from typing import Tuple
from components import Parallel
import torch
import torch.nn as nn
import numpy as np


def initialize(net: nn.Module, variances: Tuple[float, float]):
    """
    Initialize model with variances
    :param net: nn.Module = model to initialize
    :param variances: Tuple[float, float] = weight variance, bias variance
    :return: void
    """
    todo = [l for l in net.modules() if not isinstance(l, nn.Sequential)]
    for layer in todo:
        if isinstance(layer, Parallel):
            # the forward pass is initialized normally
            todo.append(layer.sb.modules())

            # the projection
            if not isinstance(list(layer.pb.modules())[0][0], nn.Identity):
                raise NotImplemented("Can currently only handle identity in projection")

        elif isinstance(layer, nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=(np.sqrt(variances[0] / layer.in_features)))
            if not (layer.bias is None):
                torch.nn.init.normal_(layer.bias, mean=0.0, std=(np.sqrt(variances[1])))  # (np.sqrt(variances[1]))

        elif isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0,
                                  std=np.sqrt(variances[0] / (layer.in_channels * np.prod(layer.kernel_size)))
                                  )
            torch.nn.init.normal_(layer.bias, mean=0.0, std=(np.sqrt(variances[1])))
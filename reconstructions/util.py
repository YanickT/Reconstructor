from typing import Tuple
from reconstructions.components import Parallel
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


def parse_einops_pattern(s: str):
    result = []
    current = []
    token = ""
    inside = False

    for char in s:
        if char == "(":
            inside = True
            current = []
        elif char == ")":
            if token:
                current.append(token)
                token = ""
            result.append(current)
            inside = False
        elif char == " ":
            if token:
                if inside:
                    current.append(token)
                else:
                    result.append(token)
                token = ""
        else:
            token += char

    if token:  # letzter Token
        result.append(token)

    return result


def rearange_inversion_control(left, right, kwargs, dims):
    parts = parse_einops_pattern(right)

    # filter parts which are known
    parts = [[e for e in part if not (e in kwargs)] for part in parts]
    parts = {index: part for index, part in enumerate(parts) if len(part) > 1}

    # identify necessary information
    known_parts = parse_einops_pattern(left)
    for index in parts:
        missing = parts[index][:-1]  # index = 1, missing = [h]
        while len(missing):
            search = missing.pop(0)  # h
            dim_index = [i for i, kpart in enumerate(known_parts) if search in kpart]
            dim = dims[dim_index[0]]
            for e in known_parts[dim_index[0]]:
                if e == search:
                    continue
                dim //= kwargs[e]
            kwargs[search] = dim
    return kwargs

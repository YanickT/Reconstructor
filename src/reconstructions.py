from typing import Union, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

import src.components as components
import src.util as util

from src.components import Parallel, PosEncoding

WEIGHT_LAYERS = (
    nn.Identity,
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,

    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,

    components.PosEncoding,
    components.Attention,
)

CC = {
    nn.Conv1d: nn.ConvTranspose1d,
    nn.Conv2d: nn.ConvTranspose2d,
    nn.Conv3d: nn.ConvTranspose3d
}

CA = {
    nn.AdaptiveAvgPool1d: nn.ConvTranspose1d,
    nn.AdaptiveAvgPool2d: nn.ConvTranspose2d,
    nn.AdaptiveAvgPool3d: nn.ConvTranspose3d
}

CCINV = {v: k for k, v in CC.items()}


def add2group(f_group: List[List[nn.Module]], r_group: List[List[nn.Module]], f_temp: List[nn.Module],
              r_temp: List[nn.Module]) -> Tuple[List, List]:
    """
    Add temps to their corresponding lists and empty them
    :param f_group: List[List[nn.Module]] = List of lists of modules forming the network
    :param r_group: List[List[nn.Module]] = List of lists of modules forming the reconstruction network
    :param f_temp: List[nn.Module] = List of modules forming the current layer in the network
    :param r_temp: List[nn.Module] = List of modules forming the current layer in the reconstruction network
    :return: [], []
    """
    if f_temp:
        f_group.append(nn.Sequential(*f_temp))
        r_group.append(nn.Sequential(*r_temp))
    return [], []


def get_cuts(rmodule: nn.Module, dims: List[Tuple[int]]):
    """
    Derive the number of additional neurons to be cut away in the reconstruction
    :param rmodule: nn.Module = module used for the reconstruction
    :param dims: List[Tuple[int]] = history of layer sizes/spatial dimensions
    :return: List = list of Cutting layers
    """
    # calculate output shape of reconstruction module
    if isinstance(rmodule, tuple(CC.keys())):  # conv layer
        os = np.floor(
            (dims[-1][-1] + 2 * rmodule.padding[-1] - rmodule.dilation[-1] * (rmodule.kernel_size[-1] - 1) - 1)
            / (rmodule.stride[-1]) + 1)
    elif isinstance(rmodule, tuple(CCINV.keys())):
        os = ((dims[-1][-1] - 1) * rmodule.stride[-1] - 2 * rmodule.padding[-1] +
              rmodule.dilation[-1] * (rmodule.kernel_size[-1] - 1) + rmodule.output_padding[-1] + 1)
    elif isinstance(rmodule, components.UnPoolConvTrans):
        c = rmodule.convtr
        os = ((dims[-2][-1] - 1) * c.stride[-1] - 2 * c.padding[-1] +
              c.dilation[-1] * (c.kernel_size[-1] - 1) + c.output_padding[-1] + 1)
    else:
        raise AttributeError(f"get_cuts called for unknown module {rmodule.__class__}")

    diff = os - dims[-2][-1]
    n, i = np.divmod(diff, 2)
    layers = [rmodule] + [components.Cut("both") for _ in range(int(n))]
    layers += [] if int(i) == 0 else [components.Cut("end")]
    return layers


def get_reco_module(module: nn.Module, dims: List[Tuple[int]]):
    """
    Derives the reconstruction module for a given module
    :param module: nn.Module = module to reconstruct
    :param dims: List[Tuple[int]] = History of layer sizes/spatial dimensions
    :return: nn.Module
    """
    match module.__class__:
        case nn.Linear:
            return nn.Linear(dims[-1][-1], dims[-2][-1], bias=not (module.bias is None))

        # transformer stuff
        case components.Attention:
            raise NotImplementedError("Transformers are not yet implemented")

        case components.PosEncoding:
            raise NotImplementedError("Transformers are not yet implemented")

        case components.StripActivation:
            raise NotImplementedError("Transformers are not yet implemented")

        case components.Rearrange:
            raise NotImplementedError("Transformers are not yet implemented")

        # convolutions
        case nn.Conv1d | nn.Conv2d | nn.Conv3d:
            rmodule = CC[module.__class__](dims[-1][1], dims[-2][1],
                                           kernel_size=module.kernel_size,
                                           stride=module.stride,
                                           output_padding=max(0, module.padding[0] - 1),
                                           bias=not (module.bias is None))
            return nn.Sequential(*get_cuts(rmodule, dims))

        # pooling
        case nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d:
            module.return_indices = True
            dim = int(module.__class__.__name__[-2])
            rmodule = components.UnPoolConvTrans(dim, dims[-1][1], dims[-2][1], module.kernel_size, module.stride,
                                                 module.padding)
            return nn.Sequential(*get_cuts(rmodule, dims))

        # adaptive pooling
        case nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d:
            kernel_size = int(dims[-2][-1] / dims[-1][-1])
            rmodule = CA[module.__class__](dims[-1][1], dims[-2][1],
                                           kernel_size=kernel_size,
                                           stride=kernel_size)
            return nn.Sequential(*get_cuts(rmodule, dims))

        # convolution transposed
        case nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d:
            rmodule = CCINV[module.__class__](dims[-1][1], dims[-2][1],
                                              kernel_size=module.kernel_size,
                                              stride=module.stride,
                                              padding=mp + 1 if isinstance((mp := module.padding), int) else 1,
                                              bias=not (module.bias is None))
            return nn.Sequential(*get_cuts(rmodule, dims))

        # flatten & unflatten layers
        case nn.Flatten:
            return nn.Unflatten(1, dims[-1])

        case nn.Unflatten:
            return nn.Flatten(1)

        # padding
        case nn.CircularPad1d | nn.CircularPad2d | nn.CircularPad3d:
            return nn.Identity()

        # special activation functions
        case nn.LeakyReLU:
            module.inplace = False
            return nn.LeakyReLU(module.negative_slope)

        case nn.PReLU:
            module.inplace = False
            return nn.PReLU(module.num_parameters, module.weight.mean().item())

        # everything else
        case _:
            module.inplace = False
            return module.__class__()


def get_conet_layout(model: Iterable[nn.Module], batch: torch.tensor, device: str,
                     start_activation: nn.Module = nn.ReLU, reco: bool = False):
    """
    Construct the conet for a given model
    :param model: nn.Module/List = model to reconstruct
    :param batch: torch.tensor = exemplary tensor to calculate the dimensions in the following layers [Hint: use bs=1]
    :param device: str = device to use for calculations
    :param start_activation: nn.Module = activation function used in the first layer
    :param reco: bool = indicates if function is called recursively
    :return: List[nn.Module], List[nn.Module] = List of forward layers, List of reconstruction layers
    """
    reco_groups = []
    reco_temp = []

    forward_groups = []
    forward_temp = []

    children = list(model.children())
    dimensions = [tuple(batch.shape)]
    activation_function = start_activation
    x = batch.to(device)
    start = reco
    for index, module in enumerate(children):
        if isinstance(module, nn.Sequential):
            forward_temp, reco_temp = add2group(forward_groups, reco_groups, forward_temp, reco_temp)
            f_gs_, r_gs, activation_function, x = get_conet_layout(module, x, device, activation_function, reco=True)
            # forward_temp += f_gs_
            # reco_temp += r_gs
            forward_temp, reco_temp = add2group(forward_groups, reco_groups, forward_temp + f_gs_, reco_temp + r_gs)

        elif isinstance(module, components.Parallel):
            forward_temp, reco_temp = add2group(forward_groups, reco_groups, forward_temp, reco_temp)
            r_blocks = []
            af = nn.GELU if activation_function == nn.Identity else activation_function
            for block in module.blocks:
                f_gs_, r_gs, _, _ = get_conet_layout(block, x, device, af, reco=True)
                r_blocks += [nn.Sequential(*r_gs[::-1])] if len(r_gs) > 1 else r_gs

            activation_function = None if activation_function == nn.Identity else activation_function
            reco_temp.append(Parallel(*r_blocks, activation=activation_function))
            activation_function = module.activation.__class__
            activation_function = nn.Identity if activation_function == None.__class__ else activation_function
            with torch.no_grad():
                x = module(x)
                dimensions.append(tuple(x.shape))

        else:
            # propagate through forward layer
            with torch.no_grad():
                x = module(x)
                dimensions.append(tuple(x.shape))

            # decide if temp layer is finished
            if isinstance(module, WEIGHT_LAYERS) or isinstance(module, (nn.Flatten, nn.Unflatten)):
                forward_temp, reco_temp = add2group(forward_groups, reco_groups, forward_temp, reco_temp)
                reco_temp.append(get_reco_module(module, dimensions))
            else:
                start = True
                reco_temp.append(activation_function())
                activation_function = module.__class__  # get_reco_module(module, dimensions)
                # It is important that append is before the get_reco_module as we want to add the restrictions to the
                # reconstructed activations as given by the previous activation function

        forward_temp.append(module)
        if not start:
            start = True
            reco_temp.append(activation_function())

    add2group(forward_groups, reco_groups, forward_temp, reco_temp)
    if reco:
        return forward_groups, reco_groups, activation_function, x
    return forward_groups, reco_groups

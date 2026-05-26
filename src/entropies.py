import torch
import numpy as np


# https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived
# https://de.wikipedia.org/wiki/Differentielle_Entropie
NORMALFACTOR = (0.5 * np.log(2 * np.pi * np.e))


def rel_entropy(first: torch.tensor, sec: torch.tensor, off: float = 1e-12) -> torch.tensor:
    """
    Calculate the relative entropy over two batches of activations
    :param first: torch.tensor = [batch, 1D activation]
    :param sec: torch.tensor = [batch, 1D activation]
    :param off: float = offset to apply to prevent zeros in first and sec
    :return: torch.tensor = [batch] relative entropy for each sample in batch
    """
    # ensure correct dimensionality
    first = torch.flatten(first, start_dim=1)
    sec = torch.flatten(sec, start_dim=1)

    # ensure positive definiteness of tensors
    first -= torch.min(first, dim=1).values[:, None]
    sec -= torch.min(sec, dim=1).values[:, None]

    # prevent zero activation
    first += off
    sec += off

    # normalize
    first /= torch.sum(first, dim=1)[:, None]
    sec /= torch.sum(sec, dim=1)[:, None]

    # calculate relative entropy for x > 0, y > 0
    return torch.sum(first * torch.log(first / sec), dim=1)


def diff_entropy(actis: torch.tensor) -> torch.tensor:
    """
    Calculate the differential entropy between a set of activations
    :param actis: torch.tensor = [batch, activations]
    :return: float = differential entropy
    """
    actis = torch.flatten(actis, start_dim=1)

    # normalize
    actis /= torch.sum(actis, dim=1)[:, None]

    # get std
    std = torch.std(actis, dim=0)
    std = torch.log(std)

    # filter std=0 out
    std = torch.where(torch.logical_or(torch.isnan(std), torch.isinf(std)), 0, std)

    # calculate entropy
    return torch.mean(NORMALFACTOR + std)


def gram_neumann_entropy(actis: torch.tensor, norm=lambda x: torch.mean(x, dim=1)) -> torch.tensor:
    """
    Calculate the von Neumann entropy of a Gram matrix for the activations
    """

    # transform actis (ndarray with n >= 2) into [batch, 1d] array
    actis = torch.flatten(actis, start_dim=1)

    # mean center activations
    actis -= norm(actis)[:, None]

    # normalize
    actis /= torch.linalg.vector_norm(actis, dim=1)[:, None]  # [100, 784]

    # construct gram matrix
    actis = torch.matmul(actis, actis.T) / (n := actis.shape[0])

    # get Eigenvalues
    actis = actis.double()
    actis = torch.linalg.eigvalsh(actis)
    actis = actis[actis > 0]

    # calculate von Neumann entropy using Eigenvalues
    return - torch.real(torch.sum(actis * torch.log(actis))) / np.log(n)


def gram_neumann_entropy_stable(actis: torch.tensor, norm=lambda x: torch.mean(x, dim=1)) -> torch.tensor:
    """
    Calculate the von Neumann entropy of a Gram matrix for the activations using svd. -> More stable but expensive
    """

    # transform actis (ndarray with n >= 2) into [batch, 1d] array
    actis = torch.flatten(actis, start_dim=1)

    # mean center activations
    actis -= norm(actis)[:, None]

    # normalize
    n = actis.shape[1]
    actis /= (db := torch.linalg.vector_norm(actis, dim=1))[:, None]  # [100, 784]
    actis = torch.where(torch.isnan(actis), 1/n, actis)

    n = actis.shape[0]

    # get Eigenvalues
    _, actis, _ = torch.linalg.svd(actis, full_matrices=False)
    actis = actis * actis / n
    actis = actis[actis > 0]

    # calculate von Neumann entropy using Eigenvalues
    return - torch.real(torch.sum(actis * torch.log(actis))) / np.log(n)

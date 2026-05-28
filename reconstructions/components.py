from typing import Optional, Any
import torch
import torch.nn as nn
from einops import rearrange


########################################################################################################################
# Packing layers

class Parallel(nn.Module):
    """
    Parallel processing block (c.f. nn.Sequential)
    For Residual connection use nn.Identity() for second block
    """

    def __init__(self, *blocks: nn.Sequential, activation: Optional[nn.Module] = None):
        """
        Initialize parallel processing structure
        :param blocks: List[nn.Sequential] = List of blocks to process parallel
        :param activation: Optional[nn.Module] = Layer/Activation function to be applied after sum over all blocks
        """
        super().__init__()
        self.blocks = blocks
        for i, block in enumerate(blocks):
            self.add_module(str(i), block)

        if activation is None:
            self.activation = None
        else:
            self.activation = activation()

    def forward(self, x):
        """
        Forward implementation used when in layer.
        :param x: input to that layer
        :return: Sum of all blocks with optional activation function
        """
        result = self.blocks[0](x)
        for block in self.blocks[1:]:
            result = result + block(x)  # this instead of += is necessary to prevent inplace for later usage

        if self.activation is None:
            return result
        return self.activation(result)


########################################################################################################################
# Component Layers

class View(nn.Module):
    """
    Class for printing the current spatial dimension of the Input
    Mostly for debugging purposes
    """

    def __init__(self, msg: str = "", **kwargs):
        """
        Initialize Layer
        :param msg: str = Message to print in front of layer size
        :param kwargs: Optional = additional parameters directly passed to print function, e.g., end=""
        """
        super().__init__()
        self.msg = msg
        self.kwargs = kwargs

    def forward(self, x):
        """
        Forward implementation used when in layer.
        Performs Identity mapping with a print incorporated
        :param x: input to that layer
        :return: x
        """
        print(f"{self.msg}: {x.shape if not isinstance(x, tuple) else [e.shape for e in x]}", **self.kwargs)
        return x


class Cut(nn.Module):
    """
    Cutting layer for removing additional padding artifacts
    """

    def __init__(self, cut: str = "both", d: int = 2, c: int = 1):
        """
        Initialize cutting layer
        :param cut: str = decides where to cut. Options are {"both", "end"}
        :param d: int = number of dimensions to cut
        :param c: int = size to cut
        """
        super().__init__()
        if not (cut in ["both", "end"]):
            raise AttributeError(f"Unknown cut keyword {self.cut}")
        self.cut = cut
        self.d = d
        self.c = c

    def forward(self, x):
        """
        Forward implementation
        :param x: input to that layer
        :return: Cut input
        """
        if self.d == 1:
            if self.cut == "both":
                return x[:, :, 1:-1]
            return x[:, :, :-1]

        elif self.d == 2:
            if self.cut == "both":
                return x[:, :, 1:-1, 1:-1]
            return x[:, :, :-1, :-1]

        elif self.d == 3:
            if self.cut == "both":
                return x[:, :, 1:-1, 1:-1, 1:-1]
            return x[:, :, :-1, :-1, :-1]

        else:
            raise NotImplementedError(f"Cutting for {self.d}D is not yet defined")


########################################################################################################################
# Weight Layers

class Attention(nn.Module):
    def __init__(self, lss, heads=8):
        super().__init__()
        assert (lss % heads == 0)
        self.heads = heads
        self.lss = lss
        self.scale = (lss // heads) ** -0.5
        self.to_qkv = nn.Linear(lss, 3 * lss, bias=False)
        self.to_out = nn.Linear(lss, lss)
        self.attn = None

    def forward(self, x):
        qkv = self.to_qkv(x)  # (b, n, 3*lss)
        qkv = rearrange(qkv, 'b n (t h d) -> t b h n d', t=3, h=self.heads)
        q, k, v = qkv.unbind(0)
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)

        self.attn = attn.clone().detach()
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PosEncoding(nn.Module):

    def __init__(self, np, lss):
        super().__init__()
        # technical parameters
        self.np = np  # number of patches to expect
        self.lss = lss  # latent space size

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.lss))  # tensor(1, 1, lss)
        # self.pos_embedding = nn.Parameter(torch.randn(1, np + 1, self.lss))  # tensor(1, np + 1, lss)

    def forward(self, x):  # x = tensor(bs, np, lss)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # tensor(bs, 1, lss)
        x = torch.cat((cls_tokens, x), dim=1)  # tensor(bs, np + 1, lss)
        # x = x + self.pos_embedding  # tensor(bs, np + 1, lss)
        return x


########################################################################################################################
# Reconstruction Layers

class UnPoolConvTrans(nn.Module):

    def __init__(self, dim, in_channels, out_channels, kernel_size, padding=0, stride=None):
        super().__init__()
        if stride is None:
            stride = kernel_size
        pad = padding + 1
        if dim == 1:
            self.unpool = nn.MaxUnpool1d(kernel_size, padding=pad, stride=stride)
            self.convtr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        elif dim == 2:
            self.unpool = nn.MaxUnpool2d(kernel_size, padding=pad, stride=stride)
            self.convtr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        elif dim == 3:
            self.unpool = nn.MaxUnpool3d(kernel_size, padding=pad, stride=stride)
            self.convtr = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        else:
            raise NotImplementedError("UnpoolConvTrans not implemented for dim > 3")

    def forward(self, x):
        x, indices = x
        x = self.unpool(x, indices)
        x = self.convtr(x)
        return x

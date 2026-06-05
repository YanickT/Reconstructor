from reconstructions.reconstruction import get_conet_layout, get_reco_module
from reconstructions.network_suits import ContraNetwork
from reconstructions.entropies import gram_neumann_entropy_stable
from itertools import accumulate
import skimage.measure
import numpy as np
import torch.nn as nn
import torch
import copy

"""
- entropy berechnen
- layer tauschen 
"""


class NasSuit(ContraNetwork):

    def __init__(self, model, train_data, val_data, train_steps=None, train_its=1,
                 device: torch.device = torch.device("cpu")):
        model_layers, co_model_layers = get_conet_layout(model, next(iter(train_data))[0][0].unsqueeze(0),
                                                         device=device)
        model_layers, co_model_layers = nn.Sequential(*model_layers), nn.Sequential(*co_model_layers)
        super().__init__(model_layers, co_model_layers, device)

        # store for dynamic actions
        self.train_data = train_data
        self.val_data = val_data
        self.train_steps = train_steps
        self.train_its = train_its

        # train reconstruction networks
        self._train_from_lvl(0)
        self.dims = []
        self.get_dims()

    @staticmethod
    def _reset_module(module):
        for m in module.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _train_from_lvl(self, lvl):

        # reset layers in deeper than lvl
        # [[l.reset_parameters() for l in c if hasattr(l, "reset_parameters")] for c in self.conet[lvl:]]
        for c in self.conet[lvl:]:
            self._reset_module(c)

        # setup new optimizers for the layers
        self.optimiers = self.optimiers[:lvl] + [
            torch.optim.Adam(layer.parameters()) if list(layer.parameters()) else None for layer in self.conet[lvl:]]

        self.set_train_mode()
        indices_flag = False
        c_steps = 0
        for i in range(self.train_its):
            for counter, (inp, _) in enumerate(self.train_data):
                x = inp.to(self.device)
                with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=(self.device.type == "cuda")
                ):
                    for depth, (f, c, opt) in enumerate(zip(self.net, self.conet, self.optimiers)):
                        # forward pass
                        with torch.no_grad():
                                x_ = f(x)

                        # filter for max pooling layers with 'return_indices' being True
                        if isinstance(x_, tuple):
                            x_, indices = x_
                            output_size = x.shape
                            indices_flag = True
                        else:
                            indices_flag = False

                        if depth >= lvl:
                            # detach to prevent any unnecessary gradients
                            x_ = x_.detach()
                            if indices_flag:
                                # if this is true, c is an UnPoolConvTrans which requires the indices when called
                                loss = torch.nn.functional.mse_loss(c((x_, indices, output_size)), x)
                                indices_flag = False

                            else:
                                # every other case
                                loss = torch.nn.functional.mse_loss(c(x_), x)

                            if not (opt is None):
                                opt.zero_grad(set_to_none=True)
                                self.scaler.scale(loss).backward()
                                self.scaler.step(opt)
                                self.scaler.update()

                        # override old x
                        x = x_

                if self.train_steps is not None:
                    c_steps += 1
                    if c_steps > self.train_steps:
                        return

    def get_entropy(self, f=gram_neumann_entropy_stable, return_cascasdes=False):
        cascades = self.cascade(next(iter(self.val_data))[0])
        if return_cascasdes:
            return [f(c).item() for c in cascades], cascades
        return [f(c).item() for c in cascades]

    def get_dims(self):
        x = next(iter(self.train_data))[0][0].unsqueeze(0).to(self.device)
        self.dims = []
        for module in self.net:
            with torch.no_grad():
                x = module(x)
                if isinstance(x, tuple):
                    x = x[0]
                self.dims.append(x.shape)

    def __getitem__(self, item):
        return self.net[item]

    def __setitem__(self, key, value):
        self.net[key] = value
        self.conet[key] = get_reco_module(value, dims=self.dims[:key + 1])
        self._train_from_lvl(key)

    def get_network_instance(self):
        return copy.deepcopy(self.net)



def evaluate_score(entropy, co_module, eta=0.5):
    """
    Evalutes a score of for with which priority it should be changed.
    The score takes the loss of information but also the cost of training the reconstruction networks for a change at
    this position into account.
    :param entropy:
    :return:
    """
    loss = - np.diff(entropy)

    # estimate costs by #paramerters
    costs = [sum(p.numel() for p in m.parameters() if p.requires_grad) for m in co_module]
    costs = np.array(list(accumulate(costs[::-1]))[::-1])
    costs = costs / costs[0]
    print(costs)

    # exclude last layer due to shrinking for classification
    loss = loss[:-1]
    costs = costs[:-1]

    return loss / (costs ** eta)


def guess_layer_improvement(index, entropies, cascades, net, history, tol=1e-3):
    """
    1. get entropies of this and next layer as e_i, e_i+1
    2. get cascade average reco image entropy for this and next layer ei_i, ei_i+1

    Possible cases:
    e_i+1 > e_i:
        We dont do that here (no joking. This is perfect and we wont touch that layer)

    else:
        ei_i << ei_i+1 -> chaos induced loss of information.
            -> Our metric introduces new chaos (i.e. redumdant information + new bias)
            Conv -> Increase kernelsize + Padding -> Remove layer if two -> AVGPOOL -> LPPool(norm_type=2) -> Maxpooling

        ei_i ~ ei_i+1:
            -> The variance eof the pixels stays roughly similar. We loose information probably due to too strict
            spatial downsampling
            Maxpooling -> LPPool(norm_type=2) -> AVGPOOL -> Conv
            -> Reduce kernel size + Add new layer to ensure same spatial size -> Reduce kernelsize + Add new layer

        ei_i >> ei_i+1:
            -> We marginalize over stuff we should not... Same as ei_i ~ ei_i+1
    """
    gid = entropies[index + 1] - entropies[index]  # global information difference
    if gid > 0:
        # we do not touch this layer as it stabilizes the information flow
        return False

    # FOR gid < 0:
    # average reconstructed image entropy difference
    aried = (np.mean([skimage.measure.shannon_entropy(c) for c in cascades[index + 1]]) -
             np.mean([skimage.measure.shannon_entropy(c) for c in cascades[index]]))

    if aried >= tol:
        # The entropy of the individual images increased.
        # This indicates additional local degrees of freedom and noise injection
        # stride increase -> MaxPool -> LPPool(p=2) or AvgPool
        if isinstance(net[index], (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            """
            1. bias = False
            2. reduce channels  # how much
            3. increase kernel size  # how much
            4. increase stride  # till kernel size
            5. change to Maxpool
            """
            pass

        elif isinstance(net[index], (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            """
            1. increase kernel size  # how much
            2. change to LPPool
            """
            pass

        elif isinstance(net[index], (nn.LPPool1d, nn.LPPool2d, nn.LPPool3d)):
            """
            1. change to AVGPOOL
            """
            pass

        elif isinstance(net[index], (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            print("Cannot further alter AvgPool layer")

        else:
            print(f"No change implemented for layer {type(net[index])}")

    elif aried < tol:
        # The entropy of the individual images decreased
        # This indicates a loss of information due to an marginalization over relevant features
        # reduce downsampling: MaxPooling -> LP Pool -> Conv -> stride reduction -> kernel size decrease + new layer
        pass

    else:
        # The entropy of the individual images stayed constant & information loss
        # Loss due to too strict downsampling
        pass
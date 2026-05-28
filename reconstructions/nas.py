from reconstructions.reconstruction import get_conet_layout, get_reco_module
import torch.nn as nn
import torch


"""
- entropy berechnen
- layer tauschen 
"""


class NasSuit:

    def __init__(self, model, train_data, test_data, train_steps, train_its, device):

        model_layers, co_model_layers = get_conet_layout(model, next(iter(train_data))[0][0])
        self.layers = nn.Sequential(*model_layers)
        self.co_layers = nn.Sequential(*co_model_layers)

        # store for dynamic actions
        self.train_data = train_data
        self.test_data = test_data
        self.train_steps = train_steps
        self.train_its = train_its
        self.device = device

    def _train_from_lvl(self, lvl):
        self._set_train_mode()
        indices_flag = False
        c_steps = 0
        for i in range(self.train_its):
            for counter, (inp, _) in enumerate(self.train_data):
                x = inp.to(self.device)

                for depth, (f, c, opt) in enumerate(zip(self.net, self.conet, self.optimiers)):
                    # forward pass
                    with torch.no_grad():
                        x_ = f(x)

                        # filter for max pooling layers with 'return_indices' being True
                        if hasattr(f, "return_indices") and f.return_indices:
                            x_, indices = x_[0]
                            indices_flag = True

                    if depth >= lvl:
                        # detach to prevent any unnecessary gradients
                        x_ = x_.detach()
                        if indices_flag:
                            # if this is true, c is an UnPoolConvTrans which requires the indices when called
                            if self.scaler is None:
                                loss = torch.nn.functional.mse_loss(c(x_, indices), x)
                            else:
                                with torch.amp.autocast('cuda'):
                                    loss = torch.nn.functional.mse_loss(c(x_, indices), x)
                            indices_flag = False

                        else:
                            # every other case
                            if self.scaler is None:
                                loss = torch.nn.functional.mse_loss(c(x_), x)
                            else:
                                with torch.amp.autocast('cuda'):
                                    loss = torch.nn.functional.mse_loss(c(x_), x)

                        if not (opt is None):
                            if self.scaler is None:
                                loss.backward()
                                opt.step()
                                opt.zero_grad()

                            else:
                                self.scaler.scale(loss).backward()
                                self.scaler.step(opt)
                                self.scaler.update()
                                opt.zero_grad(set_to_none=True)

                    # override old x
                    x = x_

                if self.steps is not None:
                    c_steps += 1
                    if c_steps > self.steps:
                        return

    def __len__(self):
        return len(self.net)

    def _set_train_mode(self):
        for net, cnet in zip(self.layers, self.co_layers):
            net.train()
            cnet.train()

    def _set_eval_mode(self):
        for net, cnet in zip(self.layers, self.co_layers):
            net.eval()
            cnet.eval()

    def get_entropy(self):
        pass

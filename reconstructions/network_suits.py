from typing import Union, Iterable, List, Tuple, Optional, Callable
import warnings
import time
import torch
import torch.nn as nn


class Network:
    """
    Default network-class
    Resembles a classical discriminator network with either softmax or sigmoid output (no multi-label)
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, f_loss: Optional[Callable] = None, device: str = "cpu"):
        """
        Initialize the network suit for the given model
        :param model: nn.Module = model to be trained
        :param lr: float = learning rate
        :param f: Callable = loss function
        :param device: str = device to perform calculations on
        """
        self.lr = lr
        self.device = device
        self.model = model.to(self.device)
        if torch.cuda.is_available() and device != "cpu":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            warnings.warn("No GPU available, using CPU instead -> scaler deactivated")
            self.scaler = None

        if f_loss is None:
            self.f = nn.functional.cross_entropy
        else:
            self.f = f_loss

    def training(self, train_data: Iterable,
                 test_data: Iterable,
                 its: int,
                 optimizer: torch.optim.Optimizer,
                 verbose: bool = False,
                 stop_f: Callable = None,
                 metric_fs: List[Callable] = None) -> Tuple[List[float]]:
        """
        Train network on provided training data. Subsequent testing on test data for each epoch
        :param train_data: Iterable = data to train on
        :param test_data: Iterable = data to test with
        :param its: int = number of epochs to train
        :param optimizer: torch.optim.Optimizer = optimizer to use for training
        :param verbose: bool = print information
        :param stop_f: Callable = function(metric_results) -> bool derives if training should be stopped
                                  f(...) -> True stops training
        :param metric_fs: List[Callable] = List of metrics to be calculated on the test data
                                           f(prediction, target) -> metric
        :return: Tuple[List[float]] = loss(epoch), metrics(epoch)
        """
        metrics = []
        t1 = time.time()
        for i in range(its):
            self.model.train()
            torch.cuda.empty_cache()
            for index, (inp, out) in enumerate(train_data):
                print(f"\r{index} / {len(train_data)}", end="")
                o = out.to(self.device, non_blocking=True)
                inp_ = inp.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)

                if self.scaler is None:
                    x = self.model(inp_)
                    loss = self.f(x, o)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                else:
                    with torch.amp.autocast('cuda'):
                        x = self.model(inp_)
                        loss = self.f(x, o)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            # evaluate on test data
            stats = self.eval(test_data, metric_fs)
            metrics.append(stats)
            if stop_f(metrics):
                break

            if verbose:
                print(f"\rTraining {(i + 1)} / {its}: {time.time() - t1} with")
                print(f"\tMetrics: {[round(stat) for stat in stats]}")

        if verbose:
            print(f"\nTraining done in {time.time() - t1}s")
        return tuple(zip(*metrics))

    def eval(self, test_data: Iterable, metric_fs: List[Callable] = None) -> List[float]:
        """
        Test network on provided test data
        :param metric_fs: List[Callable] = List of metrics to be calculated on the test data
                                           f(prediction, target) -> metric
        :param test_data: Iterable = data to test with
        :return: List[float] = loss, metrics
        """
        self.model.eval()

        metrics = []
        with torch.no_grad():
            for inp, out in test_data:

                o = out.to(self.device, non_blocking=True)  # out.to(self.device)
                inp_ = inp.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)

                if self.scaler is None:
                    x = self.model(inp_)
                    loss = self.f(x, o)
                else:
                    with torch.amp.autocast('cuda'):
                        x = self.model(inp_)
                        loss = self.f(x, o)

                metric_temp = [loss.cpu().item()]
                for f in metric_fs:
                    metric_temp.append(f(x, o))
                metrics.append(metric_temp)

        return [sum(metric) / len(metric) for metric in zip(*metrics)]


class ContraNetwork:
    """
    Networks for reconstruction
    This class is not a cascade but holds all reconstruction networks and combines them to the different cascades when
    necessary
    """

    def __init__(self, net: nn.Sequential, conet: nn.Sequential, device: Union[str, torch.device] = "cpu"):
        """
        Initalize the network based on the information given at net.
        :param net: Sequential = network to create reconstrution networks for
        :param device: Union[str, torch.device] = device to perform calculations on
        """
        self.net = net.to(device)

        self.length = len(conet)
        self.device = device

        self.conet = conet.to(device)
        self.optimiers = [torch.optim.Adam(layer.parameters()) if list(layer.parameters()) else None for
                          layer in conet]

        if torch.cuda.is_available() and device != "cpu":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            warnings.warn("No GPU available, using CPU instead -> scaler deactivated")
            self.scaler = None

    def __len__(self) -> int:
        """
        Get length of network == number of reconstruction networks
        :return: int = number of reconstruction networks
        """
        return self.length

    def set_train_mode(self):
        """
        Set all reconstruction networks into training mode
        :return: void
        """
        for layer in self.conet:
            layer.train()

    def set_eval_mode(self):
        """
        Set all reconstruction networks into eval mode
        :return: void
        """
        for layer in self.conet:
            layer.eval()

    def save(self, path: str):
        """
        Save reconstruction networks at given location
        :param path: str = path + name for file to store data in
        :return: void
        """
        model_dicts = [layer.state_dict() for layer in self.conet]
        optimizer_dicts = [None if opt is None else opt.state_dict() for opt in self.optimiers]
        torch.save({
            "model_state_dict": model_dicts,
            "optimizer_state_dict": optimizer_dicts}, path)

    def load(self, path: str):
        """
        Load reconstruction networks from file at given path
        :param path: str = path to file to load
        :return: void
        """
        data = torch.load(path)
        model_dicts = data["model_state_dict"]
        optimizer_dicts = data["optimizer_state_dict"]
        for seq, model_dict in zip(self.conet, model_dicts):
            seq.load_state_dict(model_dict)

        for opt, opt_dict in zip(self.optimiers, optimizer_dicts):
            if opt is None:
                continue
            opt.load_state_dict(opt_dict)

    def train(self, train_data: Iterable, its: int = 1, verbose: bool = True, steps=None):
        """
        Train all reconstruction networks
        :param train_data: Iterable = Training data
        :param its: int = number of epochs to train
        :return: void
        """
        t1 = time.time()

        self.set_train_mode()
        indices_flag = False
        c_steps = 0
        for i in range(its):

            if verbose:
                print(f"\nEpoch: {i} / {its}")

            for counter, (inp, _) in enumerate(train_data):
                # print(f"\r{counter} / {len(train_data)}", end="")
                x = inp.to(self.device)

                for f, c, opt in zip(self.net, self.conet, self.optimiers):
                    # forward pass
                    with torch.no_grad():
                        x_ = f(x)

                        # filter for max pooling layers with 'return_indices' being True
                        if hasattr(f, "return_indices") and f.return_indices:
                            x_, indices = x_[0]
                            indices_flag = True

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

                if steps is not None:
                    c_steps += 1
                    if c_steps > steps:
                        print(f"\nTraining done in {time.time() - t1}s")
                        return

        if verbose:
            print(f"\nTraining done in {time.time() - t1}s")

    def cascade(self, inp: torch.tensor) -> List[torch.tensor]:
        """
        Calculate the reconstructions for each layer of network for the provided inp data. This is done by
        combining the reconstruction networks into cascades for each depth in the forward network
        :param inp: torch.tensor = input to propagate through net and determine the reconstructions for
        :return: List[torch.tensor] = reconstructions for every layer
        """
        t1 = time.time()

        self.set_eval_mode()

        x_forward = inp.to(self.device)
        images = [x_forward.detach().cpu()]
        with torch.no_grad():
            indices = {}
            for i, f in enumerate(self.net):
                if i >= len(self.conet):
                    break
                # forward pass through the network
                if isinstance(x_forward, tuple):
                    indices[i] = x_forward[1]
                    x_forward = x_forward[0]

                x_forward = f(x_forward)

                # reconstruction
                x_back = x_forward  # .detach()
                for j in range(i, -1, -1):

                    if hasattr(self.net[j], "return_indices") and f.return_indices:
                        x_back = self.conet[j](x_back, indices[j])
                    else:
                        x_back = self.conet[j](x_back)

                images.append(x_back.detach().cpu())

        print(f"Cascade done in {time.time() - t1}s")
        return images

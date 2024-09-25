# coding=utf-8
'''
Created on Apr 19 5, 2021

@author: ADAPA
'''
import os

import numpy as np
import scipy
import torch
import torch.nn as nn
from PyQt5.QtCore import QObject, pyqtSignal


class SimpleNetwork(nn.Module, QObject):
    sigLoss = pyqtSignal(np.ndarray)
    sigR2 = pyqtSignal(np.ndarray)
    sigPredictions = pyqtSignal(np.ndarray)
    sigParams = pyqtSignal(list)
    progress = pyqtSignal(float)

    def __init__(self, layer_names, layer_shapes, activations, layer_options=[], init_weights_method='',
                 scale_inputs='pv',
                 iterations=100, loss_fun='MSELoss', optimizer='SGD', learning_rate=1e-3):
        QObject.__init__(self, parent=None)
        nn.Module.__init__(self)
        self.activations = activations
        self.init_weights_method = init_weights_method
        self.scale_inputs = scale_inputs
        self.iterations = iterations
        self.loss_fun = getattr(nn, loss_fun)() if hasattr(nn, loss_fun) else nn.MSELoss()
        if type(self.loss_fun) == nn.MSELoss:
            self.loss_fun = nn.MSELoss(reduction='sum')
        self.optimizer = getattr(torch.optim, optimizer) if hasattr(torch.optim, optimizer) else torch.optim.SGD
        self.learning_rate = learning_rate
        self.layers = []
        self.scale_y = 1

        # Linear ...
        for i, a in enumerate(layer_names):
            if type(a) is str and hasattr(nn, a) and any(layer_shapes[i]):
                options = {}
                self.layers.append(getattr(nn, a)(*layer_shapes[i], **options))
                if isinstance(activations[i], (list, tuple, np.ndarray)):
                    for b in activations[i]:
                        if hasattr(nn, b):
                            args = []
                            options = {}
                            if b == 'Dropout':
                                args = [0.2]
                            if b == 'BatchNorm1d':
                                args = [layer_shapes[i][1]]
                            self.layers.append(getattr(nn, b)(*args, **options))
                elif hasattr(nn, activations[i]):
                    self.layers.append(getattr(nn, activations[i])())

        # Add output layer (linear)
        self.layers.append(nn.Linear(*layer_shapes[-1]))
        self.linear_layers = nn.Sequential(*self.layers)

        # uniform_, normal_, xavier_normal_, kaiming_normal_...
        if init_weights_method != '' and hasattr(nn.init, init_weights_method):
            self.apply(self.init_net)

    # Defining the forward pass
    def forward(self, x):
        x = self.linear_layers(x)
        return x

    def init_net(self, m):
        if type(m) == nn.Linear:
            if self.init_weights_method == 'normal_':
                iw = nn.init.normal_(m.weight, mean=0, std=1)
            else:
                iw = getattr(nn.init, self.init_weights_method)(m.weight)

    def regularize_output(self, Ya, rtype='pv'):
        self.scale_y = 0.1  # (np.nanmax(Ya) - np.nanmin(Ya))
        Ya = np.nan_to_num(Ya, nan=0.0)
        return Ya / self.scale_y

    @staticmethod
    def regularize_input(Xa, rtype='pv'):
        if rtype == 'pv':
            scale = (np.nanmax(Xa, axis=0) - np.nanmin(Xa, axis=0))
            Xa = (Xa - np.nanmean(Xa, axis=0)) / scale
        elif rtype == 'zscore':
            scale = np.nanstd(Xa, axis=0)
            Xa = (Xa - np.nanmean(Xa, axis=0)) / scale
        elif rtype == 'tanh':
            # https://www.cs.ccu.edu.tw/~wylin/BA/Fusion_of_Biometrics_II.ppt
            Xa = 0.5 + 0.5 * np.tanh(0.01 * (Xa - np.nanmean(Xa, axis=0)) / np.nanstd(Xa, axis=0))
        Xa = np.nan_to_num(Xa, nan=0.0)
        return Xa

    def run_net(self, Xa, Ya, callback=None, save_and_load=False):
        path = 'C:/Users/adapa/Desktop/eclipse_workspace/simple_net.pt'
        if save_and_load and os.path.exists(path):
            try:
                self.load_state_dict(torch.load(path))
                self.eval()
            except Exception as e:
                print(e)
        # Normalize with z scores
        Xa = self.regularize_input(Xa, rtype=self.scale_inputs)
        Ya = self.regularize_output(Ya)
        X = torch.from_numpy(Xa.astype(np.float32))
        if Ya.ndim == 1:
            Ya = Ya.reshape(-1, 1)
        Y = torch.from_numpy(Ya.astype(np.float32))
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        loss_out = np.full([self.iterations], np.nan, dtype=float)
        r2_out = np.full([self.iterations], np.nan, dtype=float)
        for t in range(self.iterations):
            Ym = self.forward(X)
            loss = self.loss_fun(Ym, Y)
            loss_out[t] = loss.item()
            r, _ = self.pearsonr(Ym, Y)
            r2_out[t] = r ** 2
            self.sigLoss.emit(loss_out)
            self.sigR2.emit(r2_out)
            self.sigPredictions.emit(Ym.detach().numpy())
            # self.sigParams.emit([x.detach().numpy() for x in self.model.parameters()])

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.progress.emit(100 * t / self.iterations)

        if save_and_load:
            torch.save(self.state_dict(), path)
        return self.evaluate(X, Y)

    def evaluate(self, Xa, Ya):
        # Normalize with z scores
        Xa = self.regularize_input(Xa)
        X = torch.from_numpy(Xa.astype(np.float32)) if isinstance(Xa, np.ndarray) else Xa
        if Ya.ndim == 1:
            Ya = Ya.reshape(-1, 1)
        Y = torch.from_numpy(Ya.astype(np.float32)) if isinstance(Ya, np.ndarray) else Ya
        Ym = self.forward(X)
        loss = self.loss_fun(Ym, Y)
        r, _ = self.pearsonr(Ym, Y)
        return self.scale_y * Ym.detach().numpy(), loss.detach().numpy(), r ** 2

    @staticmethod
    def pearsonr(Y1, Y2):
        try:
            Y1a = Y1.detach().numpy()
            Y2a = Y2.detach().numpy()
            return scipy.stats.pearsonr(Y1a.ravel(), Y2a.ravel())
        except Exception as e:
            print(e)
            return 0.0, 0.0

    @staticmethod
    def get_tensorboard():
        from torch.utils.tensorboard import SummaryWriter
        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter('runs/simple_net')
        return writer

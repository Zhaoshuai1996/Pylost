# coding=utf-8
from pylost_widgets.learning.net.SimpleNetwork import SimpleNetwork
import numpy as np
import torch.nn as nn


class RNNNetwork(SimpleNetwork):
    def __init__(self, layer_names, layer_shapes, activations, input_size=1, output_size=1, hidden_size=30, bidir=False,
                 *args, **kwargs):
        super().__init__(layer_names, layer_shapes, activations, *args, **kwargs)
        ndir = 2 if bidir else 1
        self.input_size = input_size
        self.output_size = output_size
        layers1 = []
        layers2 = []
        layers1 += [nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                           bidirectional=bidir)]  # , dropout=0.1)]
        layers2 += [nn.Linear(hidden_size * ndir, output_size),
                    ]

        self.rnn_layers = nn.Sequential(*layers1)
        self.linear_layers = nn.Sequential(*layers2)

    def forward(self, x):
        pred, hidden = self.rnn_layers(x)
        pred = self.linear_layers(pred)  # .view(pred.data.shape[0], -1, 1)
        return pred

    def run_net(self, Xa, Ya, callback=None, save_and_load=False):
        if Xa.ndim == 2:
            Xa = Xa[:, :, np.newaxis]
            Ya = Ya[:, :, np.newaxis]  # .view((Ya.shape[0], -1, 1))
        return super().run_net(Xa, Ya, callback, save_and_load)

    def evaluate(self, Xa, Ya):
        if Xa.ndim == 2:
            Xa = Xa[:, :, np.newaxis]
            Ya = Ya[:, :, np.newaxis]
        return super().evaluate(Xa, Ya)

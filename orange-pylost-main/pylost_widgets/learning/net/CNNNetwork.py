# coding=utf-8
from pylost_widgets.learning.net.SimpleNetwork import SimpleNetwork
import numpy as np
import torch.nn as nn


class CNNNetwork(SimpleNetwork):
    def __init__(self, layer_names, layer_shapes, activations, scale_cnn_to_linear=1, *args, **kwargs):
        super().__init__(layer_names, layer_shapes, activations, *args, **kwargs)

        conv_names = layer_names[0]
        linear_names = layer_names[1]
        layers1 = []
        layers2 = []
        if not any(conv_names) and not any(linear_names):
            layers1 += [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                        # nn.BatchNorm2d(64),
                        # nn.Dropout(),
                        nn.ReLU(inplace=True),
                        # nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                        # nn.Dropout(),
                        # nn.BatchNorm2d(64),
                        # nn.ReLU(inplace=True),
                        # nn.MaxPool2d(kernel_size=2, stride=2)
                        ]
            layers2 += [nn.Linear(64 * 2 * 2, 100),
                        nn.ReLU(inplace=True),
                        nn.Linear(100, 100),
                        nn.ReLU(inplace=True),
                        nn.Linear(100, 2)]
        else:
            # Conv
            i = 0
            for a in conv_names:
                if type(a) is str and hasattr(nn, a) and any(layer_shapes[i]):
                    options = {}
                    if a == 'Conv2d':
                        options = {'kernel_size': 3, 'stride': 1, 'padding': 1}
                    layers1.append(getattr(nn, a)(*layer_shapes[i], **options))
                    if isinstance(activations[i], (list, tuple, np.ndarray)):
                        for b in activations[i]:
                            if hasattr(nn, b):
                                args = []
                                options = {}
                                if b == 'BatchNorm2d':
                                    args = [layer_shapes[i][1]]
                                elif b == 'MaxPool2d':
                                    options = {'kernel_size': 2, 'stride': 2}
                                layers1.append(getattr(nn, b)(*args, **options))
                    elif hasattr(nn, activations[i]):
                        layers1.append(getattr(nn, activations[i])())
                i += 1
            # resize conv to linear shape
            if scale_cnn_to_linear > 1:
                layer_shapes[i][0] = layer_shapes[i][0] * scale_cnn_to_linear
            for a in linear_names:
                if hasattr(nn, a) and any(layer_shapes[i]):
                    layers2.append(getattr(nn, a)(*layer_shapes[i]))
                    if hasattr(nn, activations[i]):
                        layers2.append(getattr(nn, activations[i])())
                i += 1

            # Add output layer (linear)
            options = {'kernel_size': 3, 'stride': 1, 'padding': 1}
            layers1.append(nn.Conv2d(*layer_shapes[-1], **options))
            # layers2.append(nn.Linear(*layer_shapes[-1]))

        self.cnn_layers = nn.Sequential(*layers1)
        self.linear_layers = nn.Sequential(*layers2)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        return x

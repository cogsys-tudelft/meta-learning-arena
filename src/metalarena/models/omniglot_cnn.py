from typing import Tuple, Optional

import torch
import torch.nn as nn


def conv_block(in_channels: int,
               out_channels: int,
               batch_norm: bool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not batch_norm),
        nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU())


class OmniglotCNN(nn.Module):

    def __init__(self, input_shape: Tuple[int, ...], num_classes: Optional[int] = None, intermediate_channels: int = 64, batch_norm: bool = True):
        """Omniglot model as described in the original MAML paper. Code based on Tensorflow implementation
        from Reptile repository https://github.com/openai/supervised-reptile/blob/master/supervised_reptile/models.py,
        PyTorch implementation from https://github.com/gabrielhuang/reptile-pytorch/blob/master/models.py
        and original MAML code https://github.com/cbfinn/maml/blob/master/utils.py.

        Args:
            input_shape (Tuple[int]): Shape of the input tensor, should be (batch_size, input_channels, height, width).
            num_classes (int): How many classes to classify with this model.
            intermediate_channels (int): Number of channels in the intermediate layers. Defaults to 64.
            batch_norm (bool): Whether to use batch normalization or not.
        """

        super(OmniglotCNN, self).__init__()

        in_channels = input_shape[1]

        self.embedder = nn.Sequential(conv_block(in_channels, intermediate_channels, batch_norm),
                                    conv_block(intermediate_channels, intermediate_channels, batch_norm),
                                    conv_block(intermediate_channels, intermediate_channels, batch_norm),
                                    conv_block(intermediate_channels, intermediate_channels, batch_norm),
                                    nn.Flatten())

        n_outputs = self.embedder(torch.empty(*input_shape)).size(-1)

        print(f"Model output shape: {n_outputs}")

        if num_classes is not None:
            self.classifier = nn.Linear(n_outputs, num_classes)
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor):
        out = self.embedder(x)
        out = self.classifier(out)

        return out

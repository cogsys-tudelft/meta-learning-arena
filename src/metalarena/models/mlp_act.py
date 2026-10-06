import torch
from typing import Callable, Optional

from torchvision.ops import MLP


class MLPAct(MLP):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # MLP with a final activation layer
        
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=inplace,
            bias=bias,
            dropout=dropout,
        )

        self.append(activation_layer() if inplace is None else activation_layer(inplace=inplace))

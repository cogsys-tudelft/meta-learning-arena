import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int = 1, d_model: int = 56,
                 nhead: int = 8, num_layers: int = 3,
                 num_classes: int = 48, dim_feedforward: int = 224):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, num_classes)
        """
        # convert x shape to x[0], -1, 1
        # x = x.view(x.size(0), -1, 1)
        # tranpose last two dimensions
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling
        return self.cls_head(x)

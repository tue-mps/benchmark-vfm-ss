import math
from models.encoder import Encoder
import torch.nn as nn
import torch


class LinearDecoder(Encoder):
    def __init__(
        self,
        model_name,
        pretrained,
        num_classes,
        img_size,
        patch_size=32,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
            patch_size=patch_size,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.head(x)
        x = x.transpose(1, 2)

        return x.reshape(x.shape[0], -1, *self.grid_size)

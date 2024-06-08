from models.encoder import Encoder
import torch.nn as nn
import torch


class LinearDecoder(Encoder):
    def __init__(
        self,
        encoder_name,
        num_classes,
        img_size,
        sub_norm=False,
        patch_size=16,
        pretrained=True,
        ckpt_path="",
    ):
        super().__init__(
            encoder_name=encoder_name,
            img_size=img_size,
            sub_norm=sub_norm,
            patch_size=patch_size,
            pretrained=pretrained,
            ckpt_path=ckpt_path,
        )

        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.head(x)
        x = x.transpose(1, 2)

        return x.reshape(x.shape[0], -1, *self.grid_size)

from models.encoder import Encoder
import torch.nn as nn
import torch
from timm.layers import LayerNorm2d


class UpscaleDecoder(Encoder):
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

        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(
                self.encoder.embed_dim,
                self.encoder.embed_dim // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            LayerNorm2d(self.encoder.embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.encoder.embed_dim // 2,
                self.encoder.embed_dim // 4,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            LayerNorm2d(self.encoder.embed_dim // 4),
            nn.Conv2d(
                self.encoder.embed_dim // 4,
                self.encoder.embed_dim // 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.encoder.embed_dim // 4),
        )

        out = nn.Conv2d(
            self.encoder.embed_dim // 4,
            num_classes,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        torch.nn.init.normal_(out.weight, 0, std=0.1)
        self.out = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], -1, *self.grid_size)

        return self.out(self.upscale(x))

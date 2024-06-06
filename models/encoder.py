import torch
import torch.nn as nn
from timm.layers import (
    resample_patch_embed,
    resample_abs_pos_embed,
)
import timm
from open_clip import create_model_from_pretrained
from open_clip.factory import get_model_config


class Encoder(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained,
        img_size: tuple[int, int],
        patch_size,
    ):
        super().__init__()

        self.encoder = create_model_from_pretrained(model_name, pretrained)[0].visual

        pixel_mean = torch.tensor(self.encoder.preprocess_cfg["mean"]).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(self.encoder.preprocess_cfg["std"]).reshape(
            1, -1, 1, 1
        )

        self.register_buffer("pixel_mean", pixel_mean, persistent=False)
        self.register_buffer("pixel_std", pixel_std, persistent=False)

        if hasattr(self.encoder, "trunk"):
            self.encoder = self.encoder.trunk

        if hasattr(self.encoder, "norm"):
            self.encoder.norm = nn.Identity()

        if hasattr(self.encoder, "norm_pre"):
            self.encoder.norm_pre = nn.Identity()

        if hasattr(self.encoder, "embed_dim"):
            self.embed_dim = self.encoder.embed_dim
        elif hasattr(self.encoder, "num_features"):
            self.embed_dim = self.encoder.num_features
        elif hasattr(self.encoder, "transformer"):
            self.embed_dim = self.encoder.transformer.width

        self.grid_size = tuple(round(size / patch_size) for size in img_size)

        if hasattr(self.encoder, "patch_embed"):
            if (
                self.encoder.patch_embed.grid_size[0]
                != self.encoder.patch_embed.grid_size[1]
                or self.encoder.patch_embed.patch_size[0]
                != self.encoder.patch_embed.patch_size[1]
            ):
                raise ValueError("pretrained grid and patch size must be square")
            self.encoder.patch_embed.patch_size = (patch_size, patch_size)
            self.encoder.patch_embed.proj.kernel_size = (patch_size, patch_size)
            self.encoder.patch_embed.proj.stride = (patch_size, patch_size)
            self.encoder.patch_embed.proj.weight = nn.Parameter(
                resample_patch_embed(
                    self.encoder.patch_embed.proj.weight,
                    [patch_size, patch_size],
                    verbose=True,
                )
            )
            self.encoder.patch_embed.grid_size = self.grid_size
            self.encoder.patch_embed.num_patches = self.grid_size[0] * self.grid_size[1]
            self.encoder.patch_embed.img_size = img_size
        elif hasattr(self.encoder, "conv1"):
            if (
                self.encoder.grid_size[0] != self.encoder.grid_size[1]
                or self.encoder.patch_size[0] != self.encoder.patch_size[1]
            ):
                raise ValueError("pretrained grid and patch size must be square")
            self.encoder.patch_size = (patch_size, patch_size)
            self.encoder.conv1.kernel_size = (patch_size, patch_size)
            self.encoder.conv1.stride = (patch_size, patch_size)
            self.encoder.conv1.weight = nn.Parameter(
                resample_patch_embed(
                    self.encoder.conv1.weight,
                    [patch_size, patch_size],
                    verbose=True,
                )
            )
            self.encoder.grid_size = self.grid_size
            self.image_size = img_size

        if hasattr(self.encoder, "pos_embed"):
            num_prefix_tokens = (
                0
                if getattr(self.encoder, "no_embed_class", False)
                else self.encoder.num_prefix_tokens
            )
            pos_embed = resample_abs_pos_embed(
                self.encoder.pos_embed,
                [
                    max(self.grid_size),
                    max(self.grid_size),
                ],
                num_prefix_tokens=num_prefix_tokens,
                verbose=True,
            )
            prefix_pos_embed = pos_embed[:, :num_prefix_tokens, :]
            pos_embed = pos_embed[:, num_prefix_tokens:, :]
            pos_embed = pos_embed.reshape(
                1, max(self.grid_size), max(self.grid_size), -1
            )[:, : self.grid_size[0], : self.grid_size[1], :]
            pos_embed = torch.cat([prefix_pos_embed, pos_embed.flatten(1, 2)], dim=1)
            self.encoder.pos_embed = nn.Parameter(pos_embed)
        elif hasattr(self.encoder, "positional_embedding"):
            num_prefix_tokens = 1
            pos_embed = resample_abs_pos_embed(
                self.encoder.positional_embedding[None, ...],
                [
                    max(self.grid_size),
                    max(self.grid_size),
                ],
                num_prefix_tokens=num_prefix_tokens,
                verbose=True,
            )
            prefix_pos_embed = pos_embed[:, :num_prefix_tokens, :]
            pos_embed = pos_embed[:, num_prefix_tokens:, :]
            pos_embed = pos_embed.reshape(
                1, max(self.grid_size), max(self.grid_size), -1
            )[:, : self.grid_size[0], : self.grid_size[1], :]
            pos_embed = torch.cat([prefix_pos_embed, pos_embed.flatten(1, 2)], dim=1)
            self.encoder.positional_embedding = nn.Parameter(pos_embed[0])

        if hasattr(self.encoder, "rope"):
            self.encoder.rope = timm.create_model(
                model_name=get_model_config(model_name)["vision_cfg"][
                    "timm_model_name"
                ],
                pretrained=True,
                img_size=img_size,
                patch_size=patch_size,
            ).rope

    def forward(self, x: torch.Tensor):
        x = (x - self.pixel_mean) / self.pixel_std

        if hasattr(self.encoder, "forward_features"):
            x = self.encoder.forward_features(x)
        else:
            x = self.encoder.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat(
                [
                    self.encoder.class_embedding.view(1, 1, -1)
                    .expand(x.shape[0], -1, -1)
                    .to(x.dtype),
                    x,
                ],
                dim=1,
            )
            x = x + self.encoder.positional_embedding.to(x.dtype)
            x = self.encoder.patch_dropout(x)
            x = self.encoder.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.encoder.transformer(x)
            x = x.permute(1, 0, 2)

        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x[:, -self.grid_size[0] * self.grid_size[1] :]

        return x

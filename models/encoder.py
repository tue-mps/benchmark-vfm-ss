from typing import Optional
import timm
import torch
import torch.nn as nn
from timm.layers import (
    resample_patch_embed,
    resample_abs_pos_embed,
    resample_abs_pos_embed_nhwc,
)
from timm.models._manipulate import checkpoint_seq
from torch.nn.functional import interpolate


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_name,
        img_size: tuple[int, int],
        ckpt_path,
        sub_norm,
        patch_size,
        pretrained,
    ):
        super().__init__()

        model_kwargs = {
            "model_name": encoder_name,
            "pretrained": pretrained,
            "num_classes": 0,
        }
        self.encoder = timm.create_model(**model_kwargs)

        pixel_mean = torch.tensor(self.encoder.default_cfg["mean"]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(self.encoder.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self.grid_size = tuple(round(size / patch_size) for size in img_size)

        self.embed_dim = (
            self.encoder.embed_dim
            if hasattr(self.encoder, "embed_dim")
            else self.encoder.num_features
        )

        if sub_norm:
            for block in self.encoder.blocks:
                new_mlp = type(block.mlp)(
                    in_features=block.mlp.fc1.in_features,
                    hidden_features=block.mlp.fc1.out_features,
                    act_layer=type(block.mlp.act),
                    drop=block.mlp.drop1.p,
                    norm_layer=nn.LayerNorm,
                )
                new_mlp.load_state_dict(block.mlp.state_dict(), strict=False)
                block.mlp = new_mlp
                block.attn.proj = nn.Sequential(
                    nn.LayerNorm(block.attn.proj.in_features), block.attn.proj
                )

        if hasattr(self.encoder, "neck"):
            self.encoder.neck = nn.Identity()

        if ckpt_path:
            self.encoder.load_state_dict(torch.load(ckpt_path))

        if hasattr(self.encoder, "rope"):
            self.encoder.rope = timm.create_model(
                img_size=img_size, patch_size=patch_size, **model_kwargs
            ).rope

        if hasattr(self.encoder, "blocks"):
            for block in self.encoder.blocks:
                old_window_size = None
                if hasattr(block, "window_size"):
                    old_window_size = block.window_size
                    window_ratio = (
                        old_window_size / self.encoder.patch_embed.grid_size[0]
                    )
                    new_window_size = window_ratio * (img_size[0] / patch_size)

                    if new_window_size != round(new_window_size):
                        raise ValueError("invalid window size")

                    block.window_size = int(new_window_size)

                if hasattr(block.attn, "rel_pos_h"):
                    block.attn.rel_pos_h = self.interpolate_rel_pos(
                        block.attn.rel_pos_h,
                        img_size[0] / patch_size,
                        self.encoder.patch_embed.grid_size[0],
                        block.window_size,
                        old_window_size,
                    )

                if hasattr(block.attn, "rel_pos_w"):
                    block.attn.rel_pos_w = self.interpolate_rel_pos(
                        block.attn.rel_pos_w,
                        img_size[1] / patch_size,
                        self.encoder.patch_embed.grid_size[1],
                        block.window_size,
                        old_window_size,
                    )

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
                )
            )

            self.encoder.patch_embed.grid_size = self.grid_size
            self.encoder.patch_embed.num_patches = self.grid_size[0] * self.grid_size[1]
            self.encoder.patch_embed.img_size = img_size

        if hasattr(self.encoder, "pos_embed"):
            if self.encoder.pos_embed.dim() == 4:
                pos_embed = resample_abs_pos_embed_nhwc(
                    self.encoder.pos_embed, [max(self.grid_size), max(self.grid_size)]
                )[:, : self.grid_size[0], : self.grid_size[1], :]
            else:
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
                )
                prefix_pos_embed = pos_embed[:, :num_prefix_tokens, :]
                pos_embed = pos_embed[:, num_prefix_tokens:, :]
                pos_embed = pos_embed.reshape(
                    1, max(self.grid_size), max(self.grid_size), -1
                )[:, : self.grid_size[0], : self.grid_size[1], :]
                pos_embed = torch.cat(
                    [prefix_pos_embed, pos_embed.flatten(1, 2)], dim=1
                )

            self.encoder.pos_embed = nn.Parameter(pos_embed)

    @staticmethod
    def interpolate_rel_pos(
        rel_pos, grid_size, old_grid_size, window_size=None, old_window_size=None
    ):
        block_size = (rel_pos.shape[0] + 1) / 2

        if block_size == old_grid_size:
            max_rel_dist = grid_size * 2 + 1
        elif block_size == old_window_size:
            if window_size is None:
                raise ValueError("window_size must be specified for non-global blocks")

            max_rel_dist = window_size * 2 + 1
        else:
            raise ValueError("invalid block size")

        max_rel_dist = int(max_rel_dist)

        rel_pos = rel_pos.reshape(1, rel_pos.shape[0], -1)
        rel_pos = rel_pos.permute(0, 2, 1)
        rel_pos = interpolate(rel_pos, size=max_rel_dist, mode="linear")
        rel_pos = rel_pos.reshape(-1, max_rel_dist).permute(1, 0)

        return nn.Parameter(rel_pos)

    def forward(self, x: torch.Tensor):
        x = (x - self.pixel_mean) / self.pixel_std

        x = self.encoder.forward_features(x)

        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x[:, self.encoder.num_prefix_tokens :]

        return x

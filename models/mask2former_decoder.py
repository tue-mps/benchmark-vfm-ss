import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerMLPPredictionHead,
    Mask2FormerSinePositionEmbedding,
)

from models.encoder import Encoder
from models.decoder_block import DecoderBlock


class Mask2formerDecoder(Encoder):
    def __init__(
        self,
        img_size,
        num_classes,
        encoder_name,
        sub_norm=False,
        patch_size=16,
        pretrained=True,
        num_queries=100,
        num_attn_heads=8,
        num_blocks=9,
        embed_dim=256,
        ckpt_path="",
    ):
        super().__init__(
            img_size=img_size,
            encoder_name=encoder_name,
            sub_norm=sub_norm,
            patch_size=patch_size,
            pretrained=pretrained,
            ckpt_path=ckpt_path,
        )

        self.num_attn_heads = num_attn_heads

        self.proj = nn.Linear(self.embed_dim, embed_dim)

        self.k_embed_pos = Mask2FormerSinePositionEmbedding(
            num_pos_feats=embed_dim // 2, normalize=True
        )

        self.q = nn.Embedding(num_queries, embed_dim)

        self.transformer_decoder = nn.ModuleList(
            [DecoderBlock(embed_dim, num_attn_heads) for _ in range(num_blocks)]
        )

        self.q_pos_embed = nn.Embedding(num_queries, embed_dim)

        self.q_norm = nn.LayerNorm(embed_dim)

        self.q_mlp = Mask2FormerMLPPredictionHead(embed_dim, embed_dim, embed_dim)

        self.q_class = nn.Linear(embed_dim, num_classes + 1)

    def _predict(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
    ):
        q_intermediate = self.q_norm(q)

        class_logits = self.q_class(q_intermediate).transpose(0, 1)

        mask_logits = torch.einsum("qbc, bchw -> bqhw", self.q_mlp(q_intermediate), x)

        attn_mask = (mask_logits < 0).bool().flatten(-2)
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        return attn_mask, mask_logits, class_logits

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        x = self.proj(x)

        q = self.q.weight
        q = q[:, None, :].repeat(1, x.shape[0], 1)

        v = x.transpose(0, 1)

        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.grid_size)

        k = v + self.k_embed_pos(x).flatten(2).permute(2, 0, 1)

        q_pos_embeds = self.q_pos_embed.weight
        q_pos_embeds = q_pos_embeds[:, None, :].repeat(1, x.shape[0], 1)

        mask_logits_per_layer, class_logits_per_layer = [], []

        for block in self.transformer_decoder:
            attn_mask, mask_logits, class_logits = self._predict(q, x)
            mask_logits_per_layer.append(mask_logits)
            class_logits_per_layer.append(class_logits)

            q: torch.Tensor = block(q, k, v, q_pos_embeds, attn_mask)

        _, mask_logits, class_logits = self._predict(q, x)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )

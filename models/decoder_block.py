from typing import Optional
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import Mask2FormerAttention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        decoder_ff_dim: int = 2048,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_attn_heads)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.self_attn = Mask2FormerAttention(
            embed_dim=embed_dim, num_heads=num_attn_heads
        )
        self.self_attn_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, decoder_ff_dim),
            nn.ReLU(),
            nn.Linear(decoder_ff_dim, embed_dim),
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_pos_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask[:, None, ...].repeat(1, self.cross_attn.num_heads, 1, 1)
            mask = mask.flatten(0, 1)

        residual = q
        q, _ = self.cross_attn(
            query=q if q_pos_embeds is None else q + q_pos_embeds,
            key=k,
            value=v,
            attn_mask=mask,
        )
        q = q + residual
        q = self.cross_attn_norm(q)

        residual = q
        q, _ = self.self_attn(
            hidden_states=q,
            position_embeddings=(
                torch.zeros_like(q) if q_pos_embeds is None else q_pos_embeds
            ),
        )
        q = q + residual
        q = self.self_attn_norm(q)

        residual = q
        q = self.ffn(q)
        q = q + residual
        q = self.final_layer_norm(q)

        return q

# https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/stdit/stdit.py

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F
import xformers.ops


approx_gelu = lambda: nn.GELU(approximate="tanh")

def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention_with_mask(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_norm = qk_norm


    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:

        if causal:
            attn_bias, x = xformers.ops.fmha.BlockDiagonalMask.from_tensor_list(x)
            attn_bias = attn_bias.make_causal()

            B, N, C = x.shape
            qkv = self.qkv(x)
            qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
            qkv_permute_shape = (2, 0, 1, 3, 4)
            qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            if self.qk_norm:
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        
            out = xformers.ops.fmha.memory_efficient_attention(q, k, v, attn_bias=attn_bias)

            x_output_shape = (B, N, C)
            x = out.reshape(x_output_shape) 
            x = self.proj(x)
            x = self.proj_drop(x)

            x = attn_bias.split(x)  
        else:
            B, N, C = x.shape
            qkv = self.qkv(x)
            qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
            qkv_permute_shape = (2, 0, 1, 3, 4)
            qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            if self.qk_norm:
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

            x = xformers.ops.memory_efficient_attention(query=q, key=k, value=v, p=self.attn_drop.p,)

            x_output_shape = (B, N, C)
            x = x.reshape(x_output_shape) 
            x = self.proj(x)
            x = self.proj_drop(x)

        return x


class STBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        st_use_qk_norm,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=True,
        enable_layernorm_kernel=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        
        self.attn_cls = Attention_with_mask

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
            qk_norm=st_use_qk_norm,
        )
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.d_s = d_s
        self.d_t = d_t

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=self.enable_flashattn,
            qk_norm=st_use_qk_norm,
        )

    def forward(self, concat_x, pad_len, tpe=None):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] 
        ).chunk(6, dim=1)
        x_m = t2i_modulate(self.norm1(concat_x), shift_msa, scale_msa)
        x_s = self.attn(x_m, causal=False,) 
        concat_x = concat_x + self.drop_path(gate_msa * x_s)
        split_x = list(torch.split(concat_x, split_size_or_sections=pad_len, dim=0))
        split_x_ = []
        for i, split_x_i in enumerate(split_x):
            if tpe is None:
                split_x_.append(split_x_i.transpose(0, 1))
            else:
                split_x_.append(split_x_i.transpose(0, 1) + tpe[:, :pad_len[i]])
        x_t = self.attn_temp(split_x_, causal=True,)
        x_t = [x_t_i.transpose(0, 1) for x_t_i in x_t]
        x_t = torch.cat(x_t, dim=0) 
        x = concat_x + self.drop_path(gate_msa * x_t)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x

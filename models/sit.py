# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# SPRINT: SPRINT: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers
# --------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F

from models.pos_embed import VisionRotaryEmbeddingFast

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.RMSNorm,
        use_v1_residual: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_v1_residual:
            self.v1_lambda = nn.Parameter(torch.tensor(0.5))
        self.v_last = None

        # for API compatibility with timm Attention
        self.fused_attn = False

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[VisionRotaryEmbeddingFast] = None,
        rope_ids: Optional[torch.Tensor] = None,
        v1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-head self-attention with optional 2D RoPE.

        x: (B, N, C)
        rope: VisionRotaryEmbeddingFast or None
        rope_ids: (B, N) or (N,) original (flattened) token indices for RoPE,
                  supporting routed / sparse subsets.
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, H, N, Dh)
        q, k, v = qkv.unbind(0)
        self.v_last = v
        q, k = self.q_norm(q), self.k_norm(k)

        if v1 is not None:
            v = self.v1_lambda * v1 + (1.0 - self.v1_lambda) * v

        if rope is not None:
            q = rope(q, rope_ids)
            k = rope(k, rope_ids)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_v1_residual: bool = True, **block_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs["qk_norm"],
            use_v1_residual=use_v1_residual,
        )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        feat_rope: Optional[VisionRotaryEmbeddingFast] = None,
        rope_ids: Optional[torch.Tensor] = None,
        v1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope,
            rope_ids=rope_ids,
            v1=v1,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, cls_token_dim):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.linear_cls = nn.Linear(hidden_size, cls_token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, cls=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)

        if cls is None:
            x = self.linear(x)
            return x, None
        else:
            cls_token = self.linear_cls(x[:, 0]).unsqueeze(1)
            x = self.linear(x[:, 1:])
            return x, cls_token.squeeze(1)


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone + SPRINT sparse-dense residual fusion.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        cls_token_dim=768,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.depth = depth

        # ----------------- SPRINT configuration -----------------
        # fθ / gθ / hθ split: default 2 / (D-4) / 2 as in the paper.
        self.num_f = 2
        self.num_h = 2
        self.num_g = self.depth - self.num_f - self.num_h
        assert self.num_g >= 0, "depth too small for SPRINT split"

        # Token drop ratio r (fraction of tokens to drop in sparse path)
        self.sprint_drop_ratio = 0.75

        # Path-drop learning probability p (drop whole sparse path during training)
        self.path_drop_prob = 0.05

        # [MASK] token for padding dropped positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Fusion projection: concat(ft, g_pad) → fused hidden
        self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        # --------------------------------------------------------

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, hidden_size), requires_grad=False)

        blocks = []
        for i in range(depth):
            use_v1_residual = i > 0
            blocks.append(
                SiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_v1_residual=use_v1_residual,
                    **block_kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            ])

        z_dim = self.z_dims[0]
        cls_token_dim = z_dim
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels, cls_token_dim)


        self.cls_projectors2 = nn.Linear(in_features=cls_token_dim, out_features=hidden_size, bias=True)
        self.wg_norm = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        # RoPE for spatial tokens
        head_dim = hidden_size // num_heads
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=head_dim // 2,
            pt_seq_len=hw_seq_len,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), cls_token=1, extra_tokens=1
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.linear_cls.weight, 0)
        nn.init.constant_(self.final_layer.linear_cls.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    # --------------------------- SPRINT helpers ---------------------------
    def _drop_tokens(self, x, drop_ratio):
        """
        Randomly drop a fraction of tokens (except we ensure at least one token kept).

        x: (B, T, C)
        drop_ratio: fraction of tokens to drop (0.0 ~ 1.0)
        Returns:
            x_keep: (B, T_keep, C)
            ids_keep: (B, T_keep) indices into original T, or None if no drop.
        """
        if drop_ratio <= 0.0:
            return x, None

        B, T, C = x.shape
        if T <= 1:
            return x, None

        num_keep = max(1, int(T * (1.0 - drop_ratio)))
        if num_keep >= T:
            return x, None

        device = x.device
        noise = torch.rand(B, T, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]  # (B, T_keep)
        x_keep = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
        return x_keep, ids_keep

    def _pad_with_mask(self, x_sparse, ids_keep, T_full):
        """
        x_sparse: (B, T_keep, C)
        ids_keep: (B, T_keep)
        T_full: full sequence length T
        Returns:
            x_pad: (B, T_full, C) with [MASK] at dropped positions.
        """
        if ids_keep is None:
            return x_sparse

        B, T_keep, C = x_sparse.shape
        assert T_full >= T_keep
        x_pad = self.mask_token.expand(B, T_full, C).clone()
        x_pad.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, C), x_sparse)
        return x_pad

    def _sprint_fuse(self, f_dense, g_full):
        """
        f_dense: (B, T, C) encoder output ft
        g_full: (B, T, C) padded sparse output g_pad
        Returns fused h: (B, T, C)
        """
        h = torch.cat([f_dense, g_full], dim=-1)  # (B, T, 2C)
        h = self.fusion_proj(h)
        return h

    # ---------------------------------------------------------------------
    def forward(self, x, t, y, return_logvar: bool = False, cls_token: Optional[torch.Tensor] = None, uncond: bool = False):
        """
        Forward pass of SiT with SPRINT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        # Patch embedding
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2

        # cls_token is expected (for your current pipeline)
        if cls_token is not None:
            cls_token = self.cls_projectors2(cls_token)
            cls_token = self.wg_norm(cls_token)
            cls_token = cls_token.unsqueeze(1)  # [B, 1, D]
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_embed
        else:
            # If you want to support no-cls mode, remove this exit and adjust pos_embed.
            exit()

        N, T, D = x.shape

        # RoPE position ids for full sequence (CLS + spatial tokens)
        hw = int(self.x_embedder.num_patches ** 0.5)
        assert hw * hw == self.x_embedder.num_patches
        device = x.device
        flat_pos = torch.arange(hw * hw, device=device, dtype=torch.long).view(1, -1)
        flat_pos = flat_pos.expand(N, -1)  # (N, HW)
        rope_ids_full = torch.cat(
            [torch.zeros(N, 1, device=device, dtype=torch.long), flat_pos], dim=1
        )  # (N, T)

        # timestep and class embedding
        t_embed = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t_embed + y

        # ------------------------------------------------------------------
        # 1) Encoder fθ on all tokens (dense, shallow)
        # ------------------------------------------------------------------
        x_enc = x
        rope_ids_enc = rope_ids_full

        v1_full = None
        for i in range(self.num_f):
            x_enc = self.blocks[i](x_enc, c, self.feat_rope, rope_ids_enc, v1=v1_full)  # (N, T, D)

            if v1_full is None:
                v1_full = self.blocks[i].attn.v_last

        # Use encoder output for z-projections (REPA / SPRINT z_t)
        zs = [
            projector(x_enc.reshape(-1, D)).reshape(N, -1, z_dim)
            for projector, z_dim in zip(self.projectors, self.z_dims)
        ]

        # ------------------------------------------------------------------
        # 2) Drop tokens to build sparse input to gθ (SPRINT sparse path)
        # ------------------------------------------------------------------
        if self.training:
            x_sparse, ids_keep = self._drop_tokens(x_enc, self.sprint_drop_ratio)
        else:
            x_sparse = x_enc
            ids_keep = None

        if ids_keep is not None:
            rope_ids_sparse = rope_ids_full.gather(1, ids_keep)
        else:
            rope_ids_sparse = rope_ids_full

        if ids_keep is not None:
            v1_sparse = v1_full.gather(
                2,
                ids_keep[:, None, :, None].expand(-1, v1_full.size(1), -1, v1_full.size(-1)),
            )
        else:
            v1_sparse = v1_full

        # ------------------------------------------------------------------
        # 3) Middle blocks gθ on sparse tokens
        # ------------------------------------------------------------------
        x_mid = x_sparse
        for i in range(self.num_f, self.num_f + self.num_g):
            x_mid = self.blocks[i](x_mid, c, self.feat_rope, rope_ids_sparse, v1=v1_sparse)  # (N, T_keep, D)

        # ------------------------------------------------------------------
        # 4) Pad back to full length with [MASK] to get g_pad
        # ------------------------------------------------------------------
        g_pad = self._pad_with_mask(x_mid, ids_keep, T_full=T)

        # ------------------------------------------------------------------
        # 5) Path-drop learning
        #    - During training: same behavior as before (stochastic path drop).
        #    - During sampling: enable path drop only for unconditional flow
        #      via the `uncond` flag.
        # ------------------------------------------------------------------
        if self.training and self.path_drop_prob > 0.0:
            # Sync random decision across all ranks
            drop_path = torch.rand(1, device=x.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(drop_path, src=0)
            if drop_path.item() < self.path_drop_prob:
                # Keep gradient flow but zero out the contribution
                g_pad = g_pad * 0.0 + self.mask_token.expand_as(g_pad)
        elif uncond: # Drop path for all samples
            g_pad = g_pad * 0.0 + self.mask_token.expand_as(g_pad)


        # ------------------------------------------------------------------
        # 6) Sparse–dense residual fusion: h_in = Fusion(ft, g_pad)
        # ------------------------------------------------------------------
        h_in = self._sprint_fuse(x_enc, g_pad)  # (N, T, D)

        # ------------------------------------------------------------------
        # 7) Decoder hθ on fused representation
        # ------------------------------------------------------------------
        x_dec = h_in
        for i in range(self.num_f + self.num_g, self.depth):
            x_dec = self.blocks[i](x_dec, c, self.feat_rope, rope_ids_full, v1=v1_full)

        x_out, cls_token_out = self.final_layer(x_dec, c, cls=cls_token)
        x_out = self.unpatchify(x_out)

        return x_out, zs, cls_token_out


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_1(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=1, num_heads=16, encoder_depth=8, **kwargs)

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, encoder_depth=8, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, encoder_depth=8, **kwargs)

def SiT_L_1(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=1, num_heads=16, encoder_depth=8, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, encoder_depth=8, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, encoder_depth=8, **kwargs)

def SiT_B_1(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=1, num_heads=12, encoder_depth=4, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, encoder_depth=4, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, encoder_depth=4, **kwargs)

def SiT_S_1(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, encoder_depth=4, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, encoder_depth=4, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, encoder_depth=4, **kwargs)

SiT_models = {
    'SiT-XL/1': SiT_XL_1,  'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,
    'SiT-L/1':  SiT_L_1,   'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,
    'SiT-B/1':  SiT_B_1,   'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,
    'SiT-S/1':  SiT_S_1,   'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,
}


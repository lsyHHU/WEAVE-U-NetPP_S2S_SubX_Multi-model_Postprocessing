#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8

class CrossAttentionWeightPredictorVarIn(nn.Module):
    """
    Cross-attention predictor that supports different input channel sizes per model,
    and accepts scalars as a single tensor of shape (B, N) where N = n_models.
    """
    def __init__(self,
                 in_ch_list,
                 n_models=None,
                 embed_dim: int = 64,
                 nhead: int = 8,
                 token_pool: int = 1,
                 temperature: float = 1.0,
                 use_layernorm: bool = True,
                 scalar_dim: int = None):
        super().__init__()
        if n_models is None:
            n_models = len(in_ch_list)
        assert len(in_ch_list) == n_models, "in_ch_list length must equal n_models"
        assert embed_dim % nhead == 0, "embed_dim must be divisible by nhead"

        self.n_models = n_models
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.token_pool = token_pool
        self.temperature = torch.tensor(float(temperature))
        self.use_layernorm = use_layernorm
        # scalar_dim is the dimension we map each scalar to (defaults to embed_dim)
        self.scalar_dim = scalar_dim or embed_dim

        # Per-model projection convs: map C_i -> D (1x1 conv preserves spatial structure)
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(in_c, embed_dim, kernel_size=1) for in_c in in_ch_list
        ])

        # per-model LayerNorm (applied over channel dim for each pixel)
        if self.use_layernorm:
            self.pn = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_models)])
        else:
            self.pn = None

        # learnable query embeddings: one per model
        self.query_embed = nn.Parameter(torch.randn(n_models, embed_dim) * 0.01)

        # multi-head attention
        self.mha = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)

        # refine MLP for queries
        self.query_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # scalar projector (maps scalar -> D) for each model
        # Now we expect scalars as a single tensor (B, N); we'll index columns inside forward
        self.scalar_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(n_models)
        ])

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_pool == 1:
            return x
        k = self.token_pool
        return F.avg_pool2d(x, kernel_size=k, stride=k, ceil_mode=True)

    def forward(self, feat_list, scalar_tensor=None):
        """
        feat_list: list of length n_models, each (B, C_i, H, W)
        scalar_tensor: optional tensor (B, N) where each column is that model's scalar for the batch.
                       If None, no scalar injection is performed.
        returns: weights (B, n, H, W), logits (B, n, H, W)
        """
        B = feat_list[0].shape[0]
        n = self.n_models
        assert len(feat_list) == n

        # 1) pool per-model features
        pooled = [self.spatial_pool(f) for f in feat_list]   # each (B, C_i, h, w)
        _, _, h, w = pooled[0].shape
        HW = h * w

        # 2) project each model to embed_dim using its 1x1 conv; result (B, D, h, w)
        proj_feats = []
        for i, p in enumerate(pooled):
            x = self.proj_convs[i](p)  # (B, D, h, w)
            if self.pn is not None:
                x_resh = x.view(B, self.embed_dim, HW).permute(0, 2, 1).contiguous()  # (B, HW, D)
                x_norm = self.pn[i](x_resh)  # LN on last dim
                x = x_norm.permute(0, 2, 1).view(B, self.embed_dim, h, w)
            proj_feats.append(x)

        # 3) optional: add scalar embedding to each per-pixel token
        if scalar_tensor is not None:
            # scalar_tensor: (B, N)
            assert scalar_tensor.shape[1] == n, "scalar_tensor second dim must equal n_models"
            for i in range(n):
                s = scalar_tensor[:, i].unsqueeze(-1)  # (B,1)
                se = self.scalar_fc[i](s)              # (B, D)
                se = se.unsqueeze(-1).unsqueeze(-1)    # (B, D, 1, 1)
                proj_feats[i] = proj_feats[i] + se

        # 4) form tokens sequence: for each model x -> (B, D, HW) -> permute -> (B, HW, D)
        token_seqs = [p.reshape(B, self.embed_dim, HW).permute(0, 2, 1) for p in proj_feats]  # list of (B, HW, D)
        tokens = torch.cat(token_seqs, dim=1)  # (B, n*HW, D)\

        # 5) queries attend to tokens
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, n, D)
        attn_out, _ = self.mha(queries, tokens, tokens)  # (B, n, D)
        attn_out = self.query_mlp(attn_out)  # refine

        # 6) representative per-pixel feature: mean across models for each HW
        per_pixel = tokens.view(B, n, HW, self.embed_dim).mean(dim=1)  # (B, HW, D)

        # 7) compute logits by dot product between attn_out (B,n,D) and per_pixel (B,HW,D)
        q = attn_out.unsqueeze(2)  # (B, n, 1, D)
        p = per_pixel.unsqueeze(1)  # (B, 1, HW, D)
        logits = (p * q).sum(-1)   # (B, n, HW)
        logits = logits.view(B, n, h, w)

        # 8) upsample if pooled
        H = feat_list[0].shape[-2]
        W = feat_list[0].shape[-1]
        if self.token_pool != 1:
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        # 9) temperature and softmax
        logits = logits / (self.temperature + EPS)
        weights = F.softmax(logits, dim=1)
        return weights, logits




# f1 = torch.randn(2, 5, 48, 80)
# f2 = torch.randn(2, 8, 48, 80)
# f3 = torch.randn(2, 11, 48, 80)
# f4 = torch.randn(2, 3, 48, 80)
# f = [f1, f2, f3, f4]

# lead = torch.randint(0, 5, (2, 4), dtype=torch.float32)

# model = CrossAttentionWeightPredictorVarIn(in_ch_list=[5,8,11,3], n_models=4)
# out = model(f, lead)
# weights, logits = model(f, lead)






# %%

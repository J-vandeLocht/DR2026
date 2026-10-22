import torch
import torch.nn as nn
import torch.nn.functional as F


def pos_enc_2d(H, W, D, device):
    """2D sine/cosine encoding (DETR-style). D must be divisible by 4."""
    d = D // 4
    freq = 1.0 / (10000 ** (torch.arange(d, device=device) / d))       # d
    y = torch.arange(H, device=device)[:, None] * freq                  # H×d
    x = torch.arange(W, device=device)[:, None] * freq                  # W×d
    y = torch.cat([y.sin(), y.cos()], dim=1)[:, None, :].expand(H, W, 2 * d)
    x = torch.cat([x.sin(), x.cos()], dim=1)[None, :, :].expand(H, W, 2 * d)
    return torch.cat([y, x], dim=-1).reshape(H * W, D)                  # (H·W)×D


class LinearMeanAdapter(nn.Module):
    """1x1 conv projection -> LayerNorm -> spatial mean. No positional
    encoding: a constant PE added before mean pooling would only shift every
    frame vector by the same offset."""

    tokens_per_frame = 1

    def __init__(self, in_ch, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f):
        t = self.proj(f).flatten(2).transpose(1, 2)  # B×(H·W)×D
        return self.norm(t).mean(dim=1)              # B×D


class LinearAttnPoolAdapter(nn.Module):
    """1x1 conv projection -> LayerNorm -> + 2D sin/cos PE -> attention
    pooling with num_queries learnable queries over the H·W spatial tokens.
    Returns B×D for one query, B×K×D (K region tokens per frame) otherwise.
    use_pe=False drops the PE (purely content-based pooling)."""

    def __init__(self, in_ch, dim, num_heads=4, num_queries=1, use_pe=True):
        super().__init__()
        self.tokens_per_frame = num_queries
        self.use_pe = use_pe
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.zeros(1, num_queries, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, f):
        f = self.proj(f)                             # B×D×H×W
        B, D, H, W = f.shape
        t = f.flatten(2).transpose(1, 2)             # B×(H·W)×D
        t = self.norm(t)
        if self.use_pe:
            t = t + pos_enc_2d(H, W, D, t.device).to(t.dtype)
        pooled, _ = self.attn(self.query.expand(B, -1, -1), t, t, need_weights=False)
        pooled = self.out_norm(pooled)               # B×K×D
        return pooled.squeeze(1) if self.tokens_per_frame == 1 else pooled


class GridPoolAdapter(nn.Module):
    """1x1 conv projection -> LayerNorm -> average (or max) pooling into a fixed
    grid×grid layout (2 = quadrants), one token per cell: B×grid²×D."""

    def __init__(self, in_ch, dim, grid=2, pool="avg"):
        super().__init__()
        self.tokens_per_frame = grid * grid
        self.grid = grid
        self.pool = {"avg": F.adaptive_avg_pool2d, "max": F.adaptive_max_pool2d}[pool]
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f):
        t = self.norm(self.proj(f).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # B×D×H×W
        return self.pool(t, self.grid).flatten(2).transpose(1, 2)  # B×grid²×D


ADAPTERS = {
    "linear_mean": LinearMeanAdapter,
    "linear_attnpool": LinearAttnPoolAdapter,
    "grid": GridPoolAdapter,
}


def build_adapter(name, in_ch, dim, dense_tokens=None, adapter_pe=True, grid_pool="avg"):
    """dense_tokens: number of queries for linear_attnpool, grid side for grid
    (None = the adapter's default); linear_mean always gives one token.
    adapter_pe: 2D sin/cos PE in linear_attnpool (the others have none).
    grid_pool: 'avg' or 'max' pooling per grid cell (grid only)."""
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter '{name}', choose from {list(ADAPTERS)}")
    if not adapter_pe and name != "linear_attnpool":
        raise ValueError(f"adapter_pe=False only applies to linear_attnpool, not '{name}'")
    if grid_pool != "avg" and name != "grid":
        raise ValueError(f"grid_pool='{grid_pool}' only applies to grid, not '{name}'")
    kw = {} if adapter_pe else {"use_pe": False}
    if grid_pool != "avg":
        kw["pool"] = grid_pool
    if dense_tokens is not None and name != "linear_mean":
        kw[{"linear_attnpool": "num_queries", "grid": "grid"}[name]] = dense_tokens
    return ADAPTERS[name](in_ch, dim, **kw)

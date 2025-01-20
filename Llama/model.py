import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.head_dim * self.n_head != self.dim:
            raise ValueError(
                f"head_dim * n_head ({self.head_dim * self.n_head}) must equal dim ({self.dim})"
            )
    
    @property
    def head_dim(self):
        return self.dim // self.n_head

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: str = None) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device if device is None else device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class GQA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_head
        self.n_kv_heads = args.n_kv_head
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.dropout = nn.Dropout(args.dropout)

        self.wq = nn.Linear(args.dim, args.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_head * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos: Optional[int] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if start_pos is not None:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys, values = xk, xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention = torch.nn.functional.scaled_dot_product_attention(xq, keys, values, is_causal=True)
        output = attention.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.dropout(self.wo(output))

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attention = GQA(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: Optional[int] = None) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config))
        
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.config.head_dim,
            self.config.max_seq_len * 2,
            device=config.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: Optional[int] = None) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        h = self.dropout(self.tok_embeddings(tokens))
        
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen] if start_pos is not None else self.freqs_cis[:seqlen]
        
        for layer in self.layers:
            h = layer(h, freqs_cis, start_pos)
        
        h = self.norm(h)
        output = self.output(h)
        
        return output

    def init_weights(self) -> None:
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_init_weights)
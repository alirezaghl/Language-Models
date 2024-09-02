import argparse
import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0))

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, q, k, v):
        B, S, E = q.size()
        q = self.w_q(q).view(B, S, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, S, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, S, self.n_head, self.d_k).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.mask[:,:,:S,:S] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = (attn @ v).transpose(1, 2).contiguous().view(B, S, E)
        y = self.proj_dropout(self.w_o(y))

        return y

class Block(nn.Module):
    def __init__(self, d_model, n_head, context_length):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, n_head, context_length)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x), x, x, x)
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, d_model, n_head, vocab_size, context_length, num_blocks):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_blocks = num_blocks

        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, context_length) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.te.weight  # Tie weights to reducing number of trainable parameters

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        assert seq_len <= self.context_length

        pos = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        tok_emb = self.te(idx)
        pos_emb = self.pe(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.lm_head(x)

        return logits

def main(args):
    model = GPT(args.d_model, args.n_heads, args.vocab_size, args.context_length, args.num_blocks)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_parameters:,}")
    print(model.parameters())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Model")
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--num_blocks', type=int, default=12, help='Number of transformer blocks')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Size of the vocabulary')
    parser.add_argument('--context_length', type=int, default=1024, help='Context length')

    args = parser.parse_args()
    main(args)
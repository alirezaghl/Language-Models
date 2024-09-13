import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math
import numpy as np
from dataclasses import dataclass
import os
import urllib.request
import time
import urllib.request
import pickle


@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50304
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  n_epochs = 30
  tokenizer = tiktoken.get_encoding('gpt2')
  std = 0.02


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Text file
file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head

        self.register_buffer("mask", torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1))

        self.w_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.w_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.w_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.w_o = nn.Linear(config.n_embd, config.n_embd, bias=False)
      

    def forward(self, x):
        batch_size, context_length, d_model = x.size()
        query = self.w_q(x)
        query = query.view(batch_size, context_length, self.n_head, self.d_k).transpose(1,2) # ---> (batch_size, n_head, context_length, d_k)
        key = self.w_k(x)
        key = key.view(batch_size, context_length, self.n_head, self.d_k).transpose(1,2) # ---> (batch_size, n_head, context_length, d_k)
        value = self.w_v(x)
        value = value.view(batch_size, context_length, self.n_head, self.d_k).transpose(1,2) # ---> (batch_size, n_head, context_length, d_k)
        attention_weights = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True) # Flash Attention
        context = attention_weights.transpose(1, 2).contiguous().view(batch_size, context_length, d_model)
        context = self.w_o(context)

        return context, attention_weights

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.gelu = nn.GELU(approximate='tanh')

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x
  

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.mlp = MLP(config)
  
  def forward(self, x):
    attn_output, attn_weights = self.attn(self.ln1(x))
    x = x + attn_output
    x = x + self.mlp(self.ln2(x))
    return x, attn_weights

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd),
      wpe = nn.Embedding(config.block_size, config.n_embd),
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
      ln_f = nn.LayerNorm(config.n_embd) 
    ))
    self.transformer.to(device)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False).to(device)
   
  
  # adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py
  def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer

     
  
  def forward(self, idx, targets=None):
    B, T = idx.size()
    assert T <= self.config.block_size
    pe = self.transformer.wpe(torch.arange(T, device=idx.device))
    te = self.transformer.wte(idx)
    x = te + pe
    attention_weights = []
    for block in self.transformer.h:
      x, attention_weight = block(x)
      attention_weights.append(attention_weight)

    logits = self.lm_head(x)
    loss = None
    if targets is not None:
      loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, loss, attention_weights


  


class DataLoader:
  def __init__(self, B, T):
    with open(file_path, 'r') as f:
      text = f.read()
    
    self.B = B
    self.T = T
    enc = GPTConfig.tokenizer
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens, dtype=torch.long)
    self.current_position = 0

  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position: self.current_position+B*T+1]
    x = (buf[:-1]).view(B, T)
    y = (buf[1:]).view(B, T)
    self.current_position += B*T
    if self.current_position + (B*T*+1) > len(self.tokens):
        self.current_position = 0

    return x, y
  




def generate_text(model, idx, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        
        idx_cond = idx[:, -context_length:]
        
        with torch.no_grad():
            logits, _, _ = model(idx_cond)
        
        logits = logits[:, -1, :]  
        probs = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_length = GPTConfig.block_size
    encoded = tokenizer.encode(start_context, allowed_special={'<|endoftext|>'})
    encoded = torch.tensor(encoded).unsqueeze(0)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded,
            max_new_tokens=100, context_length=context_length
        )
    decoded_text =  tokenizer.decode(token_ids.squeeze(0).tolist())
    print(decoded_text.replace("\n", " "))  
    model.train()





def train_model_simple(model, train_loader, optimizer, device, num_epochs,
                       start_context, tokenizer):
    train_losses= []
    total_tokens = []

    for epoch in range(num_epochs):
        model.train()  
        
        t0 = time.time()
        optimizer.zero_grad() 
        x, y = train_loader.next_batch()
        x.to(device), y.to(device)
        input_tokens = x.numel()
        logits, loss, _ = model(x, y)
        loss.backward() 
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step() 
        t1 = time.time()
        dt = (t1 - t0) * 1000
        token_per_second = train_loader.B * train_loader.T / (t1-t0)
        print(f'step{epoch} | loss: {loss.item():.6f} | time: {dt:.2f}ms | norm: {norm:.3f}, tokens/s: {token_per_second:.2f}')
        train_losses.append(loss.item())
        total_tokens.append(input_tokens)


        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, total_tokens



def main():
    train_loader = DataLoader(B=4, T=1024)
    torch.set_float32_matmul_precision('high')
    model = GPT(GPTConfig())
    model.to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device=device)
    
    train_losses, total_tokens = train_model_simple(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=GPTConfig.n_epochs,
        start_context="Every effort moves you",
        tokenizer=GPTConfig.tokenizer
    )
    
    # save the model after training
    torch.save(model.state_dict(), 'gpt2_model.pth')
    print("Model training complete and saved as 'gpt2_model.pth'.")
    # save the train_loss
    train_losses = np.array(train_losses)
    np.save('train_losses.npy', train_losses)
    print("Training complete and train losses saved.")
    # save total tokens
    total_tokens = np.array(total_tokens)
    np.save('total_tokens.npy', total_tokens)

if __name__ == "__main__":
    main()
import torch
import torch.optim as optim
from transformers import GPT2Tokenizer
import time
import numpy as np
import requests
from model import ModelArgs, Transformer

class DataLoader:
    def __init__(self, file_path, B, T, download=True):
        if download:
            print("Downloading Shakespeare dataset...")
            response = requests.get(file_path)
            text = response.text
        else:
            with open(file_path, 'r') as f:
                text = f.read()
        
        self.B = B  # batch size
        self.T = T  # sequence length
        
        print("Initializing tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_auth_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        
        print("Tokenizing text...")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.current_position = 0
        self.vocab_size = len(tokenizer)
        
        print(f"Data loaded. Total tokens: {len(self.tokens)}")
        print(f"Vocabulary size: {self.vocab_size}")

    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        
        buf = self.tokens[self.current_position: self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T
        return x, y

def train_model(model, train_loader, optimizer, device, num_epochs, start_context):
    train_losses = []
    total_tokens = []

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        
        t0 = time.time()
        optimizer.zero_grad()
        
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        
        input_tokens = x.numel()
        
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        t1 = time.time()
        dt = (t1 - t0) * 1000
        token_per_second = train_loader.B * train_loader.T / (t1-t0)
        if epoch % 50 == 0:
            print(f'step {epoch} | loss: {loss.item():.6f} | time: {dt:.2f}ms | grad_norm: {norm:.3f} | tokens/s: {token_per_second:.2f}')
            
        train_losses.append(loss.item())
        total_tokens.append(input_tokens)

    return train_losses, total_tokens

def setup_training(config_args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = DataLoader(
        file_path='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        B=4,  
        T=128,  
        download=True  
    )
    
    if config_args is None:
        config_args = dict(
            dim=128,
            n_layers=8,
            n_head=4,
            n_kv_head=1,
            vocab_size=train_loader.vocab_size,
            max_seq_len=512,
            max_batch_size=32,
            dropout=0.1,
            device=device
        )
    
    config = ModelArgs(**config_args)
    model = Transformer(config)
    model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    return model, optimizer, train_loader, device
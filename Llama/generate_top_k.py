import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import time
import numpy as np
from model import ModelArgs, Transformer

def generate_text(model, tokenizer, idx, max_new_tokens, device, temperature=0.8, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.max_seq_len:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]  
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if temperature > 0.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def generate_samples(model, tokenizer, device, prompt, num_samples=3, max_tokens=100, 
                    temperature=0.8, top_k=None):
    samples = []
    
    
    encoded = tokenizer.encode(prompt)
    encoded = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    for i in range(num_samples):
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            idx=encoded,
            max_new_tokens=max_tokens,
            device=device,
            temperature=temperature,
            top_k=top_k
        )
        
        text = tokenizer.decode(output[0].tolist())
        samples.append(text)
        print(f"\nsample {i+1}:")
        print(text.replace('\n', ' '))
    
    return samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_auth_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = ModelArgs(
        dim=128,
        n_layers=8,
        n_head=4,
        n_kv_head=1,
        vocab_size=len(tokenizer),
        max_seq_len=512,
        max_batch_size=32,
        dropout=0.1,
        device=device
    )
    
    model = Transformer(config)
    model.load_state_dict(torch.load('transformer_model.pth'))
    model.to(device)
    model.eval()
    
    # Test prompts
    test_prompts = [
        "ROMEO: my love,",
        "JULIET: romeo,",
        "MERCUTIO: plague"
    ]
    
    generation_params = [
        {'temperature': 0.6, 'top_k': None},  
        {'temperature': 0.8, 'top_k': 50},     
        {'temperature': 1.0, 'top_k': 100}     
    ]
    
    all_samples = {}
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"generating for prompt: {prompt}")
        prompt_samples = []
        
        for params in generation_params:
            samples = generate_samples(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                num_samples=2,
                max_tokens=150,
                temperature=params['temperature'],
                top_k=params['top_k']
            )
            prompt_samples.extend(samples)
        
        all_samples[prompt] = prompt_samples
    
    with open('shakespeare_samples_top_k.txt', 'w', encoding='utf-8') as f:
        for prompt, samples in all_samples.items():
            f.write(f"\nPrompt: {prompt}\n")
            for i, sample in enumerate(samples, 1):
                f.write(f"\nSample {i}:\n{sample}\n")
                f.write("-"*30 + "\n")
    

if __name__ == "__main__":
    main()

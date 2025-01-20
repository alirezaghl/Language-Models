import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from model import ModelArgs, Transformer

def sample_top_p(probs, p):

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # Calculate cumulative probabilities
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # Find indices where:
    mask = probs_sum - probs_sort > p
    
    # Zero out probabilities outside the nucleus
    probs_sort[mask] = 0.0
    
    # Renormalize probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # Sample from the filtered distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    
    # Convert back to original token indices
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def generate_text(model, tokenizer, idx, max_new_tokens, device, temperature=0.8, top_p=0.9):
    """Generate text using nucleus sampling"""
    model.eval()
    for _ in range(max_new_tokens):
        # Get conditional input
        idx_cond = idx[:, -model.config.max_seq_len:]
        
        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]  
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Apply nucleus sampling
            next_token = sample_top_p(probs, top_p)
            
            # Append token
            idx = torch.cat((idx, next_token), dim=1)
            
            # Stop if we generate an end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return idx

def generate_samples(model, tokenizer, device, prompt, num_samples=2, max_tokens=150, 
                    temperature=0.8, top_p=0.9):
    """Generate multiple samples using nucleus sampling"""
    print(f"\nGenerating samples for: '{prompt}'\n")
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    encoded = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    
    samples = []
    for i in range(num_samples):
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            idx=encoded,
            max_new_tokens=max_tokens,
            device=device,
            temperature=temperature,
            top_p=top_p
        )
        
        text = tokenizer.decode(output[0].tolist())
        samples.append(text)
        
        print(f"\nSample {i+1}:")
        print(text)
        print('-' * 40)
    
    return samples

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_auth_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    prompts = [
        "ROMEO: My heart yearns for",
        "JULIET: O gentle Romeo, wherefore"
    ]
    
    for prompt in prompts:
        # Generate with different sampling parameters
        print("\nConservative sampling (lower temperature, higher top-p):")
        generate_samples(model, tokenizer, device, prompt, 
                       temperature=0.7, top_p=0.95)
        
        print("\nCreative sampling (higher temperature, lower top-p):")
        generate_samples(model, tokenizer, device, prompt,
                       temperature=0.9, top_p=0.8)
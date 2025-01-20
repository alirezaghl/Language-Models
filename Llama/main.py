import argparse
import torch
from transformers import GPT2Tokenizer
from model import ModelArgs, Transformer
from train import DataLoader, train_model
import generate_top_k as top_k
import generate_top_p as top_p
import torch.optim as optim
import time
import numpy as np

def setup_model(vocab_size, device):
    """Initialize model with default configuration"""
    config = ModelArgs(
        dim=128,
        n_layers=8,
        n_head=4,
        n_kv_head=1,
        vocab_size=vocab_size,
        max_seq_len=512,
        max_batch_size=32,
        dropout=0.1,
        device=device
    )
    
    model = Transformer(config)
    model.to(device)
    return model

def train(args):
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Setup data loader
    train_loader = DataLoader(
        file_path='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        B=4,  # batch size
        T=128,  # sequence length
        download=True
    )
    
    # Initialize model and optimizer
    model = setup_model(train_loader.vocab_size, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Train model
    print("\nStarting training...")
    train_losses, total_tokens = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        start_context="ROMEO:"
    )
    
    # Save results
    print("\nSaving model and training statistics...")
    torch.save(model.state_dict(), args.model_path)
    np.save('train_losses.npy', np.array(train_losses))
    np.save('total_tokens.npy', np.array(total_tokens))
    print("Training complete!")

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_auth_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = setup_model(len(tokenizer), device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Default prompts if none provided
    prompts = args.prompts if args.prompts else [
        "ROMEO: My heart yearns for",
        "JULIET: O gentle Romeo, wherefore",
        "MERCUTIO: A plague"
    ]
    
    print(f"\nGeneration method: {args.method}")
    print(f"Temperature: {args.temperature}")
    print(f"Number of samples per prompt: {args.num_samples}")
    
    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"Generating for prompt: {prompt}")
        
        if args.method == 'top-k':
            samples = top_k.generate_samples(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
        else:  # top-p
            samples = top_p.generate_samples(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
    
    print("\nGeneration complete!")

def main():
    parser = argparse.ArgumentParser(description='Train or generate text with the Transformer model')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    train_parser.add_argument('--model-path', type=str, default='transformer_model.pth', 
                             help='Path to save/load model')
    
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--method', choices=['top-k', 'top-p'], default='top-p',
                           help='Sampling method to use')
    gen_parser.add_argument('--model-path', type=str, default='transformer_model.pth',
                           help='Path to load model from')
    gen_parser.add_argument('--temperature', type=float, default=0.8,
                           help='Sampling temperature')
    gen_parser.add_argument('--top-k', type=int, default=50,
                           help='Top-k value for sampling')
    gen_parser.add_argument('--top-p', type=float, default=0.9,
                           help='Top-p value for nucleus sampling')
    gen_parser.add_argument('--max-tokens', type=int, default=150,
                           help='Maximum number of tokens to generate')
    gen_parser.add_argument('--num-samples', type=int, default=2,
                           help='Number of samples to generate per prompt')
    gen_parser.add_argument('--prompts', nargs='+', help='Prompts to generate from')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'generate':
        generate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
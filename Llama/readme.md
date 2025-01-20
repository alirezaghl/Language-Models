# Llama Implementation in PyTorch

A PyTorch implementation of the Llama architecture with both top-k and nucleus (top-p) sampling for text generation. This project includes a complete training pipeline and two different text generation approaches.

## Features

- Full Llama architecture implementation with:
  - Grouped Query Attention (GQA)
  - RMSNorm for layer normalization
  - SwiGLU activation in feed-forward layers
  - Rotary positional embeddings
- Training implementation with Shakespeare dataset
- Two text generation methods:
  - Top-k sampling
  - Nucleus (top-p) sampling
- Command-line interface for both training and generation

## Project Structure

```
.
├── model.py            # Llama model architecture
├── train.py           # Training implementation
├── generate_top_k.py  # Top-k sampling generation
├── generate_top_p.py  # Nucleus sampling generation
├── main.py           # Command-line interface
└── README.md         # This file
```

## Requirements

```bash
pip install torch
pip install transformers
pip install requests
pip install numpy
```

## Usage

### Training

To train the model:

```bash
python main.py train --epochs 1000 --model-path transformer_model.pth
```

Training arguments:
- `--epochs`: Number of training epochs (default: 1000)
- `--model-path`: Path to save the trained model (default: transformer_model.pth)

### Text Generation

To generate text using nucleus sampling (top-p):

```bash
python main.py generate --method top-p \
                       --temperature 0.8 \
                       --top-p 0.9 \
                       --num-samples 2 \
                       --prompts "ROMEO: My love," "JULIET: Wherefore art thou"
```

To generate text using top-k sampling:

```bash
python main.py generate --method top-k \
                       --temperature 0.8 \
                       --top-k 50 \
                       --num-samples 2
```

Generation arguments:
- `--method`: Sampling method ('top-k' or 'top-p')
- `--model-path`: Path to load model from (default: transformer_model.pth)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top-k`: Top-k value for sampling (default: 50)
- `--top-p`: Top-p value for nucleus sampling (default: 0.9)
- `--max-tokens`: Maximum number of tokens to generate (default: 150)
- `--num-samples`: Number of samples per prompt (default: 2)
- `--prompts`: Custom prompts to generate from (optional)

## Model Architecture

The implementation includes:

- **Grouped Query Attention (GQA)**: Handles attention computation by grouping queries
- **RMSNorm**: Root Mean Square Layer Normalization for better training stability
- **SwiGLU Activation**: Enhanced activation function in feed-forward layers
- **Rotary Positional Embeddings**: For handling positional information

Model configuration parameters:
```python
config = ModelArgs(
    dim=128,              # Model dimension
    n_layers=8,           # Number of transformer layers
    n_head=4,             # Number of attention heads
    n_kv_head=1,          # Number of key/value heads for GQA
    vocab_size=len(tokenizer),
    max_seq_len=512,      # Maximum sequence length
    max_batch_size=32,    # Maximum batch size
    dropout=0.1           # Dropout rate
)
```

## Training Details

The model is trained on the Shakespeare dataset with:
- AdamW optimizer
- Learning rate: 1e-3
- Weight decay: 0.1
- Batch size: 4
- Sequence length: 128

## Generation Methods

### Top-k Sampling
- Filters the probability distribution to only the top k tokens
- Configurable via the `--top-k` parameter

### Nucleus (Top-p) Sampling
- Samples from the smallest set of tokens whose cumulative probability exceeds p
- Configurable via the `--top-p` parameter




## References & Acknowledgments

This implementation draws inspiration from and references the following excellent resources:

1. **Video Tutorials and code references**:
   - [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=C5AQmwkzjtL1q9Nz) by Andrej Karpathy
   - [Let's Build Llama 3 From Scratch, in Code, Spelled Out](https://www.youtube.com/watch?v=lZj8F6EspVU) by tunadorable
   - [Coding LLaMA 2 from scratch in PyTorch - KV Cache, Grouped Query Attention, Rotary PE, RMSNorm](https://www.youtube.com/watch?v=oM4VmoabDAI) by Umar Jamil
   - [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka
   - Original Llama paper and architecture by Meta



2. **Training Data**:
   - Tiny Shakespeare dataset


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

- **Grouped Query Attention (GQA)**: Efficiently handles attention computation by grouping queries
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
- Good for controlling the randomness while maintaining diversity
- Configurable via the `--top-k` parameter

### Nucleus (Top-p) Sampling
- Samples from the smallest set of tokens whose cumulative probability exceeds p
- Adaptive approach that can handle different probability distributions
- Configurable via the `--top-p` parameter

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

MIT License

## References & Acknowledgments

This implementation draws inspiration from and references the following excellent resources:

1. **Video Tutorials**:
   - [Let's Build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy
   - [Building Llama from scratch](https://www.youtube.com/watch?v=lZj8F6EspVU) by Jorge Morais
   - [LLM Course](https://www.youtube.com/watch?v=oM4VmoabDAI) by Sebastian Raschka

2. **Code References**:
   - [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka
   - [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy
   - Original Llama paper and architecture

3. **Training Data**:
   - Tiny Shakespeare dataset


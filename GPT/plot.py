import torch
import matplotlib.pyplot as plt
import tiktoken
from dataclasses import dataclass
from gpt2 import GPT
from sklearn.decomposition import PCA
import urllib.request
import seaborn as sns
import numpy as np


@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50304
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  n_epochs = 30
  tokenizer = tiktoken.get_encoding('gpt2')

model = GPT(GPTConfig())
model.load_state_dict(torch.load('gpt2_model.pth'))
model.eval()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

model = GPT(GPTConfig())  
model.load_state_dict(torch.load('gpt2_model.pth'))
model.eval()

train_loss = np.load('train_losses.npy', allow_pickle=True)
total_tokens = np.load('total_tokens.npy', allow_pickle=True)



print(f"Training Loss: {train_loss}")
print(f"Total Tokens Seen: {total_tokens}")

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses=None):
    fig, ax1 = plt.subplots(figsize=(7, 5))  

    ax1.plot(epochs_seen, train_losses, label="Training Loss", color='blue', linewidth=2)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    if val_losses is not None:
        ax1.plot(epochs_seen, val_losses, label="Validation Loss", color='orange', linewidth=2)
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  
    ax2.set_xlabel("Tokens Seen")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  

    ax1.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    plt.savefig("training_validation.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, GPTConfig().n_epochs, len(train_loss))

plot_losses(epochs_tensor, total_tokens, train_loss, val_losses=None)
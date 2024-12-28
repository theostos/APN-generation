import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch

file_in = 'data/export_0.pt'

dataset = torch.load(file_in, weights_only=True)
dataset = dataset.flatten()
token_counts = torch.bincount(dataset, minlength=64)

# Calculate probabilities
total_tokens = token_counts.sum()
probabilities = token_counts / total_tokens

# Avoid log(0) by masking zero probabilities
non_zero_probs = probabilities[probabilities > 0]

# Compute entropy
entropy = -torch.sum(non_zero_probs * torch.log2(non_zero_probs))
print(entropy)
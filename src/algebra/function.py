from math import sqrt
from functools import cache
import json

import torch

@cache
def load_add_table(exponent):
    with open(f"table/add/2_{exponent}", "r") as fp:
        T = json.load(fp)
    T = torch.tensor(T)
    return T

def add_table(F0, F1, T):
    return T[F0[...], F1[...]]

def compute_derivative(F, T):
    # Unpack shapes
    batch, field_size = F.shape

    # Create indices for broadcasting
    indices = torch.arange(field_size, device=F.device).view(1, -1, 1)  # Shape: (1, elements, 1)
    add_indices = torch.arange(field_size, device=F.device).view(1, 1, -1)  # Shape: (1, 1, elements)

    # Compute all translations using the add table
    # Shape of translated_indices: (1, elements, elements)
    translated_indices = T[indices, add_indices]

    # Gather values from the lookup table based on the translations
    # Shape of R: (len_batch, elements, elements)
    F1 = F[:, translated_indices].squeeze(1)
    # .view(batch*field_size, field_size)
    F = F.unsqueeze(1).expand(batch, field_size, field_size)
    return add_table(F, F1, T)[:,:-1,:]

def compute_delta_table(DF):
    """
    Evaluate delta coefficient of a family of functions

    Args:
        DF (torch.Tensor): Tensor of shape (batch_f, field_size-1, field_size), for each function the lookup table of its derivative.
    Returns:
        torch.Tensor: Result of shape (batch_p, batch_input).
    """
    batch = DF.size(0)
    field_size = DF.size(2)
    # Flatten the first two dimensions
    flat_DF = DF.reshape(batch * (field_size-1), field_size)  # Shape: (batch * (field_size-1), field_size)

    # Initialize the max_counts tensor
    xrow = flat_DF.size(0)
    xlim = field_size
    minl = field_size * xrow

    # beware, overflow if xflow*field_size*field_size is greater than max_int
    assert sqrt(DF.size(0)) < 2**(31/2)/field_size, "Bincount overflow, please reduce batch size"
    xlab = flat_DF + xlim * torch.arange (xrow, device=DF.device).unsqueeze (1)
    xtmp = torch.bincount (xlab.flatten(), minlength = minl)
    xcnt = xtmp.reshape (xrow, xlim)
    return xcnt

def compute_delta_max(DF):
    """
    Evaluate delta coefficient of a family of functions

    Args:
        DF (torch.Tensor): Tensor of shape (batch_f, field_size-1, field_size), for each function the lookup table of its derivative.
    Returns:
        torch.Tensor: Result of shape (batch_p, batch_input).
    """
    xcnt = compute_delta_table(DF)
    values, _ = xcnt.max(dim=-1)
    values = values.view(DF.size(0), -1)
    values, _ = values.max(dim=-1)
    return values

def compute_delta_mean(DF):
    """
    Evaluate delta coefficient of a family of functions

    Args:
        DF (torch.Tensor): Tensor of shape (batch_f, field_size-1, field_size), for each function the lookup table of its derivative.
    Returns:
        torch.Tensor: Result of shape (batch_p, batch_input).
    """
    xcnt = compute_delta_table(DF).float()
    values,_ = xcnt.max(dim=-1)
    values = values.view(DF.size(0), -1)
    values = values.mean(dim=-1)
    return values

def filter_functions_delta(F, deltas, treshold):
    return F[deltas <= treshold]

def score_functions(F, T):
    DF = compute_derivative(F, T)
    deltas_max = compute_delta_max(DF)
    deltas_mean = compute_delta_mean(DF)
    return deltas_max, deltas_mean

def check_if_permutation(F):
    F_sort, _ = torch.sort(F)
    field_size = F.size(1)

    return torch.all(F_sort == torch.arange(0, field_size, device=F.device), dim=1)

def compute_pre_compositions(permutation, random_permutations):
    # Compute pre-composition of `permutation` with each random permutation
    # permutation: shape (1, size)
    # random_permutations: shape (num_samples, size)
    return random_permutations[:, permutation[0]]

def generate_composed_permutations(permutation):
    size = permutation.shape[1]
    
    # Generate all pairs of indices for transpositions (i, j) where i < j
    idx_i, idx_j = torch.triu_indices(size, size, offset=1, device = permutation.device)  # Upper triangular indices
    num_transpositions = idx_i.shape[0]
    
    # Expand the permutation tensor for batch processing
    batch_permutations = permutation.repeat(num_transpositions, 1)  # Shape: (num_transpositions, size)
    
    # Create a copy of the batch permutations to apply transpositions
    swapped_permutations = batch_permutations.clone()
    
    # Swap the indices using advanced indexing
    swapped_permutations[torch.arange(num_transpositions, device=permutation.device), idx_i], \
    swapped_permutations[torch.arange(num_transpositions, device=permutation.device), idx_j] = \
    batch_permutations[torch.arange(num_transpositions, device=permutation.device), idx_j], \
    batch_permutations[torch.arange(num_transpositions, device=permutation.device), idx_i]
    return torch.cat([permutation, swapped_permutations])

def precompose(A, B):
    batch_size, n = A.shape
    # Expand B to get the indices for each batch
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, n)
    # Index A using B
    return A[batch_indices, B]
from tqdm import tqdm
import torch

from src.algebra.generator import generate_random_permutations_vectorized
from src.algebra.function import compute_pre_compositions, compute_derivative, compute_delta_mean

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

def random_improvement(T, F, num_perm, max_k=10):
    random_perms = generate_random_permutations_vectorized(T.size(0), num_perm, max_k, device=T.device)
    new_F = compute_pre_compositions(F, random_perms)
    DF = compute_derivative(new_F, T)
    delta_mean = compute_delta_mean(DF)
    print(torch.min(delta_mean))

def local_improvement(T, F):
    old_delta = T.size(0) + 1
    new_delta = T.size(0)
    while new_delta < old_delta:
        old_delta = new_delta
        new_F = generate_composed_permutations(F)
        DF = compute_derivative(new_F, T)
        delta_mean = compute_delta_mean(DF)
        
        new_delta, idx = torch.min(delta_mean, dim=0)
        new_delta = new_delta.item()
        # to avoid to keep everything in memory, we clone the view
        F = new_F[idx,:].unsqueeze(0).clone()
    return F, new_delta

def improve_top_k(F, T, k=10):
    all_values, all_functions = [], []
    for ex in F:
        ex = ex.unsqueeze(0)
        new_F = generate_composed_permutations(ex)
        DF = compute_derivative(new_F, T)
        delta_mean = compute_delta_mean(DF)

        values, indices = torch.topk(-delta_mean, k=k)
        all_values.append(-values)
        all_functions.append(new_F[indices,:])
    
    all_values = torch.concat(all_values)
    all_functions = torch.concat(all_functions)

    values, indices = torch.topk(-all_values, k=k)
    return all_functions[indices,:].clone(), -values

def improve_beam(F,T, n_iter, k=10):
    for _ in range(n_iter):
        F, delta = improve_top_k(F,T,k=k)
    return F[0,...].unsqueeze(0).clone(), delta[0]

def improve_score_functions(F, T):
    deltas = []
    for ex in F:
        ex = ex.unsqueeze(0)
        _, new_delta = local_improvement(T, ex)
        deltas.append(new_delta)
    return torch.tensor(deltas)

def improve_beam_score_functions(F,T):
    deltas = []
    for ex in tqdm(F):
        ex = ex.unsqueeze(0)
        _, delta_beam = improve_beam(ex, T, 15, k=5)
        deltas.append(delta_beam.item())
    
    return torch.tensor(deltas)
from math import log2

from tqdm import tqdm
import torch

from src.algebra.linear import evaluate_all_matrix, evaluate_matrix
from src.algebra.function import compute_derivative, compute_delta_mean, filter_functions_delta

def generate_gl_2_matrix(size, batch_size=1_000_000, device='cpu'):
    mat = torch.randint(0, 2, (batch_size, size,size), device=device)
    det = torch.linalg.det(mat.float())
    idx_inv = (det%2)==1
    return mat[idx_inv,:]

def generate_all_bit_function(num_bit):
    if num_bit <= 0:
        return []
    elif num_bit == 1:
        return [[0], [1]]
    else:
        all_functions_prev = generate_all_bit_function(num_bit-1)
        all_functions = []
        for function in all_functions_prev:
            all_functions.append(function + [0])
            all_functions.append(function + [1])
        return all_functions

def generate_basis(T):
    field_size = T.size(0)
    exponent = round(log2(field_size),6)

    M = torch.tensor(generate_all_bit_function(exponent), device=T.device).unsqueeze(0)
    F = evaluate_matrix(M, T)
    _, indices = torch.sort(F)
    return M[0, indices,...]

def generate_linear_polynomial(exponent, T, num_f, batch_size=10_000):
    M_basis = generate_basis(T)
    count = 0
    M_out = []
    while count < num_f:
        M = generate_gl_2_matrix(exponent, device=T.device, batch_size=batch_size)
        F = evaluate_all_matrix(M, M_basis, T)
        count += F.size(0)
        M_out.append(F)
    
    M_out_t = torch.concat(M_out)
    return M_out_t[:num_f,:]

def generate_batch_perm(batch, size):
    return torch.rand(batch, size).argsort (dim = 1)

def draw_sparse_polynomials(field_size, num_coef, num_ech, device='cpu'):
    # Beware, NOT a permutation
    poly = torch.randint(0, field_size-1, (num_ech, field_size), device=device)

    sample = torch.rand((num_ech, field_size), device=device).topk(num_coef, dim=1).indices
    mask = torch.ones(num_ech, field_size, dtype=torch.bool, device=device)
    mask.scatter_(dim=1, index=sample, value=False)
    poly[mask] = field_size-1
    return poly

def generate_permutations_delta(T, target_gen, export=None, device='cpu', threshold=6, exponent=6, batch_p=5_000):
    batch_p = 5_000
    field_size = 2**exponent
    current_num = 0
    out = []
    pbar = tqdm(total=target_gen)
    while current_num < target_gen:
        # Random functions
        F_perm = torch.rand(batch_p, field_size, device=device).argsort (dim = 1)

        DF_perm = compute_derivative(F_perm, T)
        delta = compute_delta_mean(DF_perm)

        filtered_functions = filter_functions_delta(F_perm, delta, threshold)

        current_num += len(filtered_functions)
        pbar.update(len(filtered_functions))
        out.append(filtered_functions)

    output = torch.concat(out)
    if export:
        torch.save(output.cpu(), export)
    return output

def generate_random_permutations_vectorized(size, num_samples, max_k, device='cpu'):
    # Step 1: Create a base identity permutation of shape (num_samples, size)
    random_perms = torch.arange(size, device=device).repeat(num_samples, 1)  # Shape: (num_samples, size)

    # Step 2: Randomly choose the number of elements (k) to permute for each sample
    ks = torch.randint(1, max_k + 1, (num_samples,), device=device)  # Shape: (num_samples,)

    # Step 3: Generate random indices for permuted subsets in parallel
    all_indices = torch.arange(size, device=device).repeat(num_samples, 1)  # Shape: (num_samples, size)
    random_indices = torch.argsort(torch.rand(num_samples, size, device=device), dim=1)  # Randomly permute each row
    
    # Create a mask for each sample to pick the first k indices
    mask = torch.arange(size, device=device).expand(num_samples, size) < ks.unsqueeze(1)  # Mask of shape (num_samples, size)

    # Step 4: Apply the mask to select the subset of indices to permute
    subset_indices = random_indices * mask  # Shape: (num_samples, size)

    # Step 5: Generate random permutations for each subset
    permuted_indices = torch.zeros_like(subset_indices, device=device)
    for i in range(num_samples):
        permuted_indices[i, :ks[i]] = torch.randperm(ks[i], device=device)  # Shuffle the selected subset
        
    # Step 6: Apply the random permutations to the base random_perms tensor
    # We use advanced indexing to apply the permutations
    for i in range(num_samples):
        random_perms[i, subset_indices[i, :ks[i]]] = random_perms[i, subset_indices[i, permuted_indices[i, :ks[i]]]]

    return random_perms
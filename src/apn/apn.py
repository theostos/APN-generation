from math import sqrt, log2, ceil
import json
from functools import cache

from tqdm import tqdm
import torch

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


def compute_delta_all(DF):
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
    return values.max(dim=-1)[0], values.mean(dim=-1)

def compute_delta_spectra(DF):
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
    return values.max(dim=-1)[0], values.mean(dim=-1), torch.sort(values, dim=-1)[0]

def evaluate_matrix(M, T):
    field_size = T.size(0)
    M_mask = (M == 0)
    M_field = torch.arange(0, M.size(2), device=M.device)
    M_field = M_field.expand_as(M).clone()
    M_field[M_mask] = field_size - 1
    return reduce_evaluate_matrix(M_field, T)

def evaluate_all_matrix(M, X, T):
    F_part = evaluate_matrix(M, T)
    F_mask = (X==0)
    F_mask = F_mask.expand(M.size(0), -1, -1)
    X = X.expand(M.size(0), -1, -1)
    F_part = F_part.unsqueeze(1).expand(-1, X.size(1), -1)
    F_tot = F_part * X
    F_tot[F_mask] = T.size(0) - 1
    return reduce_evaluate_matrix(F_tot, T)

def reduce_evaluate_matrix(M, T):
    matrix_size = M.size(2)
    exponent = round(log2(matrix_size),6)
    # Reduce M to final result using the addition table
    for step in range(ceil(exponent)):  # Log2(matrix_size) steps
        stride = 2**step
        indices = torch.arange(0, matrix_size, 2 * stride, device=M.device)
        indices_stride = torch.arange(stride, matrix_size, 2 * stride, device=M.device)

        indices = indices[:len(indices_stride)]
        left = M[..., indices]
        right = M[..., indices_stride]
        # Look up addition results in T
        reduced = T[left, right]
        M[..., indices] = reduced  #
    return M[...,0].clone()

def evaluate_polynomials(P, X, T):
    """
    Evaluate polynomials over a finite field in parallel.

    Args:
        P (torch.Tensor): Tensor of shape (batch_p, field_size), polynomials coefficients.
        X (torch.Tensor): Tensor of shape (batch_input, field_size), powers of inputs.
        T (torch.Tensor): Tensor of shape (field_size, field_size), addition table for the field.
        expect to be indexed with alpha^0 | alpha^1 | ... | alpha^{field_size-2} | 0

    Returns:
        torch.Tensor: Result of shape (batch_p, batch_input).
    """
    field_size = T.size(0)
    batch_p = P.size(0)
    batch_x = X.size(0)
    # Step 1: Compute R[i, j, :] = P[i, :] + X[j, :]
    # Add dimensions to align for broadcasting
    P_expanded = P.unsqueeze(1).expand(batch_p, batch_x, field_size)  # Shape: (batch_p, 1, field_size)
    X_expanded = X.expand(batch_p, batch_x, field_size)  # Shape: (1, batch_input, field_size)
    R = (P_expanded + X_expanded) % (field_size-1) # Shape: (batch_p, batch_input, field_size)
    mask_R = torch.logical_or(P_expanded == field_size-1, X_expanded == field_size-1)
    R[mask_R] = field_size-1
    # to avoid rounding issue in the case of power of 2, we truncate up to 10^-6 (to avoid log2(2**31) = 31 + epsilon), useless with our python's version
    exponent = round(log2(field_size),6)
    # Step 2: Reduce R to final result using the addition table
    for step in range(ceil(exponent)):  # Log2(256) steps
        stride = 2**step
        indices = torch.arange(0, field_size, 2 * stride, device=T.device)
        indices_stride = torch.arange(stride, field_size, 2 * stride, device=T.device)

        indices = indices[:len(indices_stride)]
        left = R[..., indices]
        right = R[..., indices_stride]
        # Look up addition results in T
        reduced = T[left, right]
        R[..., indices] = reduced  # Update with reduced results
    # # Final reduction gives shape (batch_p, batch_input)
    return R[..., 0].clone()

@cache
def power_table(exponent):
    with open(f"power_table_{exponent}", "r") as fp:
        T = json.load(fp)
    T = torch.tensor(T)
    return T

@cache
def interpol_table(exponent):
    with open(f"interpol_table_{exponent}", "r") as fp:
        T = json.load(fp)
    T = torch.tensor(T)
    return T

def interpolate_function(F, T):
    field_size = T.size(0)
    exponent = int(round(log2(field_size),6))
    X = interpol_table(exponent).to(F.device)
    return evaluate_polynomials(F, X, T)

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


def get_filter_function_delta(F, deltas, treshold):
    return F[deltas <= treshold]

def score_functions(F, T):
    DF = compute_derivative(F, T)
    deltas_max = compute_delta_max(DF)
    deltas_mean = compute_delta_mean(DF)
    return deltas_max, deltas_mean

def generate_functions(T, target_gen, export=None, device='cpu', threshold=6, exponent=6, batch_p=5_000):
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

        filtered_functions = get_filter_function_delta(F_perm, delta, threshold)

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
        # DF = compute_derivative(ex, T)
        # deltas = compute_delta_mean(DF)
    return torch.tensor(deltas)

def improve_beam_score_functions(F,T):
    deltas = []
    for ex in tqdm(F):
        ex = ex.unsqueeze(0)
        _, delta_beam = improve_beam(ex, T, 15, k=5)
        deltas.append(delta_beam.item())
    
    return torch.tensor(deltas)

def random_p2(batch, T):
    field_size = T.size(0)
    P = torch.ones(batch, field_size, dtype=torch.long)*field_size
    random_idx = torch.randint(1,field_size, (batch,))
    random_coef_high = torch.randint(0, field_size, (batch,))
    random_coef_low = torch.randint(0, field_size, (batch,))

    P[:, random_idx] = random_coef_high
    P[:, 0] = random_coef_low
    return P

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

def precompose(A, B):
    batch_size, n = A.shape
    # Expand B to get the indices for each batch
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, n)
    # Index A using B
    return A[batch_indices, B]

def check_if_permutation(F):
    F_sort, _ = torch.sort(F)
    field_size = F.size(1)
    return torch.all(F_sort == torch.arange(0, field_size, device=F.device), dim=1)

def _bit_tensor_sum(packed_tensor):
    """Counts the number of 1-bits in a packed int64 tensor using the Hamming weight"""
    count = packed_tensor
    count = (count - ((count >> 1) & 0x5555555555555555))
    count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333)
    count = (count + (count >> 4)) & 0x0F0F0F0F0F0F0F0F
    count = (count * 0x0101010101010101) >> 56
    return torch.sum(count, dim=1).item()

def compute_degrees(P, exponent):
    """
    Given a tensor P of shape (batch_size, n), this function computes, for each
    row, the maximum popcount (number of 1-bits) of the indices i where P[row, i] < 2**exponent-1.
    """
    # P is assumed to be an integer tensor of shape (batch_size, n)
    if P.size(0) == 0:
        return torch.empty(0, dtype=torch.int, device=P.device)
    
    # Create a boolean mask for valid monomials (p < 63)
    valid_mask = P < 2**exponent - 1
    const_bits = torch.tensor([bin(a).count('1') for a in range(2**exponent)], device=P.device)
    # Mask out the invalid ones by setting them to -1 so they are not chosen by max.
    popcounts_valid = torch.where(valid_mask, const_bits, torch.tensor(-1, device=P.device))
    
    # Compute the maximum popcount for each row.
    degrees = popcounts_valid.max(dim=1).values
    return degrees

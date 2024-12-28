from math import log2, ceil
import torch

def _evaluate_matrix(M, T):
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

def evaluate_matrix(M, T):
    field_size = T.size(0)
    M_mask = (M == 0)
    M_field = torch.arange(0, M.size(2), device=M.device)
    M_field = M_field.expand_as(M).clone()
    M_field[M_mask] = field_size - 1
    return _evaluate_matrix(M_field, T)

def evaluate_all_matrix(M, X, T):
    F_part = evaluate_matrix(M, T)
    F_mask = (X==0)
    F_mask = F_mask.expand(M.size(0), -1, -1)
    X = X.expand(M.size(0), -1, -1)
    F_part = F_part.unsqueeze(1).expand(-1, X.size(1), -1)
    F_tot = F_part * X
    F_tot[F_mask] = T.size(0) - 1
    return _evaluate_matrix(F_tot, T)
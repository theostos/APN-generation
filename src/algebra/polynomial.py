
from math import log2, ceil
import json
from functools import cache

import torch

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
def load_power_table(exponent):
    with open(f"table/power/2_{exponent}", "r") as fp:
        T = json.load(fp)
    T = torch.tensor(T)
    return T

@cache
def load_interpolation_table(exponent):
    with open(f"table/interpolation/2_{exponent}", "r") as fp:
        X = json.load(fp)
    X = torch.tensor(X)
    return X

def interpolate_function(F, X, T):
    return evaluate_polynomials(F, X, T)

def compute_degree(P):
    field_size = P.size(1)

    mask = (P != (field_size - 1))
    indices = torch.arange(field_size, device=P.device).expand_as(P)
    masked_indices = mask * indices

    masked_indices[masked_indices == 0] = -1
    T = masked_indices.max(dim=1).values
    return T

def compute_sparsity(P):
    field_size = P.size(1)
    return (P != (field_size - 1)).sum(dim=1)
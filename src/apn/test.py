import json

import torch

from src.apn.apn import *

def _test_evaluate_all_matrix(device='cpu'):
    for exponent in [6, 8]:
        T = add_table(exponent)
        M = generate_gl_2_matrix(exponent, batch_size=10_000)

        assert M.size(0) > 0, "No luck with the gl_2 drawing, please increase batch_size in generate_gl_2_matrix to mitigate this issue"
        X_basis = generate_basis(T)
        F = evaluate_all_matrix(M, X_basis, T)
        F = F.unsqueeze(-1).expand(-1,-1, X_basis.size(-1))

        X_basis = X_basis.expand(F.size(0), -1, -1)
        M_end = torch.gather(X_basis, 1, F[:,:exponent,:])
        assert torch.all(M_end == M)

def _test_reduce_evaluate_matrix(device='cpu'):
    T = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]], device=device)
    M = torch.tensor([[[0, 1, 0], [1, 0, 1], [1, 1, 1]], [[0, 0, 0], [1, 1, 1], [1, 0, 0]]], device=device)
    R = reduce_evaluate_matrix(M, T)
    R_truth = torch.tensor([[1, 2, 0], [0, 0, 1]], device=device)
    assert torch.all(R == R_truth)

def _test_evaluate_polynomials(device='cpu'):
    # T is indexed with alpha^0 | alpha^1 | ... | alpha^{field_size-2} | 0
    # 0 -> 2^0, 1 -> 2^1, 2 -> 0
    T = torch.tensor([[1, 2, 0], [2, 0, 1], [0, 1, 2]], device=device)

    # P_1 = X^2, P_2 = 1, P_3 = 2 + 2X + 2X^2
    P = torch.tensor([[2, 2, 0], [0,2,2], [1, 1, 1]], device=device)

    # first row = [2^0, 0, 0] -> associated to 0
    # second row = [2^0, 2^0, 2^0] -> associated to 2^0
    # third row = [2^0, 2^1, 2^0] -> associated to 2^1

    X = torch.tensor([[0, 2, 2], [0,0,0], [0,1,0]], device=device)
    R = evaluate_polynomials(P, X, T)
    R_truth = torch.tensor([[2, 0, 0], [0, 0, 0], [1, 2, 1]], device=device)

    assert torch.all(R == R_truth)

def _test_compute_delta(device='cpu'):
    F = torch.tensor([[
         [2, 2, 2],
         [1, 0, 1]],

        [[2, 1, 0],
         [1, 0, 2]]], device=device)
    R = compute_delta_max(F)
    R_truth = torch.tensor([3, 1], device=device)
    assert torch.all(R == R_truth)

def _test_compute_derivative(device='cpu'):
    T = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]], device=device)
    F = torch.tensor([[0, 1, 2], [1, 0, 0]], device=device)
    R = compute_derivative(F, T)
    R_exp = torch.tensor([[[0, 2, 1],
         [1, 0, 2]],

        [[2, 0, 0],
         [1, 0, 1]]], device=device)
    assert torch.all(R == R_exp)

def _test_add(device='cpu'):
    T = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]], device=device)
    F0 = torch.tensor([[0, 1, 2], [1,0,0]], device=device)
    F1 = torch.tensor([[0, 1, 2], [0,1,0]], device=device)
    assert torch.all(add(F0, F1, T) == torch.tensor([[0, 2, 1], [1,1,0]], device=device))

def _test_cubic_6_apn(device='cpu'):
    T = add_table(6, device=device)
    with open(f"tables/cube_table_6", "r") as fp:
        F_perm = json.load(fp)
    F_perm = torch.tensor([F_perm], device=device)
    DF_perm = compute_derivative(F_perm, T)
    delta = compute_delta_max(DF_perm)
    assert delta[0]==2, "Cubic should be APN for 2^6"

def _test_interpolation(device='cpu'):
    for exponent in [6, 8]:
        field_size = 2**exponent
        T = add_table(exponent, device=device)
        X = power_table(exponent, device=device)
        I = interpolation_table(exponent, device=device)
        F = torch.randint(0, field_size, (32, field_size), device=device)
        P = interpolate_function(F, T, I)
        F_new = evaluate_polynomials(P, X, T)
        assert torch.all(F==F_new), f"Issue with interpolation for finite field of order 2**{exponent}"

def _test_inv_8_apn(device='cpu'):
    T = add_table(8, device=device)
    with open(f"tables/inv_table_8", "r") as fp:
        F_perm = json.load(fp)
    F_perm = torch.tensor([F_perm], device=device)
    DF_perm = compute_derivative(F_perm, T)
    delta = compute_delta_max(DF_perm)
    assert delta[0]==4, "Inv should be of degree 4 for 2^8"

def _test_generate_basis(device='cpu'):
    for exponent in range(2,9):
        T = add_table(exponent, device=device)
        M = generate_basis(T)
        assert torch.all(M[0, :M.size(2),:] == torch.eye(M.size(2)))

def _test_evaluate_permutation(device='cpu'):
    for exponent in range(2,9):
        T = add_table(exponent, device=device)
        field_size = T.size(0)
        M_basis = generate_basis(T)
        M = generate_gl_2_matrix(exponent, batch_size=10_000)
        F = evaluate_all_matrix(M, M_basis, T)
        values, _ = torch.sort(F)
        assert torch.all(values == torch.arange(0, field_size)), "some functions are not permutation"

def _test_generate_linear_polynomial(device='cpu'):
    for exponent in range(2,9):
        T = add_table(exponent, device=device)
        field_size = T.size(0)
        F = torch.randint(0, field_size, (1_000, field_size))
        F_perm = generate_linear_polynomial(exponent, T, F.size(0))

        F_precompose = precompose(F, F_perm)
        DF = compute_derivative(F, T)
        DF_test = compute_derivative(F_precompose, T)

        delta_prime = compute_delta_max(DF)
        delta_test = compute_delta_max(DF_test)

        assert torch.all(delta_prime==delta_test), "generated polynomials do not preserve delta values"

def _test_walsh_optimal(device='cpu'):
    trace_table = compute_trace_table(6)
    T = add_table(6)
    with open(f"tables/cube_table_6", "r") as fp:
        F = json.load(fp)
    F = torch.tensor([F], device=device)
    DF = compute_derivative(F, T)
    output = compute_walsh_optimal(DF, trace_table)
    assert torch.all(output==2**(6*2+1) * torch.ones(F.shape[0])), "Issue with Walsh optimal"

if __name__ == "__main__":
    _test_walsh_optimal()
    print("PASS walsh_optimal")

    _test_evaluate_all_matrix()
    print("PASS evaluate_all_matrix")

    _test_reduce_evaluate_matrix()
    print("PASS evaluate_matrix")

    _test_evaluate_polynomials()
    print("PASS evaluate_polynomials")

    _test_compute_delta()
    print("PASS compute_delta")

    _test_compute_derivative()
    print("PASS compute_derivative")

    _test_add()
    print("PASS add")

    _test_cubic_6_apn()
    print("PASS Cubic 6")

    _test_interpolation()
    print("PASS Interpolation+Evaluation 6 and 8")

    _test_inv_8_apn()
    print("PASS Inv 8")

    _test_generate_basis()
    print("PASS generate_basis")

    _test_evaluate_permutation()
    print("PASS evaluate_permutation")

    _test_generate_linear_polynomial()
    print('PASS generate_linear_polynomial')
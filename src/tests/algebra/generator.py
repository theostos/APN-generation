import torch

from src.algebra.linear import evaluate_all_matrix
from src.algebra.generator import generate_linear_polynomial, generate_basis, generate_gl_2_matrix
from src.algebra.function import compute_derivative, compute_delta_max, precompose, load_add_table

def _test_generate_linear_polynomial():
    for exponent in range(2,9):
        T = load_add_table(exponent)
        field_size = T.size(0)
        F = torch.randint(0, field_size, (1_000, field_size))
        F_perm = generate_linear_polynomial(exponent, T, F.size(0))

        F_precompose = precompose(F, F_perm)
        DF = compute_derivative(F, T)
        DF_test = compute_derivative(F_precompose, T)

        delta_prime = compute_delta_max(DF)
        delta_test = compute_delta_max(DF_test)

        assert torch.all(delta_prime==delta_test), "generated polynomials do not preserve delta values"
    print('PASS generate_linear_polynomial')

def _test_generate_gl_2_matrix():
    for exponent in range(2,9):
        T = load_add_table(exponent)
        field_size = T.size(0)
        M_basis = generate_basis(T)
        M = generate_gl_2_matrix(exponent, batch_size=10_000)
        F = evaluate_all_matrix(M, M_basis, T)
        values, _ = torch.sort(F)
        assert torch.all(values == torch.arange(0, field_size)), "some functions are not permutation"
    print("PASS generate_gl_2_matrix")

def test_all():
    _test_generate_linear_polynomial()
    _test_generate_gl_2_matrix()
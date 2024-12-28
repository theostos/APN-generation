import torch

from src.algebra.generator import generate_basis, generate_gl_2_matrix
from src.algebra.linear import _evaluate_matrix, evaluate_all_matrix
from src.algebra.function import load_add_table

def _test_evaluate_matrix():
    T = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    M = torch.tensor([[[0, 1, 0], [1, 0, 1], [1, 1, 1]], [[0, 0, 0], [1, 1, 1], [1, 0, 0]]])
    R = _evaluate_matrix(M, T)
    R_truth = torch.tensor([[1, 2, 0], [0, 0, 1]])
    assert torch.all(R == R_truth)
    print("PASS evaluate_matrix")

def _test_evaluate_all_matrix():
    for exponent in [6, 8]:
        T = load_add_table(exponent)
        M = generate_gl_2_matrix(exponent, batch_size=10_000)

        assert M.size(0) > 0, "No luck with the gl_2 drawing, please increase batch_size in generate_gl_2_matrix to mitigate this issue"
        X_basis = generate_basis(T)
        F = evaluate_all_matrix(M, X_basis, T)
        F = F.unsqueeze(-1).expand(-1,-1, X_basis.size(-1))

        X_basis = X_basis.expand(F.size(0), -1, -1)
        M_end = torch.gather(X_basis, 1, F[:,:exponent,:])
        assert torch.all(M_end == M)
    print("PASS evaluate_all_matrix")

def _test_generate_basis():
    for exponent in range(2,9):
        T = load_add_table(exponent)
        M = generate_basis(T)
        assert torch.all(M[0, :M.size(2),:] == torch.eye(M.size(2)))
    print("PASS generate_basis")

def test_all():
    _test_evaluate_matrix()
    _test_evaluate_all_matrix()
    _test_generate_basis()
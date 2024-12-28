import torch

from src.algebra.polynomial import evaluate_polynomials, interpolate_function, load_power_table, load_interpolation_table
from src.algebra.function import load_add_table

def _test_evaluate_polynomials():
    # T is indexed with alpha^0 | alpha^1 | ... | alpha^{field_size-2} | 0
    # 0 -> 2^0, 1 -> 2^1, 2 -> 0
    T = torch.tensor([[1, 2, 0], [2, 0, 1], [0, 1, 2]])

    # P_1 = X^2, P_2 = 1, P_3 = 2 + 2X + 2X^2
    P = torch.tensor([[2, 2, 0], [0,2,2], [1, 1, 1]])

    # first row = [2^0, 0, 0] -> associated to 0
    # second row = [2^0, 2^0, 2^0] -> associated to 2^0
    # third row = [2^0, 2^1, 2^0] -> associated to 2^1

    X = torch.tensor([[0, 2, 2], [0,0,0], [0,1,0]])
    R = evaluate_polynomials(P, X, T)
    R_truth = torch.tensor([[2, 0, 0], [0, 0, 0], [1, 2, 1]])

    assert torch.all(R == R_truth)
    print("PASS evaluate_polynomials")

def _test_interpolation():
    for exponent in [6, 8]:
        field_size = 2**exponent
        T = load_add_table(exponent)
        X = load_power_table(exponent)
        X_interpol = load_interpolation_table(exponent)

        F = torch.randint(0, field_size, (32, field_size))
        P = interpolate_function(F, X_interpol, T)
        F_new = evaluate_polynomials(P, X, T)
        assert torch.all(F==F_new), f"Issue with interpolation for finite field of order 2**{exponent}"
    print("PASS Interpolation+Evaluation 6 and 8")

def test_all():
    _test_evaluate_polynomials()
    _test_interpolation()
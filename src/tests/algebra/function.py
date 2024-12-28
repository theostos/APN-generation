import json

import torch

from src.algebra.function import compute_delta_max, compute_derivative, add_table, load_add_table

def _test_compute_delta():
    F = torch.tensor([[
         [2, 2, 2],
         [1, 0, 1]],

        [[2, 1, 0],
         [1, 0, 2]]])
    R = compute_delta_max(F)
    R_truth = torch.tensor([3, 1])
    assert torch.all(R == R_truth)
    print("PASS compute_delta")

def _test_compute_derivative():
    T = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    F = torch.tensor([[0, 1, 2], [1, 0, 0]])
    R = compute_derivative(F, T)
    R_exp = torch.tensor([[[0, 2, 1],
         [1, 0, 2]],

        [[2, 0, 0],
         [1, 0, 1]]])
    assert torch.all(R == R_exp)
    print("PASS compute_derivative")

def _test_add_table():
    T = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    F0 = torch.tensor([[0, 1, 2], [1,0,0]])
    F1 = torch.tensor([[0, 1, 2], [0,1,0]])
    assert torch.all(add_table(F0, F1, T) == torch.tensor([[0, 2, 1], [1,1,0]]))
    print("PASS add_table")

def _test_cubic_6_apn():
    T = load_add_table(6)
    with open(f"table/test/cubic_6", "r") as fp:
        F_perm = json.load(fp)
    F_perm = torch.tensor([F_perm])
    DF_perm = compute_derivative(F_perm, T)
    delta = compute_delta_max(DF_perm)
    assert delta[0]==2, "Cubic should be APN for 2^6"
    print("PASS cubic_6")

def _test_inv_8_apn():
    T = load_add_table(8)
    with open(f"table/test/inv_8", "r") as fp:
        F_perm = json.load(fp)
    F_perm = torch.tensor([F_perm])
    DF_perm = compute_derivative(F_perm, T)
    delta = compute_delta_max(DF_perm)
    assert delta[0]==4, "Inv should be of degree 4 for 2^8"
    print("PASS inv_8")

def test_all():
    _test_compute_derivative()
    _test_compute_delta()
    _test_add_table()
    _test_cubic_6_apn()
    _test_inv_8_apn()
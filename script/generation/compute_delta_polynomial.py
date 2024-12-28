import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.algebra.polynomial import evaluate_polynomials, load_power_table, compute_degree
from src.algebra.function import load_add_table, check_if_permutation, compute_derivative, compute_delta_max

exponent = 6
field_size = 2**exponent
device = 'cpu'
in_path = f'data/export_0.pt'
batch_size = 1_000

P = torch.load(in_path, weights_only=True)

T = load_add_table(exponent)
X = load_power_table(exponent)

for sub_P in torch.split(P, batch_size):
    F = evaluate_polynomials(sub_P, X, T)
    res = check_if_permutation(F)

    DF = compute_derivative(F, T)
    delta = compute_delta_max(DF)
    print(delta)
    # print(res)
    print(compute_degree(sub_P))
    print((sub_P != (field_size - 1)).sum(dim=1))
    # exit()
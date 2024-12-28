import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tqdm import tqdm
import torch

from src.algebra.generator import draw_sparse_polynomials
from src.algebra.polynomial import evaluate_polynomials, load_power_table
from src.algebra.function import check_if_permutation, compute_derivative, compute_delta_max, load_add_table

exponent = 6
field_size = 2**exponent
device = 'cpu'
num_coef = 4
export_path = f'export/sparse_polynomial_{num_coef}_{device}'

T = load_add_table(exponent)
X = load_power_table(exponent)

list_P = []
for _ in tqdm(range(150_000)):
    P = draw_sparse_polynomials(field_size, num_coef, 200_000, device=device)
    F = evaluate_polynomials(P, X, T)
    res = check_if_permutation(F)
    F = F[res,:]
    P = P[res,:]
    if F.size(0) > 0:
        print(P)
        DF = compute_derivative(F, T)
        delta = compute_delta_max(DF)
        sub_P_idx = delta <= 4
        sub_P = P[sub_P_idx,:].clone()
        list_P.append(sub_P.to('cpu'))

result = torch.concat(list_P)
torch.save(result, export_path)
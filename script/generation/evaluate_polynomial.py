import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tqdm import tqdm
import torch

from src.algebra.polynomial import evaluate_polynomials, load_power_table
from src.algebra.function import load_add_table

exponent = 6
field_size = 2**exponent
device = 'cpu'
num_coef = 4
out_path = f'export/file_out.pt'
in_path = f'data/file_in.pt'

T = load_add_table(exponent)
X = load_power_table(exponent)

all_P = torch.load(in_path, weights_only=True)
P = all_P.to(device)
F_list = []
for P_sub in tqdm(torch.split(P, 1000)):
    F = evaluate_polynomials(P_sub, X, T)
    F_list.append(F)

F = torch.concat(F_list)

torch.save((P, F), out_path)
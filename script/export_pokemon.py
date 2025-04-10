import argparse
import json

import torch
from tqdm import tqdm

from apn import improve_beam_score_functions, score_functions, evaluate_polynomials, check_if_permutation, compute_degrees

from model import generate, ModelConfig, Transformer


def evaluate_delta(model, X, T, device, batch_size=8192, bar=None):
    field_size = T.size(0)
    idx = torch.ones((batch_size, 1), dtype=torch.long, device=device) * field_size

    P = generate(model, idx, field_size, do_sample=True)
    P = torch.clip(P, max=T.size(0)-1)

    degrees = compute_degrees(P, 6)
    P = P[degrees > 2,:]
    F = evaluate_polynomials(P, X, T)
    deltas_max, deltas_mean = score_functions(F, T)
    idx_sol = deltas_mean < 2.85
    P = P[idx_sol,:]

    if P.size(0) > 0:  
        with open('test.txt', 'a') as file:
            json.dump(P.tolist(), file)
            file.write('\n')
        if bar is not None:
            bar.update(P.size(0))

device = 'cuda:1'
# dataset = torch.load('export_tot.pt')[:30]
# print(dataset)
# exit()
# degrees = compute_degrees(dataset, 6)
# degrees = degrees.float()


with open(f"add_table_6", "r") as fp:
    T = json.load(fp)
T = torch.tensor(T, device=device)

with open(f"power_table_6", "r") as fp:
    X_pow = json.load(fp)

X_pow = torch.tensor(X_pow, device=device)

result = []
with open('test.txt', 'r') as file:
    for line in file.readlines():
        result += json.loads(line)

result = torch.load('export_sparse_NONperm_1_8_deg3_4_0.pt')[:30].tolist()
for p in result:
    for k, coef in enumerate(p):
        if coef != 63:
            print(f"\\alpha^{coef} X^{k} + ", end='')
    print()
P = torch.tensor(result, device=device)
degrees = compute_degrees(P, 6)
print(degrees)
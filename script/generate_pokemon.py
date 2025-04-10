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

parser = argparse.ArgumentParser(description="Make More")
# system/input/output
# parser.add_argument('--input-file', '-i', type=str, default='export_eval_P.pt', help="input file with things one per line")
parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
parser.add_argument('--device', type=str, default='cuda:0', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
parser.add_argument('--seed', type=int, default=3407, help="seed")
# sampling
parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
parser.add_argument('--n-layer', type=int, default=6, help="number of layers")
parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
parser.add_argument('--n-embd', type=int, default=128, help="number of feature channels in the model")
# optimization
parser.add_argument('--batch-size', '-b', type=int, default=1024, help="batch size during optimization")
parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
args = parser.parse_args()
exponent = 6
vocab_size = 2**exponent + 2
block_size = 130
config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd)
PATH = "out/model_17500.pt"
device = "cuda:0"
model_base = Transformer(config)

state_dict = torch.load(PATH, map_location=device)

# Remove the 'module.' prefix
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model_base.load_state_dict(new_state_dict)
model_base = model_base.to(device)
model_base.half()
model_base.eval()

with open(f"add_table_6", "r") as fp:
    T = json.load(fp)
T = torch.tensor(T, device=device)

with open(f"power_table_6", "r") as fp:
    X_pow = json.load(fp)

X_pow = torch.tensor(X_pow, device=device)
bar = tqdm(desc='Pokémon', position=1)
with torch.no_grad():
    for _ in tqdm(range(10_000), position=0):
        evaluate_delta(model_base, X_pow, T, device, bar=bar)
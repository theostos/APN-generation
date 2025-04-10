import json

from tqdm import tqdm
import torch

from src.apn.apn import *


if __name__ == "__main__":
    import random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-device', default=1, help='GPU index')
    parse = parser.parse_args()
    exponent=6
    field_size = 2**exponent
    num_device = parse.num_device
    device = f'cuda:{num_device}'
    with open(f"power_table_{exponent}", "r") as fp:
        X = json.load(fp)
    X = torch.tensor(X).to(device)
    with open(f"add_table_{exponent}", "r") as fp:
        T = json.load(fp)
    T = torch.tensor(T).to(device)

    num_gen = 100_000
    for sub in range(8, 100):
        progress_bar = tqdm(total=num_gen)
        total_size = 0
        list_P = []
        while total_size < num_gen:
            k = random.randint(3,6)

            P = torch.load('export_sparse_NONperm_1_8_deg3_4_0.pt').to(device)
            P = P[0,:].unsqueeze(0)
            F = evaluate_polynomials(P, X, T)
            DF = compute_derivative(F, T)

            spectrum_set = set()
            delta_max_or, delta_mean_or, spectrum_or = compute_delta_spectra(DF)
            print(delta_mean_or, delta_max_or)
            spectrum_set.add(str(spectrum_or[0,:].tolist()))
            while True:
                for k in range(63):
                    eps = torch.randint(0, 2, (40_000, 64), device=device)
                    new_eps = torch.where(eps==1, k, 63)

                    F_perturbate = add_table(F, new_eps, T)
                    DF = compute_derivative(F_perturbate, T)
                    delta_max, delta_mean, spectrum = compute_delta_spectra(DF)

                    idx_delta_mean = torch.argmin(delta_mean)
                    
                    if torch.min(delta_mean) <= delta_mean_or:
                        delta_max = delta_max[idx_delta_mean]
                        delta_mean = delta_mean[idx_delta_mean]
                        F = F_perturbate[idx_delta_mean,:].unsqueeze(0)

                        opt_perturbate = new_eps[idx_delta_mean,:].unsqueeze(0)
                        eps_pol = interpolate_function(opt_perturbate, T)
                        degrees = compute_degrees(eps_pol, 6)
                        poly = ""
                        for deg, coef in enumerate(eps_pol[0,:].tolist()):
                            if coef != 63:
                                poly += f"\\alpha^{coef} X^{deg} +"
                        poly = poly[:-2]
                        print(poly)
                        print(degrees.item(), delta_mean.item(), delta_max.item())
                        spectrum = spectrum[idx_delta_mean, :]
                        hash_sp = str(spectrum.tolist())
                        if hash_sp in spectrum_set:
                            print("ALED")
                        else:
                            spectrum_set.add(hash_sp)
                        delta_mean_or = delta_mean
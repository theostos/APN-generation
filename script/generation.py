import argparse
import random

from tqdm import tqdm

from src.apn.apn import *


def local_search_delta_mean(P, X, T, max_steps=32, batch_perturbation=2_000):
    F = evaluate_polynomials(P, X, T)
    DF = compute_derivative(F, T)
    
    _, delta_mean, _ = compute_delta_spectra(DF)
    delta_mean_temp = delta_mean[0]
    acc = 0
    while True:
        k = random.randint(0, 63)
        P_eps = torch.randint(0, 2, (batch_perturbation, 64), device=device)
        P_new_eps = torch.where(P_eps==1, k, 63)

        F_eps = evaluate_polynomials(P_new_eps, X, T)
        F_perturbate = add(F, F_eps, T)

        DF = compute_derivative(F_perturbate, T)
        _, delta_mean, _ = compute_delta_spectra(DF)
        idx_delta_mean = torch.argmin(delta_mean)
        if delta_mean[idx_delta_mean] < delta_mean_temp:
            delta_mean_temp = delta_mean[idx_delta_mean]
            F = F_perturbate[idx_delta_mean,:].unsqueeze(0)
            acc = 0
        else:
            acc += 1
        
        if acc > max_steps:
            return F, delta_mean_temp

def step_fun(acc, max_steps):
    if acc < int(max_steps*0.8):
        return 1
    dist = int(32*(acc-max_steps*0.8)/(0.2*max_steps))
    return dist

def local_search_spectrum(P, X, T, max_steps=20, batch_perturbation=1_000):
    target = torch.zeros(T.size(0), device=P.device)
    target[-1] = 2016
    target[-2] = 2016

    F = evaluate_polynomials(P, X, T)
    DF = compute_derivative(F, T)
    delta_table = compute_delta_table(DF).reshape(1, 63, 64)
    twisted_delta = compute_delta_twisted_table(delta_table)
    distance_old = (torch.sqrt((twisted_delta - target)**2)).sum(dim=1)[0]
    acc = 0
    while True:
        k = step_fun(acc, max_steps)
        F_new_eps = draw_sparse_polynomials(64, k, batch_perturbation, device=device)
        F_perturbate = add(F, F_new_eps, T)
            
        DF = compute_derivative(F_perturbate, T)
        
        delta_table = compute_delta_table(DF).reshape(F_perturbate.size(0), 63, 64)
        twisted_delta = compute_delta_twisted_table(delta_table)
        distances = (torch.sqrt((twisted_delta - target)**2)).sum(dim=1)
        indice = torch.argmin(distances)
        if distances[indice] < distance_old:
            F = F_perturbate[indice,:].unsqueeze(0)
            distance_old = distances[indice]
            acc = 0
        else:
            acc += 1
        
        if acc > max_steps:
            return F, distance_old

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parse = parser.parse_args()

    exponent=6
    field_size = 2**exponent
    num_gen = 700_000

    device = parse.device
    T = add_table(exponent, device=device)
    X = power_table(exponent, device=device)
    I = interpolation_table(exponent, device=device)

    progress_bar = tqdm(total=num_gen)
    total_size = 0
    list_P = []
    min_distance = float('inf')
    for k in tqdm(range(num_gen)):
        P = draw_sparse_polynomials(field_size, 63, 1, device=device)
        # F,_ = local_search_delta_mean(P, X, T)
        # P = interpolate_function(F, T, I)
        F , distance = local_search_spectrum(P, X, T)
        P = interpolate_function(F, T, I)
        list_P.append(P.unsqueeze(0).to('cpu'))

        if len(list_P) % 50_000 == 0:
            torch.save(torch.cat(list_P, dim=0), f'training_{device}_{k}.pt')
            list_P = []
    torch.save(torch.cat(list_P, dim=0), f'training_{device}_final.pt')
        # # print(distance)
        # if distance < min_distance:
        #     min_distance = distance
        #     print(distance)
        # for k in range(63):
        #     eps = torch.randint(0, 2, (40_000, 64), device=device)
        #     new_eps = torch.where(eps==1, k, 63)
        #     new_eps = evaluate_polynomials(new_eps, X, T)

        #     F_perturbate = add(F, new_eps, T)

        #     DF = compute_derivative(F_perturbate, T)
        #     twisted_delta = compute_delta_twisted_table(DF)//2
        #     delta_max, delta_mean, spectrum = compute_delta_spectra(DF)
        #     candidates, indices = lexicographical_sort(twisted_delta)
            # spectrum_set.add(str(spectrum_or[0,:].tolist()))
            # while True:
            #     for k in range(63):
            #         eps = torch.randint(0, 2, (40_000, 64), device=device)
            #         new_eps = torch.where(eps==1, k, 63)

            #         F_perturbate = add_table(F, new_eps, T)
            #         DF = compute_derivative(F_perturbate, T)
            #         delta_max, delta_mean, spectrum = compute_delta_spectra(DF)

            #         idx_delta_mean = torch.argmin(delta_mean)
                    
            #         if torch.min(delta_mean) <= delta_mean_or:
            #             delta_max = delta_max[idx_delta_mean]
            #             delta_mean = delta_mean[idx_delta_mean]
            #             F = F_perturbate[idx_delta_mean,:].unsqueeze(0)

            #             opt_perturbate = new_eps[idx_delta_mean,:].unsqueeze(0)
            #             eps_pol = interpolate_function(opt_perturbate, T)
            #             degrees = compute_degrees(eps_pol, 6)
            #             poly = ""
            #             for deg, coef in enumerate(eps_pol[0,:].tolist()):
            #                 if coef != 63:
            #                     poly += f"\\alpha^{coef} X^{deg} +"
            #             poly = poly[:-2]
            #             print(poly)
            #             print(degrees.item(), delta_mean.item(), delta_max.item())
            #             spectrum = spectrum[idx_delta_mean, :]
            #             hash_sp = str(spectrum.tolist())
            #             if hash_sp in spectrum_set:
            #                 print("ALED")
            #             else:
            #                 spectrum_set.add(hash_sp)
            #             delta_mean_or = delta_mean
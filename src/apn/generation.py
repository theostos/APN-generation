import json

from tqdm import tqdm
import torch


# ds_training = torch.load('export_tot.pt')
# num_device = 2
# device = f'cuda:{num_device}'
# exponent=6
# field_size = 2**exponent

# with open(f"add_table_{exponent}", "r") as fp:
#     T = json.load(fp)
# T = torch.tensor(T).to(device)
# with open(f"power_table_{exponent}", "r") as fp:
#     X = json.load(fp)
# X = torch.tensor(X).to(device)

# P = ds_training.to(device)
# F = evaluate_polynomials(P, X, T)
# res = check_if_permutation(F)
# F = F[res,:]
# P = P[res,:]
# if F.size(0) > 0:
#     DF = compute_derivative(F, T)
#     delta = compute_delta_max(DF)
#     sub_P_idx = delta <= 4
#     sub_P = P[sub_P_idx,:].clone()

# list_deg = []
# for p in P:
#     non_zero = (p < 63).nonzero()
#     non_zero = non_zero.flatten().tolist()
#     deg = max([bin(a).count('1') for a in non_zero])

#     list_deg.append(deg)

# print(torch.tensor(list_deg).float().mean())
# print(torch.tensor(list_deg).float().max())
# print(torch.tensor(list_deg).float().min())
# print(torch.tensor(list_deg).float().std())

# exit()

# export_test_1 = torch.load('export_1.pt')
# export_test_0 = torch.load('export_0.pt')
# all_P = torch.concat([export_test_0, export_test_1])

# exponent = 6
# num_device = 2
# device = f'cuda:{num_device}'
# with open(f"add_table_{exponent}", "r") as fp:
#     T = json.load(fp)
# T = torch.tensor(T).to(device)
# with open(f"power_table_{exponent}", "r") as fp:
#     X = json.load(fp)
# X = torch.tensor(X).to(device)
# field_size = T.size(0)

# P = all_P.to(device)
# F_list = []
# for P_sub in tqdm(torch.split(P, 1000)):
#     F = evaluate_polynomials(P_sub, X, T)
#     F_list.append(F)

# F = torch.concat(F_list)

# torch.save((P, F), 'export_sparse_P_eval.pt')
# # # exit()
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

    # import matplotlib.pyplot as plt

    # P = draw_sparse_polynomials(field_size, 5, 20_000, device=device)
    # F = evaluate_polynomials(P, X, T)
    # DF = compute_derivative(F, T)
    # delta_max, delta_mean = compute_delta_all(DF)

    # plt.scatter(delta_mean.cpu().numpy(), delta_max.cpu().numpy())
    # plt.show()
    # exit()
    num_gen = 100_000
    for sub in range(8, 100):
        progress_bar = tqdm(total=num_gen)
        total_size = 0
        list_P = []
        while total_size < num_gen:
            k = random.randint(3,6)
            # P = draw_sparse_polynomials(field_size, k, 20_000, device=device)

            # for i in range(6):
            #     for j in range(i):
            #         P[:,2**i + 2**j] = field_size-1

            # F = evaluate_polynomials(P, X, T)
            # res = check_if_permutation(F)
            # F = F[res,:]
            # P = P[res,:]
            # F = torch.randint(0, 64, (1, 64), device=device)
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
                        # print(opt_perturbate.shape)
                        eps_pol = interpolate_function(opt_perturbate, T)
                        # print(eps_pol.shape)
                        # exit()
                        # print(P_perturb)
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
            
            while True:
                
                local_improvement(T,F)
            exit()
            if F.size(0) > 0:
                DF = compute_derivative(F, T)
                delta_max, delta_mean = compute_delta_all(DF)

                sub_P_idx = torch.logical_and(delta_max <= 4, delta_mean <= 2.875)
                sub_P = P[sub_P_idx,:].clone()

                degrees = compute_degrees(sub_P, exponent)
                idx_degrees = torch.logical_and(degrees < 6, 2 < degrees)
                sub_P = sub_P[idx_degrees,:]
                if sub_P.size(0) > 0:
                    progress_bar.update(sub_P.size(0))
                    total_size += sub_P.size(0)
                    list_P.append(sub_P)
            if total_size % 1500 == 1499:
                chunk = total_size // 1500
                tensor_P = torch.concat(list_P, dim=0)
                torch.save(tensor_P, f'export_sparse_NONperm_{num_device}_{sub}_deg3_4_{chunk}.pt')
                list_P = []
                total_size = 0
        
        
    exit()
# delta_max_list = []
# # for k in tqdm(range(2,65)):
# list_P = []
# for _ in tqdm(range(150_000)):
#     k = 4
#     P = draw_sparse_polynomials(field_size, k, 200_000, device=device)
#     F = evaluate_polynomials(P, X, T)
#     res = check_if_permutation(F)
#     F = F[res,:]
#     P = P[res,:]
#     if F.size(0) > 0:
#         DF = compute_derivative(F, T)
#         delta = compute_delta_max(DF)
#         sub_P_idx = delta <= 4
#         sub_P = P[sub_P_idx,:].clone()
#         list_P.append(sub_P.to('cpu'))
    
# result = torch.concat(list_P)
# torch.save(result, f'export_{num_device}.pt')
# exit()
#     # DF = compute_derivative(F, T)
#     # delta_max = compute_delta_max(DF)
#     # delta_max_list.append(delta_max.to('cpu'))

# # delta_max_list = []
# # for k in tqdm(range(2,65)):
# #     P = draw_sparse_polynomials(field_size, k, 10_000)
# #     F = evaluate_polynomials(P, X, T)
# #     DF = compute_derivative(F, T)
# #     delta_max = compute_delta_max(DF)
# #     delta_max_list.append(delta_max)

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter

# # Create the animation
# fig, ax = plt.subplots()

# # Initialize histogram
# n_bins = 4
# ax.set_xlim(min(delta_max_list[0]), max(delta_max_list[-1]))
# ax.set_ylim(0, 1000)  # Adjust based on expected frequency
# ax.set_xlabel("Delta Max")
# ax.set_ylabel("Frequency")

# # Update function for animation
# def update(frame):
#     ax.clear()
#     ax.hist(delta_max_list[frame], bins=n_bins, color='blue', alpha=0.7)
#     ax.set_title(f"Histogram of delta_max (sparse polynomial with {frame+1} non zero coefficients)")
#     ax.set_xlim(min(delta_max_list[-1]), max(delta_max_list[0]))
#     ax.set_ylim(0, 1000)  # Adjust based on expected frequency
#     ax.set_xlabel("Delta Max")
#     ax.set_ylabel("Frequency")

# # Create animation
# anim = FuncAnimation(fig, update, frames=len(delta_max_list), interval=200)

# # Export as GIF
# anim.save("delta_max_histogram.gif", writer=PillowWriter(fps=5))


# exit()
# # Count occurrences of each token (0 to 63)
# dataset = dataset.flatten()
# token_counts = torch.bincount(dataset, minlength=64)

# # Calculate probabilities
# total_tokens = token_counts.sum()
# probabilities = token_counts / total_tokens

# # Avoid log(0) by masking zero probabilities
# non_zero_probs = probabilities[probabilities > 0]

# # Compute entropy
# entropy = -torch.sum(non_zero_probs * torch.log2(non_zero_probs))
# print(entropy)
# exit()
# exponent = 6
# with open(f"add_table_{exponent}", "r") as fp:
#     T = json.load(fp)
# T = torch.tensor(T).to("cuda:0")
# field_size = T.size(0)
# list_P = []

# F = torch.load('export_tot_4_P_augmented_x10.pt', weights_only=True)
# for k in tqdm(range(10)):

#     F_perm = generate_linear_polynomial(exponent, T, F.size(0))

#     F_precompose = precompose(F, F_perm)


#     for F_part in torch.tensor_split(F_precompose, 100):
#         P = interpolate_function(F_part, T)
#         list_P.append(P.to('cpu'))

# tensor_P = torch.concat(list_P, dim=0)
# # print(tensor_P.shape)
# torch.save(tensor_P, 'export_tot_4_P_augmented_x10.pt')
# exit()
# exponent=6
# device='cuda:0'
# with open(f"add_table_{exponent}", "r") as fp:
#     T = json.load(fp)
# T = torch.tensor(T).to(device)
# print(random_p2(3, T))
# exit()

# list_P = []
# for F_part in tqdm(torch.tensor_split(F, 100)):
#     P = interpolate_function(F_part, T)
#     list_P.append(P)

# tensor_P = torch.concat(list_P, dim=0)
# sub_P = tensor_P[:100,:]
# sub_F = evaluate_polynomials(sub_P, X, T)

# print(torch.all(sub_F==F[:100,:]))
# exit()
# torch.save(tensor_P, 'export_tot_4_P.pt')
# exit()
# exponent = 6
# with open(f"add_table_{exponent}", "r") as fp:
#     T= json.load(fp)
# T = torch.tensor(T, device=device)

# filenames = []
# filetype_bis = 'export_delta_4_bis_{k}.pt'
# filetype_ter = 'export_delta_4_ter_{k}.pt'

# for k in range(3):
#     filenames.append(filetype_bis.format(k=k))
#     filenames.append(filetype_ter.format(k=k))

# list_t = []
# for filename in filenames:
#     list_t.append(torch.load(filename, weights_only=True).to(device))

# for F in list_t:
#     F = F.to(device)
#     DF = compute_derivative(F, T)
#     delta_max = compute_delta_max(DF)
#     print(delta_max.max())

# tensor = torch.concat(list_t)
# torch.save(tensor, 'export_tot_4.pt')
# exit()
# list_t = []
# for filename in filenames:
#     list_t.append(torch.load(filename + '_curated.pt', weights_only=True))

# tensor = torch.concat(list_t)
# torch.save(tensor, 'export_tot_curated.pt')

# tensor = torch.load('export_tot_curated_improve.pt')
# tensor2 = torch.load('export_tot_curated.pt')

# print(tensor[0,:])
# print(tensor2[0,:])
# exit()
# if __name__ == '__main__':
#     exponent = 6
#     field_size = 2**6
#     device = 'cuda:0'
#     # F = torch.load('export_tot_4.pt')

#     with open(f"add_table_{exponent}", "r") as fp:
#         T = json.load(fp)
#     T = torch.tensor(T).to(device)

#     with open(f"power_table_{exponent}", "r") as fp:
#         X = json.load(fp)
#     X = torch.tensor(X).to(device)
#     list_P = []
#     list_F = []
#     for k in tqdm(range(1, 4)):
#         for _ in range(100):
#             P = draw_sparse_polynomials(field_size, k, 256*100, device=device)
#             # P = torch.randint(0, field_size, (32_000, field_size)).to(device)
#             F = evaluate_polynomials(P, X, T)
#             list_P.append(P.to('cpu'))
#             list_F.append(F.to('cpu'))
#     tensor_P = torch.concat(list_P, dim=0)
#     tensor_F = torch.concat(list_F, dim=0)

#     # print(torch.all(tensor_P[:,0] == tensor_F[:,-1]))
#     torch.save((tensor_P, tensor_F), 'export_eval_P.pt')
#     exit()
#     num_device = 0
#     BATCH_SIZE = 10_000
#     exponent=6
#     field_size = 2**exponent
#     filename = 'export_5'
#     device = f'cuda:{num_device}'
#     with open(f"add_table_{exponent}", "r") as fp:
#         T= json.load(fp)
#     T = torch.tensor(T, device=device)
    
#     F_perm = torch.rand(1_000, field_size, device=device).argsort (dim = 1)
#     # F_perm = torch.load('export_tot_curated.pt', weights_only=True).to(device)

#     new_out = []
#     deltas = []
#     delta_mean_hist = []
#     delta_max_hist = []

#     delta_mean_hist_improve = []
#     delta_max_hist_improve = []

#     delta_mean_hist_beam = []
#     delta_max_hist_beam = []

#     for ex in tqdm(F_perm):
#         ex = ex.unsqueeze(0)
#         DF = compute_derivative(ex, T)
#         delta_max = compute_delta_max(DF)[0].item()
#         delta_mean = compute_delta_mean(DF)[0].item()

#         delta_max_hist.append(delta_max)
#         delta_mean_hist.append(delta_mean)

#         F_new_beam, delta_mean_beam = improve_beam(ex, T, 15, k=5)
#         delta_mean_beam = delta_mean_beam[0].item()
    
#         DF = compute_derivative(F_new_beam, T)
#         delta_max_beam = compute_delta_max(DF)[0].item()

#         delta_max_hist_beam.append(delta_max_beam)
#         delta_mean_hist_beam.append(delta_mean_beam)

#         F_improve, delta_mean_improve = local_improvement(T, ex)
#         DF = compute_derivative(F_improve, T)
#         delta_max_improve = compute_delta_max(DF)[0].item()

#         delta_max_hist_improve.append(delta_max_improve)
#         delta_mean_hist_improve.append(delta_mean_improve)
    
#     # deltas = torch.tensor(deltas)
#     # print(torch.mean(deltas))
#         # F_new,delta = local_improvement(T, ex)

        
#         # if delta_max.item() <= 4:
#         #     new_out.append(F_new_beam)
#         # print(delta_beam)
#         # print(delta)
#         # print()
#     # max_bin = min(field_size, 20)

#     plt.hist(delta_mean_hist, density=True, color='skyblue', edgecolor='black')
#     plt.xlabel('Delta')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of delta mean values for random functions')

#     # Display the plot
#     plt.savefig(f"hist_mean.png")
#     plt.clf()

#     plt.hist(delta_max_hist, density=True, color='skyblue', edgecolor='black')
#     plt.xlabel('Delta')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of delta max values for random functions')
#     # Display the plot
#     plt.savefig(f"hist_max.png")
#     plt.clf()


#     plt.hist(delta_mean_hist_beam, density=True, color='skyblue', edgecolor='black')
#     plt.xlabel('Delta')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of delta mean values for beam improved functions')

#     # Display the plot
#     plt.savefig(f"hist_mean_beam.png")
#     plt.clf()

#     plt.hist(delta_max_hist_beam, density=True, color='skyblue', edgecolor='black')
#     plt.xlabel('Delta')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of delta max values for beam improved functions')
#     # Display the plot
#     plt.savefig(f"hist_max_beam.png")
#     plt.clf()


#     plt.hist(delta_mean_hist_improve, density=True, color='skyblue', edgecolor='black')
#     plt.xlabel('Delta')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of delta mean values for improved functions')

#     plt.savefig(f"hist_mean_improved.png")
#     plt.clf()

#     plt.hist(delta_max_hist_improve, density=True, color='skyblue', edgecolor='black')
#     plt.xlabel('Delta')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of delta max values for improved functions')

#     plt.savefig(f"hist_max_improved.png")
#     exit()

#     # new_out = torch.concat(new_out)
#     # print(new_out.shape)
#     # torch.save(new_out, f'export_delta_4_bis_bis_{num_device}.pt')
#     # exit()
#     # F_perm = torch.load('export_3_curated.pt', weights_only=True).to(device)
#     # F_perm = torch.load('export_tot_curated.pt', weights_only=True).to(device)
#     # new_out = []
#     # k = 0
#     # for ex in tqdm(F_perm[:5_000]):
#     #     ex = ex.unsqueeze(0)
#     #     DF = compute_derivative(ex, T)
#     #     # delta_mean = compute_delta_mean(DF)
#     #     delta_max = compute_delta_max(DF)
#     #     F_new,_ = local_improvement(T, ex)

#     #     DF_new = compute_derivative(F_new, T)
#     #     delta_max_new = compute_delta_max(DF_new)
#     #     if delta_max_new.item() <= 4:
#     #         new_out.append(F_new)
#     #         k += 1
    
#     # print(k)
#     # torch.save(torch.concat(new_out), 'export_tot_curated_improve_rec.pt')
#         # print(delta_max, delta_max_new)
    
#     # examples = torch.load('export_tot_curated.pt', weights_only=True).to(device)
#     # new_out = []
#     # for ex in tqdm(examples):
#     #     new_F = local_improvement(T, ex.unsqueeze(0))
#     #     new_out.append(new_F)
    
#     # torch.save(torch.concat(new_out), 'export_tot_curated_improve.pt')
#     # exit()
# #     tensor = torch.load(filename + '.pt', weights_only=True).to(device)
# #     list_t = torch.tensor_split(tensor, BATCH_SIZE)
# #     deltas = []
# #     for F in tqdm(list_t):
# #         DF = compute_derivative(F, T)
# #         delta_mean = compute_delta_mean(DF)
# #         deltas.append(delta_mean)

# #     deltas = torch.concat(deltas)
# #     deltas = (deltas*10).int()
# #     filtered_functions = get_filter_function_delta(tensor, deltas, 46).cpu()
# #     print(filtered_functions.shape)
# #     torch.save(filtered_functions, filename+'_curated.pt')
# #     pass


# # exit()
# # # print(torch.initial_seed())
# # result = result.cpu()
# # print(result)
# # exit()
# # max_bin = min(field_size, 20)
# # plt.hist(list(range(max_bin)), bins=(list(range(max_bin+1))), weights=result[:max_bin], density=True, color='skyblue', edgecolor='black')
# # print(result[:max_bin]/result.sum()*100)

# # # plt.stairs(result.cpu()[:9])

# # # Adding labels and title
# # plt.xlabel('Delta')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of delta values')
 
# # # Display the plot
# # plt.savefig(f"hist_{field_size}.png")
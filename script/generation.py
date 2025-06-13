import argparse

from tqdm import tqdm

from src.apn.apn import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', default=10, type=int)
    args = parser.parse_args()

    T = add_table(6, device='cuda:0')
    trace_table = compute_trace_table(6, device='cuda:0')

    list_P = []
    count = 0

    with tqdm() as pbar:
        while True:
            F = torch.randint(0, 64, (args.batch, 64), device='cuda:0')
            gradient_descent(F, T, trace_table)
            list_P.append(F)
            count += args.batch
            pbar.update(args.batch)
            if len(list_P) % 500 == 0:
                torch.save(torch.cat(list_P, dim=0), f'training_{args.device}_{count}.pt')
                list_P = []

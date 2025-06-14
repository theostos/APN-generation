import argparse
import concurrent.futures

from tqdm import tqdm

from src.apn.apn import *

def make(batch_size, device):
    T = add_table(6, device=args.device)
    trace_table = compute_trace_table(6, device=args.device)
    while True:
        F = torch.randint(0, 64, (batch_size, 64), device=device)
        gradient_descent(F, T, trace_table)
        list_P.append(F)
        count += args.batch
        if len(list_P) % 20 == 0:
            torch.save(torch.cat(list_P, dim=0), f'training_{device}_{count}.pt')
            list_P = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-device', default=0, type=int)
    parser.add_argument('--batch-size', default=10, type=int)

    args = parser.parse_args()

    list_P = []
    count = 0
    to_do = [f'cuda:{k}' for k in range(args.num_device)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_device) as executor:
        futures = []
        for device in enumerate(to_do):
            futures.append(executor.submit(make, device, args.batch_size))
        for _ in tqdm(concurrent.futures.as_completed(futures), desc="Overall progress", position=0, total=len(futures)):
            pass

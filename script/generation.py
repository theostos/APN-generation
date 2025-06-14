import argparse
import concurrent.futures

from tqdm import tqdm

from src.apn.apn import *

def make(batch_size, device, prefix=""):
    T = add_table(6, device=device)
    trace_table = compute_trace_table(6, device=device)
    list_P = []
    count = 0
    while True:
        F = torch.randint(0, 64, (batch_size, 64), device=device)
        gradient_descent(F, T, trace_table)
        list_P.append(F)
        count += batch_size
        if len(list_P) % 20 == 0:
            torch.save(torch.cat(list_P, dim=0), f'{prefix}_training_{device}_{count}_test.pt')
            list_P = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-device', default=1, type=int)
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--prefix', default="")
    args = parser.parse_args()
    
    to_do = [f'cuda:{k}' for k in range(args.num_device)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_device) as executor:
        futures = []
        for device in to_do:
            futures.append(executor.submit(make, args.batch_size, device, prefix=args.prefix))
        for _ in tqdm(concurrent.futures.as_completed(futures), desc="Overall progress", position=0, total=len(futures)):
            pass
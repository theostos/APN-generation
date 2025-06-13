"""
you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.
"""

import os
import sys
import time
import argparse
import json

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from src.training.model import generate, ModelConfig, Transformer
from src.training.dataset import create_datasets, InfiniteDataLoader, create_eval_datasets
from src.apn.apn import score_functions, evaluate_polynomials, check_if_permutation, compute_degrees, compute_delta_twisted_table, compute_delta_table, compute_derivative

def evaluate_delta(model, X, T, device, batch_size=8192):
    field_size = T.size(0)
    idx = torch.ones((batch_size, 1), dtype=torch.long, device=device) * field_size

    model.eval()
    P = generate(model, idx, field_size, do_sample=True)
    P = torch.clip(P, max=T.size(0)-1)

    degrees = compute_degrees(P, 6)
    P = P[degrees > 2,:]
    F = evaluate_polynomials(P, X, T)
    is_perm = check_if_permutation(F)
    # if not torch.any(is_perm):
    #     return torch.tensor([-1.]), torch.tensor([-1.]), torch.tensor([-1.]), torch.tensor([-1.]), torch.tensor([-1.]), torch.tensor([-1.]), torch.tensor([-1.])
    # F = F[is_perm,:]
    # P = P[is_perm,:]
    # is_perm = torch.tensor([0])
    deltas_max, deltas_mean = score_functions(F, T)
    
    delta_max = deltas_max.max()
    delta_min = deltas_max.min()
    target = torch.zeros(T.size(0), device=P.device)
    target[-1] = 2016
    target[-2] = 2016
    DF = compute_derivative(F, T)
    delta_table = compute_delta_table(DF).reshape(batch_size, 63, 64)
    twisted_delta = compute_delta_twisted_table(delta_table)
    distance_target = (torch.sqrt((twisted_delta - target)**2)).sum(dim=1)[0]

    idx_sol = (deltas_max == 2)
    P = P[idx_sol,:]

    if P.size(0) > 0:
        print(P)
    # if P.size(0) > 0:
    #     for p in P:
    #         non_zero = (p < 63).nonzero()
    #         non_zero = non_zero.flatten().tolist()
    #         if non_zero[0] == 0:
    #             non_zero = non_zero[1:]
    #         deg = max([bin(a).count('1') for a in non_zero])
    #         if len(non_zero)> 3 or deg > 2:
    #                 print(p)
    #                 print(non_zero)
    #                 print(deg)
    #                 print()
    #         mean_deg_list.append(deg)
    model.train()
    return deltas_max.float().mean(), deltas_mean.float().mean(), deltas_mean.float().min(), delta_max, delta_min, torch.tensor([is_perm.sum()/batch_size*100]), degrees.float().mean(), distance_target.float().mean()
    

# def print_samples(num=10):
#     """ samples from the model and pretty prints the decoded samples """
#     X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
#     top_k = args.top_k if args.top_k != -1 else None
#     steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
#     X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
#     train_samples, test_samples, new_samples = [], [], []
#     for i in range(X_samp.size(0)):
#         # get the i'th row of sampled integers, as python list
#         row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
#         # token 0 is the <STOP> token, so we crop the output sequence at that point
#         crop_index = row.index(0) if 0 in row else len(row)
#         row = row[:crop_index]
#         word_samp = train_dataset.decode(row)
#         # separately track samples that we have and have not seen before
#         if train_dataset.contains(word_samp):
#             train_samples.append(word_samp)
#         elif test_dataset.contains(word_samp):
#             test_samples.append(word_samp)
#         else:
#             new_samples.append(word_samp)
#     print('-'*80)
#     for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
#         print(f"{len(lst)} samples that are {desc}:")
#         for word in lst:
#             print(word)
#     print('-'*80)

@torch.inference_mode()
def evaluate(model, rank, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(rank) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    # parser.add_argument('--input-file', '-i', type=str, default='export_eval_P.pt', help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=1, help="number of data workers for both train/test")
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
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if (rank % world_size == 0):
        os.makedirs(args.work_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.work_dir)
    
    # init datasets
    eval_filename = "export_sparse_P_eval.pt"
    delta_filename = "export_tot.pt"
    # train_dataset, test_dataset = create_datasets(args.input_file)
    train_eval_dataset, _ = create_eval_datasets(eval_filename)
    train_delta_dataset, _ = create_datasets(delta_filename)

    batch_loader_eval = InfiniteDataLoader(train_eval_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    batch_loader_delta = InfiniteDataLoader(train_delta_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    vocab_size = train_eval_dataset.get_vocab_size()
    block_size = train_eval_dataset.get_output_length()
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    model_base = Transformer(config)
    model_base = model_base.to(rank)
    optimizer = torch.optim.AdamW(model_base.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)
    model = DDP(model_base, device_ids=[rank])
    step = 0
    eval_train = False
    delta_train = True

    with open(f"tables/add_table_6", "r") as fp:
        T = json.load(fp)
    T = torch.tensor(T, device=rank)

    with open(f"tables/power_table_6", "r") as fp:
        X_pow = json.load(fp)
    X_pow = torch.tensor(X_pow, device=rank)
    assert eval_train or delta_train, "need at least one criterium of training"
    
    while True:
        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        loss_eval = torch.tensor(0.)
        loss_delta = torch.tensor(0.)

        if eval_train:
            batch_eval = batch_loader_eval.next()
            batch_eval = [t.to(rank) for t in batch_eval]
            X, Y = batch_eval
            # feed into the model
            _, loss_eval = model(X, Y)

        if delta_train:
            batch_delta = batch_loader_delta.next()
            batch_delta = [t.to(rank) for t in batch_delta]
            X, Y = batch_delta
            _, loss_delta = model(X, Y)
    
        loss = loss_eval + loss_delta
        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0 and rank%world_size == 0:
            print(f"step {step} | loss eval {loss_eval.item():.4f} | loss delta {loss_delta.item():.4f} | step time {(t1-t0)*1000:.2f}ms")
            writer.add_scalar("Loss/train_eval", loss_eval.item(), step)
            writer.add_scalar("Loss/train_delta", loss_delta.item(), step)
            writer.flush()

        # evaluate the model
        if step % 20 == 0 and rank == 0:
            deltas_max, deltas_mean, delta_mean_min, delta_max, delta_min, proba_perm, mean_deg, distance_spectra  = evaluate_delta(model_base, X_pow, T, rank)
            print(f"Distance spectra: {distance_spectra.item():.4f}, deltas max: {deltas_max.item():.4f}, deltas mean: {deltas_mean.item():.4f}, deltas mean_min: {delta_mean_min.item():.4f}, max: {delta_max.item():.4f}, min: {delta_min.item():.4f}, proba permutation: {proba_perm.item():.4f}, mean deg: {mean_deg.item():.4f}")
            writer.add_scalar("Eval/Deltas", deltas_max.item(), step)
            writer.add_scalar("Eval/Deltas mean", deltas_mean.item(), step)
            writer.add_scalar("Eval/Deltas mean min", delta_mean_min.item(), step)

            writer.add_scalar("Eval/Delta max", delta_max.item(), step)
            writer.add_scalar("Eval/Delta min", delta_min.item(), step)
            writer.add_scalar("Eval/Proba permutation", proba_perm.item(), step)
            writer.add_scalar("Eval/Proba apn", proba_perm.item(), step)
            writer.add_scalar("Eval/Mean deg", mean_deg.item(), step)

            writer.add_scalar("Eval/Distance target spectra", distance_spectra.item(), step)
            # writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
        if step % 1100 == 1000 and rank == 0:
            out_path = os.path.join(args.work_dir, f"model_{step}.pt")
            torch.save(model.state_dict(), out_path)
        step += 1
        if args.max_steps >= 0 and step >= args.max_steps:
            break
    destroy_process_group()

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    device=0
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, ), nprocs=world_size)
    main(device, world_size)
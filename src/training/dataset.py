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
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

class EvalDataset(Dataset):

    def __init__(self, data_P, data_F):
        self.field_size = data_P.size(1)
        self.data_P = data_P
        self.data_F = data_F

    def __len__(self):
        return len(self.data_P)

    def get_vocab_size(self):
        return self.field_size + 2 # ALL FIELD ELEMENTS + <START> + <EVAL>

    def get_output_length(self):
        return 2*self.field_size + 2 # <START> + POLYNOMIAL + <EVAL> + RESULT

    def __getitem__(self, idx):
        P = self.data_P[idx]
        F = self.data_F[idx]

        x = torch.zeros(1 + self.field_size + 1 + (self.field_size-1), dtype=torch.long)
        y = torch.ones(self.field_size + 1 + self.field_size, dtype=torch.long) * -1
        x[0] = self.field_size
        x[1:1+self.field_size] = P
        x[1+self.field_size] = self.field_size + 1
        x[1+self.field_size + 1:] = F[:-1]

        y[:-1] = x[1:]
        return x, y

class CharDataset(Dataset):

    def __init__(self, data):
        self.field_size = data.size(1)
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_vocab_size(self):
        return self.field_size + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.field_size + 1 # words followed by <START> token

    def __getitem__(self, idx):
        entry = self.data[idx]
        x = torch.zeros(self.field_size + 1, dtype=torch.long)
        y = torch.zeros(self.field_size + 1, dtype=torch.long)
        x[1:] = entry
        y[:-1] = entry
        x[0] = self.field_size
        y[-1] = -1 # index -1 will mask the loss at the inactive location (the last position)
        return x, y

def split_tensor(t, size):
    batch_size = t.size(0)
    sample = torch.rand(batch_size).topk(size).indices

    mask = torch.ones(batch_size, dtype=torch.bool)
    mask.scatter_(dim=0, index=sample, value=False)
    return t[mask], t[torch.logical_not(mask)]

def split_bitensor(t0, t1, size):
    batch_size = t0.size(0)
    sample = torch.rand(batch_size).topk(size).indices

    mask = torch.ones(batch_size, dtype=torch.bool)
    mask.scatter_(dim=0, index=sample, value=False)
    return t0[mask], t1[mask], t0[torch.logical_not(mask)], t1[torch.logical_not(mask)]

def create_eval_datasets(input_file):
    # preprocessing of the input text file
    data_P, data_F = torch.load(input_file, weights_only=True)
    data_P = data_P.to('cpu')
    data_F = data_F.to('cpu')
    batch_size = data_P.size(0)

    test_set_size = min(1000, int(batch_size * 0.1))
    train_P, train_F, test_P, test_F = split_bitensor(data_P, data_F, test_set_size)

    # wrap in dataset objects
    train_dataset = EvalDataset(train_P, train_F)
    test_dataset = EvalDataset(test_P, test_F)

    return train_dataset, test_dataset

def create_datasets(input_file):
    # preprocessing of the input text file
    data = torch.load(input_file, weights_only=True)
    data = data.to('cpu')
    batch_size = data.size(0)

    test_set_size = min(1000, int(batch_size * 0.1))
    train_t, test_t = split_tensor(data, test_set_size)

    # wrap in dataset objects
    train_dataset = CharDataset(train_t)
    test_dataset = CharDataset(test_t)

    return train_dataset, test_dataset

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, multi_gpu=False, **kwargs):
        
        # train_sampler = torch.utils.data.SequentialSampler(dataset)
        if multi_gpu:
            train_sampler = DistributedSampler(dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
    
        self.train_loader = DataLoader(dataset, sampler=train_sampler, shuffle=False, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

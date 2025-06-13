import torch
import os

files = []
for filename in os.listdir('.'):
    if filename.endswith('.pt'):
        files.append(filename)

list_tensor = []

for file in files:
    tensor = torch.load(file, map_location='cpu')
    list_tensor.append(tensor.squeeze(1))
tensor_tot = torch.concat(list_tensor, dim=0)
torch.save(tensor_tot, 'export_tot.pt')



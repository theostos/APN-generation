import torch

files_templates = ['export_sparse_NONperm_0', 'export_sparse_NONperm_1', 'export_sparse_NONperm_2']
files = []
for k in range(4, 38):
    for file in files_templates:
        files.append(file + f'_{k}.pt')

list_tensor = []

for file in files:
    list_tensor.append(torch.load(file))
tensor_tot = torch.concat(list_tensor, dim=0)

torch.save(tensor_tot, 'export_tot.pt')



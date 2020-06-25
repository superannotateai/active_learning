import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms

import csv


# Chooses 'count' images, returns their indexes in the dataset and corresponding loss values.
def choose_active_learning_indices(net, active_cycle, rs, pool_idx, dataset, device, count=1000):
    if active_cycle == 0:
        idx = rs.choice(pool_idx, count, replace=False)
        for id in idx:
            pool_idx.remove(id)
        return idx, None
    cycle_subs_idx = rs.choice(pool_idx, 10 * count, replace=False)
    cycle_pool = Subset(dataset, cycle_subs_idx)
    cycle_loader = DataLoader(
        cycle_pool, batch_size=1, shuffle=False, num_workers=2
    )
    net.eval()
    pred_l = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(cycle_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            out, loss_pred = net(inputs)
            pred_l.append(loss_pred.item())
        pred_l = np.array(pred_l)
        idx = pred_l.argsort()[-count:][::-1]
        cycle_ret_idx = []
        for id in idx:
            cycle_ret_idx.append(cycle_subs_idx[id])
            pool_idx.remove(cycle_subs_idx[id])
        return cycle_ret_idx, pred_l[idx]


def random_indices(pool_idx, rand_state, count=1000):
    idx = rand_state.choice(pool_idx, count, replace=False)
    for id in idx:
        pool_idx.remove(id)
    return idx


def write_entropies_csv(dataset, indices, losses, file_out):
    image_names = dataset.image_list[indices]
    with open(file_out, 'w', newline='') as csvfile:
        fieldnames = ['name', 'entropy_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for name, entropy in zip(indices, losses):
            writer.writerow({'name': name, 'entropy_value': entropy})



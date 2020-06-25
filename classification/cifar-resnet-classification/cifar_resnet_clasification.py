'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

from torchvision import datasets
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tqdm
from models import *
import sys

sys.path.append(os.path.abspath('../../active_learning'))

from active_learning import ActiveLearning
from active_loss import LossPredictionLoss
from active_learning_utils import choose_active_learning_indices, random_indices, write_entropies_csv


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument(
    '--resume', '-r', action='store_true', help='resume from checkpoint'
)
args = parser.parse_args()

rand_state = np.random.RandomState(1311)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
weight_decay = 1e-3
pool_idx = list(range(50000))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

testset = datasets.CIFAR10(
    root='/media/disk_drive/datasets/cifar10',
    train=False,
    download=True,
    transform=transform_test
)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck'
)

criterion = nn.CrossEntropyLoss(reduction='none')

# Training
def train(epoch, net, train_loader, optimizer, use_active_learning=False, lamda=1):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, targets) in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if use_active_learning:
            if epoch < 120:
                outputs, loss_pred = net(inputs)
            else:
                outputs, loss_pred = net(inputs, detach_lp=True)
            loss_pred = loss_pred.view(loss_pred.size(0))
        else:
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        if use_active_learning:
            criterion_lp = LossPredictionLoss()
            lp = lamda * criterion_lp(loss_pred, loss)
        else:
            lp = 0
        loss = torch.sum(loss) / loss.size(0)
        loss_total = loss + lp
        loss_total.backward()
        optimizer.step()
        train_loss += loss_total.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress = tqdm.tqdm(enumerate(testloader), total=len(testloader))
        for batch_idx, (inputs, targets) in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs = outputs[0] if len(outputs) == 2 else outputs
            loss = criterion(outputs, targets).mean()
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress.set_description('Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return acc


accuracies = []
net = None

def run_training(use_active_learning=False):
    global net
    global pool_idx
    global rand_state
    accuracies = []
    train_idx = []
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    pool_idx = list(range(50000))
    net = ResNet18()
    if use_active_learning:
        net = ActiveLearning(net)
    net = net.to(device)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    if device == 'cuda':
        cudnn.benchmark = True
    for cycle in range(10):
        print("========= Running with {} cycle={}".format(
            "Active Learning" if use_active_learning else "Random", cycle))
        if use_active_learning:
            dataset = datasets.CIFAR10(
                root='/media/disk_drive/datasets/cifar10',
                train=True,
                download=True,
                transform=transform_test
            )
            indices, losses = choose_active_learning_indices(net, cycle, rand_state, pool_idx,
                    dataset, device)
            # TODO(martun): Write out entropy CSV file, need to access the file names of the dataset.
            # write_entropies_csv(dataset, indices, losses, "entropy_file_{}.csv".format(cycle)
        else:
            indices = random_indices(pool_idx, rand_state, count=1000)
        train_idx.extend(indices)
        train_dataset = Subset(
            datasets.CIFAR10(
                root='/media/disk_drive/datasets/cifar10',
                train=True,
                download=True,
                transform=transform_train
            ), train_idx
        )
        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=2
        )
        optimizer = optim.SGD(
            net.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay
        )
        if cycle == 0:
            first_1k = list(train_idx)
        for epoch in range(200):
            if epoch == 160:
                optimizer = optim.SGD(
                    net.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay
                )
            train(epoch, net, train_loader, optimizer, use_active_learning=use_active_learning)
        cycle_acc = test(cycle)
        accuracies.append(cycle_acc)
    print("{} accuracies: {}".format(
        "Active Learning" if use_active_learning else "Random",str(accuracies)))


lp_accs = []
rand_accs = []


def plot_accuracies():
    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(1, 11), rand_accs, 'b-')
    plt.plot(np.arange(1, 11), lp_accs, 'r-')
    plt.xlabel("Number of images used in training in 1000s")
    plt.ylabel("Test Accuracy")
    plt.title("Loss prediction vs. random sample")
    plt.savefig("lp_rand.png")

#run_training(use_active_learning=False)
#rand_accs = accuracies
run_training(use_active_learning=True)
lp_accs = accuracies

plot_accuracies()


'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

from torchvision import datasets
import torchvision.transforms as transforms

import pickle
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
from active_learning_utils import *
from discriminative_learning import * 


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--resume', '-r', action='store_true', help='resume from checkpoint'
)
parser.add_argument(
    '--use_discriminative_al',
    required=False,
    type=bool,
    default=False,
    help='If True, will use Discriminative Active Learning.'
)
parser.add_argument(
    '--use_loss_prediction_al',
    required=False,
    type=bool,
    default=False,
    help='If True, will use Loss Prediction Active Learning.'
)
parser.add_argument(
    '--input_pickle_file',
    required=False,
    type=str,
    default=None,
    help='Path to a pickle file with pre-computed indices to use.'
)
parser.add_argument(
    '--output_pickle_file',
    required=False,
    type=str,
    default=None,
    help='Path to the output pickle file with the selected indices. Can be later used to reproduce results.'
)


args = parser.parse_args()

# For reproducability.
rand_state = np.random.RandomState(1311)
torch.manual_seed(1311)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
weight_decay = 5e-4
unlabeled_idx = list(range(50000))

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

complete_train_dataset = datasets.CIFAR10(
    root='/media/disk_drive/datasets/cifar10',
    train=True,
    download=True,
    transform=transform_train)

complete_train_dataset_no_augmentation = datasets.CIFAR10(
    root='/media/disk_drive/datasets/cifar10',
    train=True,
    download=True,
    transform=transform_test)

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
def train(epoch, net, train_loader, optimizer,
          use_loss_prediction_al=False, lamda=1, use_discriminative_al=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # For tracking progress on loss prediction active learing.
    correctly_ranked = 0
    total_ranked = 0

    progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (inputs, targets) in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if use_loss_prediction_al:
            # Loss prediction module is trained with the main network.
            if epoch < 120:
                outputs, loss_pred = net(inputs)
            else:
                outputs, loss_pred = net(inputs, detach_lp=True)
            loss_pred = loss_pred.view(loss_pred.size(0))
        elif use_discriminative_al:
            # We are not training the active learning part here, will be 
            # trained separataly layer.
            outputs, labeled_unlabeled_predictions = net(inputs)
        else:
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        if use_loss_prediction_al:
            criterion_lp = LossPredictionLoss()
            lp = lamda * criterion_lp(loss_pred, loss)
            # Also compute (an estimate) of the ranking accuracy for the training set.
            batch_size = loss.shape[0]
            for l1 in range(batch_size):
                for l2 in range(l1):
                    total_ranked += 1
                    if (loss[l1] - loss[l2]) * (loss_pred[l1] - loss_pred[l2]) > 0:
                        correctly_ranked += 1
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

        if use_loss_prediction_al:
            progress.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d) Prediction Acc %.3f%%' % (
                train_loss / (batch_idx + 1),
                100. * correct / total, correct, total,
                correctly_ranked / (total_ranked + 0.0001)))
        else:
            progress.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(net, epoch):
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


def run_training(
        use_loss_prediction_al=False, input_pickle_file=None,
        images_per_cycle=1000, cycles_count=10, use_discriminative_al=False):
    global unlabeled_idx
    global rand_state

    accuracies = []
    labeled_idx = []
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    unlabeled_idx = list(range(50000))

    if device == 'cuda':
        cudnn.benchmark = True
    labeled_idx_per_cycle = []
    net = None
    for cycle in range(cycles_count):
        new_indices, entropies = choose_new_labeled_indices(
            net, complete_train_dataset_no_augmentation,
            cycle, rand_state, labeled_idx, unlabeled_idx, device, images_per_cycle,
            use_loss_prediction_al, use_discriminative_al, input_pickle_file)
        
        labeled_idx.extend(new_indices)
        unlabeled_idx = [x for x in unlabeled_idx if x not in new_indices]
        print("Number of labeled images now is {}, unlabeled {}".format(
            len(labeled_idx), len(unlabeled_idx)))

        # Remember new indices as a list of lists.
        labeled_idx_per_cycle.append(new_indices)
        
        # Reset the network, train from the start.
        net = ResNet18()
        if use_loss_prediction_al:
            net = ActiveLearning(net)
        elif use_discriminative_al:
            net = DiscriminativeActiveLearning(net)
        net = net.to(device)
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
        
        train_dataset = Subset(complete_train_dataset, labeled_idx)
        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=2
        )
        optimizer = optim.SGD(
            net.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay
        )
        for epoch in range(200):
            if epoch == 160:
                optimizer = optim.SGD(
                    net.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay
                )
            train(epoch, net, train_loader, optimizer,
                  use_loss_prediction_al=use_loss_prediction_al,
                  use_discriminative_al=use_discriminative_al)
        cycle_acc = test(net, cycle)
        accuracies.append(cycle_acc)
        print("===/*/*=== Accuracies so far {}".format(accuracies))
    return accuracies, labeled_idx_per_cycle


accuracies, labeled_idx_per_cycle = run_training(
    use_loss_prediction_al=args.use_loss_prediction_al,
    use_discriminative_al=args.use_discriminative_al,
    input_pickle_file=args.input_pickle_file)


print("{} accuracies: {}".format(
    get_algorithm_name(
        args.use_loss_prediction_al, args.use_discriminative_al,
        args.input_pickle_file),
    accuracies))

# Save selected indices to a pickle file.
if args.output_pickle_file is not None:
    write_indices_file(args.output_pickle_file, labeled_idx_per_cycle)


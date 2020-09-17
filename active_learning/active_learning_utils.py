import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import numpy as np

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets
from discriminative_learning import *

from hdf5_dataset import HDF5Dataset

import copy
import pickle
import csv


def choose_indices_loss_prediction_active_learning(
        net, active_cycle, rand_state, unlabeled_idx, dataset, device, count=1000,
        subset_factor=10, is_human_pose=False):
    """ Chooses 'count' images, returns their indices in the dataset and corresponding loss values,
        using loss prediction Active Learning algorithm.
    """
    if active_cycle == 0:
        idx = rand_state.choice(unlabeled_idx, count, replace=False)
        for id in idx:
            unlabeled_idx.remove(id)
        return idx, None
    cycle_subs_idx = rand_state.choice(
        unlabeled_idx,
        min(subset_factor * count, len(unlabeled_idx)),
        replace=False)
    cycle_pool = Subset(dataset, cycle_subs_idx)
    cycle_loader = DataLoader(
        cycle_pool, batch_size=4, shuffle=False, num_workers=2
    )
    net.eval()
    pred_l = []
    with torch.no_grad():
        if is_human_pose:
            for batch_idx, (inputs, targets, target_weight, meta) in enumerate(cycle_loader):
                inputs = inputs.to(device)
                out, loss_pred = net(inputs)
                loss_pred = torch.flatten(loss_pred)
                pred_l.extend(loss_pred.tolist())
        else:
            for batch_idx, (inputs, targets) in enumerate(cycle_loader):
                inputs = inputs.to(device)
                out, loss_pred = net(inputs)
                loss_pred = torch.flatten(loss_pred)
                pred_l.extend(loss_pred.tolist())
        pred_l = np.array(pred_l)
        idx = pred_l.argsort()[-count:][::-1]
        new_labeled_idx = []
        for id in idx:
            new_labeled_idx.append(cycle_subs_idx[id])
            unlabeled_idx.remove(cycle_subs_idx[id])
        return new_labeled_idx, pred_l[idx]


def choose_discriminative_al_indices(
        discriminative_model, hdf5_dataset_path, total_image_count,
        active_cycle, rand_state, labeled_idx,
        unlabeled_idx, device, sub_sample_size=1000, is_human_pose=False):
    """ Chooses 'count' images, returns their indices in the dataset and 
        corresponding confidence values, using Discriminative Active Learning algorithm.
    """
    if active_cycle == 0:
        new_idx = rand_state.choice(unlabeled_idx, count, replace=False)
        unlabeled_idx = [x for x in unlabeled_idx if x not in new_idx]
        return new_idx, None
    dataset = HDF5Dataset(hdf5_dataset_path,
        image_ids=["features_{}".format(i) for i in range(total_image_count)])
    cycle_pool = Subset(dataset, unlabeled_idx)
    cycle_loader = DataLoader(
        cycle_pool, batch_size=1, shuffle=False, num_workers=1
    )
    discriminative_model.eval()
    confidence_values = []
    with torch.no_grad():
        # Targets is for potential target values/masks for images, which are 0 for us.
        for batch_idx, (inputs, targets) in enumerate(cycle_loader):
            inputs = inputs.to(device)
            outputs, confidence = discriminative_model(inputs)
            # Apply softmax, and take the predicted probability that
            # the image from unabeled set is from unlabeled set. 
            # If the classifier is able to predict that a given example is from
            # the unlabeled set, we better move it to labeled.
            confidence = nn.Softmax(dim=1)(confidence)[:, 0]
            confidence_values.append(confidence.item())
        confidence_values = np.array(confidence_values)
        idx = confidence_values.argsort()[-sub_sample_size:][::-1]
        new_labeled_idx = unlabeled_idx[idx]
        unlabeled_idx = [x for x in unlabeled_idx if x not in new_labeled_idx]
    return new_labeled_idx, confidence_values[idx]


def random_indices(unlabeled_idx, rand_state, count=1000):
    idx = rand_state.choice(unlabeled_idx, count, replace=False)
    for id in idx:
        unlabeled_idx.remove(id)
    return idx


def read_indices_from_file(indices_pickle_file, cycle, images_per_cycle):
    with open(indices_pickle_file, 'rb') as f:
        idx = pickle.load(f)
    return idx[0][cycle];


def write_indices_file(indices_pickle_file, labeled_idx_per_cycle):
    with open(indices_pickle_file, 'wb') as f:
        pickle.dump([labeled_idx_per_cycle], f)
    print("Saved selected indices to {}".format(indices_pickle_file))


def write_entropies_csv(dataset, indices, entropies, file_out):
    """ Writes image paths and entropy values of images with given ids to an annotate.online readable csv file.
        Parameters:
            dataset - Any pytorch dataset, which has member function "get_image_path".
            indices - List of indices of the images to write to csv.
            entropies - Entropy values of images, can be any real values.
            file_out - Output csv file path.
    """
    with open(file_out, 'w', newline='') as csvfile:
        fieldnames = ['name', 'entropy_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, entropy in zip(indices, entropies):
            writer.writerow(
                {'name': dataset.get_image_path(index),
                 'entropy_value': entropy}
                )


def get_algorithm_name(use_loss_prediction_al, use_discriminative_al, indices_pickle_file):
    if use_loss_prediction_al:
        return "Loss prediction Active Learning"
    elif use_discriminative_al:
        return "Discriminative Active Learning"
    elif indices_pickle_file is not None:
        return "Loading indices from pickle file"
    else:
        return "Random"


def loss_value_histogram(
        net, cycle, rand_state, unlabeled_idx,
        dataset, device, criterion, is_human_pose=False):
    """ Draws a histogram of loss values for all unlabeled training images.
    """
    losses = get_loss_values(net, cycle, rand_state, unlabeled_idx,
        dataset, device, criterion, is_human_pose)
    print("Histograming losses of shape {}".format(losses.shape))
    print(losses)
    writer = SummaryWriter(comment=f'Losses_at_cycle_{cycle}')
    writer.add_histogram('Loss_histogram', losses, cycle, 'auto')
    writer.close()



def loss_value_min_max_average(net, cycle, rand_state, unlabeled_idx,
        dataset, device, criterion, is_human_pose=False):
    """ Runs inference 10 times, computes min, max and avg of losses per images and 
        draws a chart in tensorboard.
    """
    all_losses = []
    for i in range(10):
        losses = get_loss_values(net, cycle, rand_state, unlabeled_idx,
            dataset, device, criterion, is_human_pose)
        all_losses.append(losses)
    losses_array = np.vstack(all_losses)
    print("Shapes of losses_array = {}".format(losses_array.shape))
    min_array = np.amin(losses_array, axis=0)
    max_array = np.amax(losses_array, axis=0)
    mean_array = np.mean(losses_array, axis=0)
    writer = SummaryWriter(comment=f'Losses_at_cycle_{cycle}')
    for idx, value in enumerate(min_array):
        writer.add_scalar('Min_losses_{}'.format(cycle), value, idx)
    for idx, value in enumerate(max_array):
        writer.add_scalar('Max_losses_{}'.format(cycle), value, idx)
    for idx, value in enumerate(mean_array):
        writer.add_scalar('Mean_losses_{}'.format(cycle), value, idx)

    # Also draw histograms for all these values.
    writer.add_histogram('Min_loss_histogram', min_array, cycle, 'auto')
    writer.add_histogram('Max_loss_histogram', max_array, cycle, 'auto')
    writer.add_histogram('Mean_loss_histogram', mean_array, cycle, 'auto')
    writer.close()


def get_loss_values(
        net, cycle, rand_state, unlabeled_idx,
        dataset, device, criterion, is_human_pose=False):
    cycle_pool = Subset(dataset, unlabeled_idx)
    cycle_loader = DataLoader(
        cycle_pool, batch_size=4, shuffle=False, num_workers=2
    )
    net.eval()
    all_losses = []
    with torch.no_grad():
        if is_human_pose:
            for batch_idx, (inputs, targets, target_weight, meta) in enumerate(cycle_loader):
                inputs = inputs.to(device)
                out = net(inputs)
                if type(criterion) in [torch.nn.modules.loss.L1Loss,
                                       torch.nn.modules.loss.MSELoss]:
                    targets = targets.float()
                targets = targets.cuda(non_blocking=True)
                loss = criterion(out, targets)
                all_losses.extend(loss.tolist())
        else:
            for batch_idx, (inputs, targets) in enumerate(cycle_loader):
                inputs = inputs.to(device)
                # Next line's [0] may need to be removed if task is not segmentation.
                # Also loss.mean([1, 2]) may be not required. This function was tested 
                # for segmentation only.
                out = net(inputs)[0]
                if type(criterion) in [torch.nn.modules.loss.L1Loss,
                                       torch.nn.modules.loss.MSELoss]:
                    targets = targets.float()
                targets = targets.cuda(non_blocking=True)
                loss = criterion(out, targets)
                # Compute means from [N, W, H] to [N].
                loss = loss.mean([1, 2])
                all_losses.extend(loss.tolist())
        all_losses = np.array(all_losses)
        return all_losses


def choose_new_labeled_indices_using_gt(
        net, cycle, rand_state, unlabeled_idx,
        dataset, device, criterion, count=1000,
        subset_factor=10, is_human_pose=False):
    if cycle == 0:
        idx = rand_state.choice(unlabeled_idx, count, replace=False)
        for id in idx:
            unlabeled_idx.remove(id)
        return idx, None
    cycle_subs_idx = rand_state.choice(
        unlabeled_idx,
        min(subset_factor * count, len(unlabeled_idx)),
        replace=False)
    cycle_pool = Subset(dataset, cycle_subs_idx)
    cycle_loader = DataLoader(
        cycle_pool, batch_size=4, shuffle=False, num_workers=2
    )
    net.eval()
    pred_l = []
    with torch.no_grad():
        if is_human_pose:
            for batch_idx, (inputs, targets, target_weight, meta) in enumerate(cycle_loader):
                inputs = inputs.to(device)
                out = net(inputs)
                if type(criterion) in [torch.nn.modules.loss.L1Loss,
                                       torch.nn.modules.loss.MSELoss]:
                    targets = targets.float()
                targets = targets.cuda(non_blocking=True)
                loss = criterion(out, targets)
                pred_l.extend(loss.tolist())
        else:
            for batch_idx, (inputs, targets) in enumerate(cycle_loader):
                inputs = inputs.to(device)
                # Next line's [0] may need to be removed if task is not segmentation.
                # Also loss.mean([1, 2]) may be not required. This function was tested 
                # for segmentation only.
                out = net(inputs)[0]
                if type(criterion) in [torch.nn.modules.loss.L1Loss,
                                       torch.nn.modules.loss.MSELoss]:
                    targets = targets.float()
                targets = targets.cuda(non_blocking=True)
                loss = criterion(out, targets)
                # Compute means from [N, W, H] to [N].
                loss = loss.mean([1, 2])
                pred_l.extend(loss.tolist())
        pred_l = np.array(pred_l)
        idx = pred_l.argsort()[-count:][::-1]
        new_labeled_idx = []
        for id in idx:
            new_labeled_idx.append(cycle_subs_idx[id])
            unlabeled_idx.remove(cycle_subs_idx[id])
        return new_labeled_idx, pred_l[idx]


def choose_new_labeled_indices(
        net, complete_trainset_no_augmentation, 
        cycle, rand_state, labeled_idx, unlabeled_idx, device, images_per_cycle, 
        use_loss_prediction_al, use_discriminative_al, input_pickle_file):
    print("========= Chossing new labeled indices algorithm={} cycle={}".format(
        get_algorithm_name(
            use_loss_prediction_al, use_discriminative_al, input_pickle_file),
        cycle))
    if cycle == 0:
        new_indices = random_indices(unlabeled_idx, rand_state, count=images_per_cycle)
        return new_indices, None

    if use_discriminative_al:
        # Select count/subquery_count images at a time.
        subquery_count = 10
        entropies = []
        all_new_indices = []
        labeled_idx_copy = copy.deepcopy(labeled_idx)
        for i in range(subquery_count):
            # Reset the active learning layers.
            # net.reset_al_layers()

            subset_factor = 10
            cycle_subs_idx = rand_state.choice(
                unlabeled_idx,
                min(subset_factor * len(labeled_idx), len(unlabeled_idx)),
                replace=False)
            # This is the only time when we train the active learning algorithm
            # during image subset selection.
            hdf5_dataset_path = "features_dataset.h5"
            discriminative_model = train_discriminative_al(
                net, device, complete_trainset_no_augmentation, labeled_idx_copy,
                cycle_subs_idx, hdf5_dataset_path, len(complete_trainset_no_augmentation))
            new_indices, subquery_entropies = choose_discriminative_al_indices(
                discriminative_model, hdf5_dataset_path, len(complete_trainset_no_augmentation),
                cycle, rand_state, labeled_idx_copy,
                cycle_subs_idx, device, sub_sample_size=images_per_cycle // subquery_count)
            entropies.extend(subquery_entropies.tolist())
            labeled_idx_copy.extend(new_indices)
            all_new_indices.extend(new_indices)
            unlabeled_idx = [x for x in unlabeled_idx if x not in new_indices]
        entropies = np.array(entropies)
        return all_new_indices, entropies
    elif use_loss_prediction_al:
        new_indices, losses = choose_indices_loss_prediction_active_learning(
            net, cycle, rand_state, unlabeled_idx,
            complete_trainset_no_augmentation, device, count=images_per_cycle)
        return new_indices, losses
    elif input_pickle_file is not None:
        new_indices = read_indices_from_file(
            input_pickle_file, cycle=cycle,
            images_per_cycle=images_per_cycle)
        print("Loaded indices {}".format(indices))
        return new_indices, None
    else:
        new_indices = random_indices(unlabeled_idx, rand_state, count=images_per_cycle)
        return new_indices, None



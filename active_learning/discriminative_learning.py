''' Discriminative Active learning wrapper model for pytoch Models.
    Implement function get_discriminative_active_learning_layers() and pass your model to
    constructor of the DiscriminativeActiveLearning class.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import time
import numpy as np
import tqdm
import sys
import os

import h5py

sys.path.append(os.path.abspath('../../hdf5_wrappers'))
from hdf5_dataset import HDF5Dataset
from hdf5_wrappers import matrix_to_hdf5


class DiscriminativeActiveLearning(nn.Module):
    
    def __init__(self, base_model, input_shapes):
        """ 
        Params:
            base_model - The model being trained. Used for image features representation.
            input_shapes - Shape of the input images. Used for selection of perceptron width. Can use base_model.get_discriminative_al_layer_shapes() to get this values.
        """
        super(DiscriminativeActiveLearning, self).__init__()
        self.base_model = base_model
        self.input_shapes = input_shapes
        self.total_input_size = np.sum([np.prod(x) for x in self.input_shapes])
        # Running on Mnist, we will not use this.
        if self.total_input_size < 30:
            width = 20
        # Running ResNet-18 on cifar-10.
        elif self.total_input_size <= 512:
            width = 256
        else:
            # Running something larger on cityscapes or coco
            width = 512

        print("Width for Discriminative active learning is {}".format(width));
        self.fc1 = nn.Linear(self.total_input_size, width) 
        self.fc2 = nn.Linear(width, width) 
        self.fc3 = nn.Linear(width, width) 
        self.out = nn.Linear(width, 2)

        layers = [self.fc1,
                  nn.ReLU(inplace=True),
                  self.fc2,
                  nn.ReLU(inplace=True),
                  self.fc3,
                  nn.ReLU(inplace=True),
                  self.out]
        self.sequential = nn.Sequential(*layers)

    # Re-initializes the active learning layers. Called on every cycle of image labeling,
    # so instead of creating a new model instance, we can re-initialize the old one.
    def reset_al_layers(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x):
        if self.base_model is not None:
            base_model_output = self.base_model.forward(x)
            active_learning_features = self.base_model.get_discriminative_al_features()
            # active_learning_features = [f.detach() for f in active_learning_features]
            features = [torch.flatten(f, 1) for f in active_learning_features]
            features_flat = torch.cat(features, dim=1);
        else:
            base_model_output = x
            features_flat = x
        return base_model_output, self.sequential(features_flat)

    def freeze_main_layers(self, requires_grad):
        """ Used to freeze/unfreeze the layers of main network. Need to freeze when training 
            the Active Learning part, and unfreeze when re-training the network.
        """
        for param in self.base_model.parameters():
            param.requires_grad = requires_grad


# Takes in a Dataset and labeled/unlabeled index lists. Creates a dataset with the same data
# but uses belonging to the labeled list as a 0/1 ground truth.
class DiscriminativeDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, labeled_idx, unlabeled_idx):
        self.dataset = dataset
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self.total_size = len(self.labeled_idx) + len(self.unlabeled_idx)

    def __getitem__(self, index):
        if index < len(self.labeled_idx):
            inputs, targets = self.dataset.__getitem__(self.labeled_idx[index])
            return inputs, 1
        else:
            inputs, targets = self.dataset.__getitem__(
                self.unlabeled_idx[index - len(self.labeled_idx)])
            return inputs, 0
        
    def __len__(self):
        return self.total_size


def training_dataset_to_features(net, device, dataset, hdf5_file_path):
    """ Runs training dataset through net and saves resulting features into a hdf5 file.
    """
    batch_size = 128
    train_loader = DataLoader(
        dataset, batch_size, shuffle=False, num_workers=4)
    progress = tqdm.tqdm(
        enumerate(train_loader),
        "Creating hdf5 dataset with image features", total=len(train_loader))
    with h5py.File(hdf5_file_path, 'w', libver='latest', swmr=True) as f:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in progress:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                al_features = net.get_discriminative_al_features()
                features = [torch.flatten(f, 1) for f in al_features]
                features_flat = torch.cat(features, dim=1)
                for i in range(features_flat.shape[0]):
                    matrix_to_hdf5(f, features_flat[i].cpu(),
                        "features_{}".format(batch_size * batch_idx + i))


def train_discriminative_al(net, device, train_dataset, labeled_idx, unlabeled_idx,
        hdf5_dataset_path, total_image_count):
    """ Trains Discriminative Active Learning, the part which classifies image as labeled/unlabeled.
        Creates an hdf5 file with features of the main network.
    Args:
        net - The (trained) base model.
        train_dataset - Whole training dataset.
        labeled_idx - Indices of the labeled dataset entries.
        unlabeled_idx - Indices of the unlabeled dataset entries. This can be a selected subset of all unlabeled examples in the dataset of size 10*len(labeled_idx).
    Returns:
        A trained discriminative network.
    """
    training_dataset_to_features(net, device, train_dataset, hdf5_dataset_path)

    optimizer = optim.Adam(net.parameters(), lr=0.0003)
    
    total = len(labeled_idx) + len(unlabeled_idx)
    class_weights = torch.FloatTensor(
        [len(labeled_idx) / total, len(unlabeled_idx) / total]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    hdf5_dataset = HDF5Dataset(hdf5_dataset_path,
        image_ids=["features_{}".format(i) for i in range(total_image_count)]) 
    discriminative_dataset = DiscriminativeDataset(hdf5_dataset, labeled_idx, unlabeled_idx)
    train_loader = DataLoader(
            discriminative_dataset, batch_size=128,
            shuffle=True, num_workers=4)
    discriminative_model = DiscriminativeActiveLearning(
        base_model=None, input_shapes=net.get_discriminative_al_layer_shapes())
    discriminative_model = discriminative_model.to(device)
    for epoch in range(500):
        print('Discriminative AL Epoch: %d' % epoch)
        discriminative_model.train()
        train_loss = 0
        correct = 0
        total = 0
        progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (inputs, targets) in progress:
            # HDF5Dataset class is designed to also return masks for each image,
            # So it returns a dict, we need to take the 'image' part.
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # We don't use the output predictions here, only the classifier output.
            # If we have pre-computed the network features for the whole dataset, and 
            # discriminative model does not have the main model in it, outputs==inputs.
            outputs, labeled_unlabeled_predictions = discriminative_model(inputs)
            loss = criterion(labeled_unlabeled_predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            labeled_unlabeled_predictions = nn.Softmax(dim=1)(labeled_unlabeled_predictions)
            total += targets.size(0)
            for i, target in enumerate(targets):
                if labeled_unlabeled_predictions[i][target] > 0.5:
                    correct += 1

            progress.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # Early stopping, stop if prediction accuracy is above 98%.
        if correct / total >= 0.982:
            break
    return discriminative_model



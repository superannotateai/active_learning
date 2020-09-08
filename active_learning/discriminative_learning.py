''' Discriminative Active learning wrapper model for pytoch Models.
    Implement function get_discriminative_active_learning_layers() and pass your model to
    constructor of the DiscriminativeActiveLearning class.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import numpy as np
import tqdm


class DiscriminativeActiveLearning(nn.Module):
    
    def __init__(self, base_model):
        """ 
        Params:
            base_model - The model being trained. Used for image features representation.
            input_shape - Shape of the input images. Used for selection of perceptron width.
        """
        super(DiscriminativeActiveLearning, self).__init__()
        self.base_model = base_model
        self.input_shapes = base_model.get_discriminative_al_layer_shapes()
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
        base_model_output = self.base_model.forward(x)
        active_learning_features = self.base_model.get_discriminative_al_features()
        # active_learning_features = [f.detach() for f in active_learning_features]
        features = [torch.flatten(f, 1) for f in active_learning_features]
        features_flat = torch.cat(features, dim=1);
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


def train_discriminative_al(net, device, train_dataset, labeled_idx, unlabeled_idx):
    """  Trains Discriminative Active Learning, the part which classifies image as labeled/unlabeled.
    Args:
        net(DiscriminativeActiveLearning) - The (trained) model with Active Learning part included.
        train_dataset - Whole training dataset.
        labeled_idx - Indices of the labeled dataset entries.
        unlabeled_idx - Indices of the unlabeled dataset entries. This can be a selected subset of all unlabeled examples in the dataset of size 10*len(labeled_idx).
    """
    # Freeze layers of the main algorithm, train only the active learning classifier.
    net.freeze_main_layers(requires_grad=False)

    optimizer = optim.Adam(net.parameters(), lr=0.0003)
    
    total = len(labeled_idx) + len(unlabeled_idx)
    class_weights = torch.FloatTensor(
        [len(labeled_idx) / total, len(unlabeled_idx) / total]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    discriminative_dataset = DiscriminativeDataset(train_dataset, labeled_idx, unlabeled_idx)
    train_loader = DataLoader(
            discriminative_dataset, batch_size=128,
            shuffle=True, num_workers=2)
    for epoch in range(5): # 5
        print('Discriminative AL Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (inputs, targets) in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # We don't use the output predictions here, only the classifier output.
            outputs, labeled_unlabeled_predictions = net(inputs)
            # print("targets = {}".format(targets))
            # print("predictions = {}".format(labeled_unlabeled_predictions));
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
    # Unfreeze layers of the main algorithm. Not really required.
    net.freeze_main_layers(requires_grad=True)



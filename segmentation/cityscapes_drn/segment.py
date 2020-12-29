#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
from os.path import exists, join, split
import threading
import itertools

import time

import numpy as np
import scipy
import shutil
import tqdm

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
from torch.autograd import Variable

from skimage.segmentation import watershed
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage import measure

import drn
import data_transforms

sys.path.append(os.path.abspath('../../active_learning'))
from active_learning import ActiveLearning
from active_loss import LossPredictionLoss
from active_learning_utils import *
from discriminative_learning import *

from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage.morphology import distance_transform_edt as edt


try:
    from modules import batchnormsync
except ImportError:
    pass

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False, add_dropout=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000, remove_last_2_layers=True,
            add_dropout=add_dropout)

        # Remember channel sizes for the active learning.
        self.channels = list(model.get_active_learning_feature_channel_counts())
        # Adding 2 more layers for the active learning.
        self.channels.append(classes)
        self.channels.append(classes)

        pmodel = nn.DataParallel(model, device_ids=[0])
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = model
        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        self.active_learning_features = self.base.get_active_learning_features()
        x = self.seg(x)
        self.active_learning_features.append(x)
        y = self.up(x)
        self.active_learning_features.append(self.softmax(y))
        return self.softmax(y), x

    def get_active_learning_feature_channel_counts(self):
        return self.channels

    def get_active_learning_features(self):
        #print("Active learning feature Shapes are ================")
        #for f in self.active_learning_features:
        #    print(f.size())
        return self.active_learning_features

    def get_discriminative_al_layer_shapes(self):
        # All we have is one flat tensor of size 512.
        return self.base.get_discriminative_al_layer_shapes()

    def get_discriminative_al_features(self):
        return self.base.get_discriminative_al_features()

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False, include_instance_gt_masks=False, al_cycle=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.label_instance_list = None
        self.bbox_list = None
        self.al_cycle = al_cycle
        self.read_lists()
        self.include_instance_gt_masks = include_instance_gt_masks


    # Returns path to ground truth mask 'index'.
    def get_gt_path(self, index):
        return join(self.data_dir, self.label_list[index])

    def __getitem__(self, index):
        image_path = join(self.data_dir, self.image_list[index])
        data = [Image.open(image_path)]
        if self.label_list is not None:
            mask_path = join(self.data_dir, self.label_list[index])
            # Change the ground truth mask if required. Used for preprocessed, partially annotated
            # ground truth masks.
            if self.al_cycle is not None:
                mask_path = (mask_path + "_cycle_{}.png").format(self.al_cycle)
            data.append(Image.open(mask_path))
            if self.include_instance_gt_masks:
                # Also return the instance mask.
                instance_mask_path = join(self.data_dir, self.label_instance_list[index])
                # print("Returning instance mask {}".format(instance_mask_path))
                data.append(Image.open(instance_mask_path))

        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(item()[0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            self.label_instance_list = [
                line.replace("trainIds.png", "instanceIds.png") for line in self.label_list]
            assert len(self.image_list) == len(self.label_list)

    # Needed for writing csv files to be uploaded to annotate.online.
    def get_image_path(self, index):
        return self.image_list[index]


class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = item().size
        if self.label_list is not None:
            data.append(Image.open(
                join(self.data_dir, self.label_list[index])))
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(item().resize((int(w * s), int(h * s)),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


def validate(val_loader, model, criterion, eval_score=None, print_freq=40, num_classes=1000,
             use_loss_prediction_al=False, use_discriminative_al=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()
    mAP = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    for i, (input, target) in enumerate(val_loader):
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        if use_loss_prediction_al or use_discriminative_al:
            output = model(input_var)[0][0]
        else:
            output = model(input_var)[0]
        loss = criterion(output, target_var).mean()

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        if eval_score is not None:
            score.update(eval_score(output, target_var), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        _, pred = torch.max(output, 1)
        pred = pred.cpu().data.numpy()
        label = target.cpu().numpy()
        # Remove the 'background' class and compute the matrix hist, where
        # hist[i][j] is the number of pixels for which ground truth class
        # was i, but predicted j.
        hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
        current_mAP = round(np.nanmean(per_class_iu(hist)) * 100, 2)
        mAP.update(current_mAP)
        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})\t'
                        'mAP {mAP.val:.3f} ({mAP.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score, mAP=mAP))

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg, mAP.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    correct_size = correct.size(0)
    if correct_size == 0:
        return 0
    score = correct.float().sum(0).mul(100.0 / correct_size)
    return score.item()


def train(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=3, use_loss_prediction_al=False, active_learning_lamda=1, 
          use_discriminative_al=False, entropy_superpixels=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # Values used for Loss Prediction Active Learning.
    total_ranked = 0
    correctly_ranked = 0
    criterion_lp = LossPredictionLoss()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda()
        target = target.cuda(non_blocking=True)
        input_var = input # torch.autograd.Variable(input)
        target_var = target # torch.autograd.Variable(target)

        # compute output
        if use_loss_prediction_al:
            if epoch < 150:
                output, loss_pred = model(input_var)
            else:
                output, loss_pred = model(input_var, detach_lp=True)
            output = output[0]
        elif use_discriminative_al:
            output, labeled_unlabeled_predictions = model(input_var)
        else:
            output = model(input_var)[0]

        if entropy_superpixels:
            # This is for partially annotated images. 250 marks a pixel of the image,
            # which is not yet annotated, I.E. we don't have the ground truth for it
            # and must not compute any loss for it.
            loss = criterion(output, target_var)
        else:
            loss = criterion(output, target_var)
        # Compute means from shape [N, W, H] to [N].
        loss = loss.mean([1, 2])
        # Let the main model "warm-up" for 1 epoch, loss prediction does not
        # work well otherwise.
        if use_loss_prediction_al and epoch > 1:
            loss_prediction_loss = criterion_lp(loss_pred, loss)
            # Also compute (an estimate) of the ranking accuracy for the training set.
            batch_size = loss.shape[0]
            for l1 in range(batch_size):
                for l2 in range(l1):
                    total_ranked += 1
                    if (loss[l1] - loss[l2]) * (loss_pred[l1] - loss_pred[l2]) > 0:
                        correctly_ranked += 1
            if i % print_freq == 0:
                logger.info(
                    "loss.mean() = {} active_learning_lamda = {}, loss_prediction_loss = {}".format(
                        loss.mean(), active_learning_lamda, loss_prediction_loss));
            loss = loss.mean() + active_learning_lamda * loss_prediction_loss
        else:
            loss = loss.mean()

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        if eval_score is not None:
            scores.update(eval_score(output, target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('{0} Epoch: [{1}][{2}/{3}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.6f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'
                        'Ranking accuracy estimate ({ranking_accuracy})'.format(
                get_algorithm_name(use_loss_prediction_al, use_discriminative_al, None),
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores,
                ranking_accuracy=correctly_ranked/(total_ranked+0.00001)))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Counts the total number of instances in all the training images.
# Returns 94372.
def count_total_instances_number(dataset):
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=32,
            pin_memory=True, drop_last=True)
    superpixels = []
    progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
            desc="Counting object instances in the whole training dataset.")
    total_count = 0
    for i, (input, target, instances_target) in progress:
        total_count += (np.unique(instances_target)).shape[0]
    return total_count


# TODO(martun): change this to our superpixels algorithm once done.
def get_all_superpixels(dataset, superpixels_per_image):
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=2,
            pin_memory=True, drop_last=False)
    superpixels = []
    progress = tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
            desc="Finding superpixels for the whole dataset.")
    for i, (input, target, instances_target) in progress:
        for j in range(input.shape[0]):
            input_image = input.cpu().detach().numpy()[j] # Shape (3, 1024, 2048)
            input_image = np.swapaxes(input_image, 0, 2) # Shape (2048, 1024, 3)
            input_image = np.swapaxes(input_image, 0, 1) # Shape (1024, 2048, 3)
            gradient = sobel(rgb2gray(input_image))
            segments_watershed = watershed(
                gradient, markers=superpixels_per_image, compactness=0.001)
            superpixels.append(segments_watershed)
    return superpixels


def preprocess_data(dataset, all_superpixels, labeled_superpixels,
        al_cycle, num_workers=16):
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(
        PartiallyLabeledDataset(dataset, all_superpixels,
            labeled_superpixels, None),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, drop_last=False
        )
    # Tensorboard summary writer.
    writer = SummaryWriter()
    for batch_idx, (input, target) in enumerate(train_loader):
        for image_idx in range(input.shape[0]):
            image_index = batch_idx * batch_size + image_idx
            
            gt_path = dataset.get_gt_path(image_index)
            single_target_np = target[image_idx].cpu().detach().numpy()
            single_target = Image.fromarray(single_target_np.astype(np.uint8))
            new_gt_path = (gt_path + "_cycle_{}.png").format(al_cycle)
            single_target.save(new_gt_path)
            if image_index % 300 == 0:
                single_target_np = single_target_np.squeeze()
                current_image = input[image_idx].cpu().detach().numpy()
                current_image_2 = current_image
                current_image_2 = np.swapaxes(current_image_2, 0, 2) # Shape (3, 2048, 1024)
                current_image_2 = np.swapaxes(current_image_2, 1, 2) # Shape (3, 1024, 2048)
                writer.add_image('image_annotated_FULL_{}'.format(image_index), 
                    current_image_2, al_cycle)
                current_image[single_target_np == 255] = 0
                # Change current_image from HWC, to CHW.
                current_image = np.swapaxes(current_image, 0, 2) # Shape (3, 2048, 1024)
                current_image = np.swapaxes(current_image, 1, 2) # Shape (3, 1024, 2048)
                writer.add_image('image_annotated_part_{}'.format(image_index), 
                    current_image, al_cycle)
    writer.close()


class PartiallyLabeledDataset(torch.utils.data.Dataset):
    """ Creates a dataset with partially annotated images. Only objects which have intersection with given superpixels are annotated.
    """

    def __init__(self, dataset, superpixels, labeled_superpixels, transforms):
        """
        Parameters:
            dataset: Base dataset, each entry is (input, target, instance_target).
            superpixels: Array of masks, where each mask marks the superpixels of an image.
            labeled_superpixels: Map from image number to a list of superpixels which are labeled by annotators. Only objects which intersect with these superpixels are considered annotated.
        """
        self.dataset = dataset
        self.superpixels = superpixels
        self.labeled_superpixels = {}
        self.transforms = transforms
        print("Labeled superpixels are: {}".format(labeled_superpixels))
        for image_id, superpixel_id in labeled_superpixels:
            if image_id in self.labeled_superpixels.keys():
                self.labeled_superpixels[image_id].append(superpixel_id)
            else:
                self.labeled_superpixels[image_id] = [superpixel_id]

    def get_gt_path(self, index):
        return self.dataset.get_gt_path(index)

    def __getitem__(self, index):
        input, target, instances_target = self.dataset.__getitem__(index)
        # Remove objects from 'target', which must not be annotated, I.E. the ones which do not 
        # intersect any superpixel from self.labeled_superpixels.
       
        # No superpixel annotated in a given image, return an empty mask with all pixels set to
        # 255 (background).
        if index not in self.labeled_superpixels.keys():
            #print("Index {} not in keys {}".format(index, self.labeled_superpixels.keys()))
            target = np.full(target.shape, 255)
        else:
            # 1. Create a mask of annotated superpixels.
            #print("Labeled superpixels {} for image {}".format(self.labeled_superpixels[index], index))
            f = lambda x: x not in self.labeled_superpixels[index]
            vf = np.vectorize(f)
            not_annotated_superpixels_mask = vf(self.superpixels[index])

            # 2. Create a mask of instances, which intersect at least 1 annotated superpixel.
            annotated_instances_mask = np.copy(instances_target)
            annotated_instances_mask[not_annotated_superpixels_mask] = 0
            instance_ids = np.unique(annotated_instances_mask)
            #print("Selected instances {} for image {}".format(instance_ids, index))
            #print("instances_target = {}".format(instances_target))
            f2 = lambda x: x not in instance_ids
            vf2 = np.vectorize(f2)
            not_annotated_pixels_mask = vf2(instances_target)
            #print("not_annotated_pixels_mask = {}".format(not_annotated_pixels_mask))
            
            # 3. Apply created mask to target. 250 Will stand for "ignore", I.E. for not 
            #    counting any loss for given pixel, no matter what the prediction is.
            #print("Mean target value before {}".format(np.mean(target.cpu().detach().numpy())))
            target[not_annotated_pixels_mask] = 255 # 250
            #print("Mean target value after {}".format(np.mean(target.cpu().detach().numpy())))
            target = target.cpu().detach().numpy()
       
        # Convert input image and ground truth target to Pillow images, so we can use the
        # transformations the exacy same way as it was used before.
        input = input.cpu().detach().numpy() # Shape (3, 1024, 2048)
        input = np.swapaxes(input, 0, 2) # Shape (2048, 1024, 3)
        input = np.swapaxes(input, 0, 1) # Shape (1024, 2048, 3)
        if self.transforms is None:
            return torch.from_numpy(input), torch.from_numpy(target)
        input, target = Image.fromarray(input.astype(np.uint8)), Image.fromarray(target.astype(np.uint8))
        # Apply the augmentation over the image and partially annotated ground truth.
        input, target = list(self.transforms(input, target)) 
        return input, target

    def __len__(self):
        return len(self.dataset)


def compute_uncertainty_map(model, inputs, device, mc_dropout=False, prediction_steps_count=10,
        use_variance=False):
    """ Computes and returns some uncertainty map for a given batch.
    """
    inputs = inputs.to(device)
    if not mc_dropout:
        out = model(inputs)[0]
        # Summing over axis=1, which has size C=19, resulting to shape [batch_size, 1024, 2048]
        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)
        entropy = -torch.sum(softmax(out) * log_softmax(out), axis=1)

        # Move the entropy values to cpu to use 'measure.regionprops'.
        entropy = entropy.cpu().detach().numpy()
        return entropy
    else:
        # We can not keep these tensors on GPUs, will run out of memory, moving to CPU.
        predictions_batches = []
        for i in range(prediction_steps_count):
            predictions_batches.append(model(inputs)[0].cpu().numpy())
        predictions = np.stack(predictions_batches, axis=1)
        #print("predictions.shape = {}".format(predictions.shape))
        if use_variance:
            # Predictions has shape (Batch#, prediction_steps_count, 19, 1024, 2048)
            X = np.var(predictions, axis=1)
            # X has shape (Batch#, 19, 1024, 2048)
            return np.mean(X, axis=1)
        else:
            out = np.mean(predictions, axis=1)
            entropy = -np.sum(
                scipy.special.softmax(out, axis=1) * scipy.special.log_softmax(out, axis=1),
                axis=1)
            return entropy


def choose_new_labeled_indices_with_highest_uncertainty(
        model, cycle, rand_state, unlabeled_idx, dataset, device, subset_factor=10,
        mc_dropout=False, images_per_cycle=150, use_variance_as_uncertainty=True):
    if cycle == 0:
        idx = rand_state.choice(unlabeled_idx, images_per_cycle, replace=False)
        for id in idx:
            unlabeled_idx.remove(id)
        return idx, None
    # Tensorboard summary writer.
    writer = SummaryWriter()

    cycle_subs_idx = rand_state.choice(
        unlabeled_idx,
        min(subset_factor * images_per_cycle, len(unlabeled_idx)),
        replace=False)
    cycle_pool = Subset(dataset, cycle_subs_idx)
    cycle_loader = DataLoader(
        cycle_pool, batch_size=4, shuffle=False, num_workers=2
    )

    model.eval()
    image_uncertainties = []
    with torch.no_grad():
        # We must not use the targets here, assume we don't have them.
        for batch_idx, (inputs, _targets) in enumerate(cycle_loader):
            uncertainties_for_batch = compute_uncertainty_map(
                model, inputs, device, mc_dropout=mc_dropout,
                use_variance=use_variance_as_uncertainty)
            # Now we need to compute the mean uncertainty for each superpixel.
            for i in range(uncertainties_for_batch.shape[0]):
                image_index = batch_idx * uncertainties_for_batch.shape[0] + i
                image_uncertainties.append([image_index, np.mean(uncertainties_for_batch[i])])
                if image_index % 50 == 0:
                    writer.add_image('image_uncertainties_{}'.format(image_index),
                            np.expand_dims(uncertainties_for_batch[i], axis=0), cycle)
    
    # Sort by intensity value.
    image_uncertainties.sort(key=lambda x: x[1])
    new_images = [image_index for image_index, intensity in
        image_uncertainties[-images_per_cycle:]]
    entropies = [intensity for image_index, intensity in
        image_uncertainties[-images_per_cycle:]]

    new_labeled_idx = []
    for id in new_images:
        new_labeled_idx.append(cycle_subs_idx[id])
        unlabeled_idx.remove(cycle_subs_idx[id])

    for idx, entropy in enumerate(entropies):
        writer.add_scalar("Entropy_superpixels_chosen_cycle_{}".format(cycle), entropy, idx)
    writer.close()
    return new_labeled_idx, entropies


def choose_superpixels_with_highest_entropies(
        model, cycle, rand_state, unlabeled_superpixels, training_dataset_no_augmentation,
        device, criterion, all_superpixels, instances_per_cycle, mc_dropout=False):
    # Tensorboard summary writer.
    writer = SummaryWriter()

    cycle_loader = DataLoader(training_dataset_no_augmentation, batch_size=4,
                              shuffle=False, num_workers=32)
    model.eval()
    superpixels_with_entropies = []
    with torch.no_grad():
        # We must not use the targets here, assume we don't have them.
        for batch_idx, (inputs, _targets, _instance_targets) in enumerate(cycle_loader):
            uncertainties_for_batch = compute_uncertainty_map(
                model, inputs, device, mc_dropout=mc_dropout,
                use_variance=args.use_variance_as_uncertainty)
            # Now we need to compute the mean uncertainty for each superpixel.
            for i in range(uncertainties_for_batch.shape[0]):
                image_index = batch_idx * uncertainties_for_batch.shape[0] + i
                uncertainty_map = uncertainties_for_batch[i]
                if image_index % 50 == 0:
                    writer.add_image('image_uncertainties_{}'.format(image_index),
                            np.expand_dims(uncertainty_map, axis=0), cycle)
                
                # Apply Euclidean Distance Transform with threshold 0.5.
                # transform = edt((uncertainty_map > 1).astype('uint8'))
                # uncertainty_map *= transform
                # if image_index % 100 == 0:
                #     writer.add_image('image_entropies_distance_transformed_{}'.format(image_index),
                #             np.expand_dims(uncertainty_map, axis=0), cycle)
                regions = measure.regionprops(all_superpixels[image_index],
                    intensity_image=uncertainty_map)
                for reg in regions:
                    superpixels_with_entropies.append([image_index, reg.label, reg.mean_intensity])
    # Sort by intensity value.
    superpixels_with_entropies.sort(key=lambda x: x[2]) 
    new_superpixels = [(image_index, superpixel_id) for image_index, superpixel_id, intensity in 
        superpixels_with_entropies[-instances_per_cycle:]]
    entropies = [intensity for image_index, superpixel_id, intensity in 
        superpixels_with_entropies[-instances_per_cycle:]]

    for idx, entropy in enumerate(entropies):
        writer.add_scalar("Entropy_superpixels_chosen_cycle_{}".format(cycle), entropy, idx)
    writer.close()
    return new_superpixels, entropies


def train_seg(args):
    rand_state = np.random.RandomState(1311)
    torch.manual_seed(1311)
    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

    # We have 2975 images total in the training set, so let's choose 150 for 10 cycles,
    # 1500 images total (~1/2 of total)
    images_per_cycle = 150

    # There are a total of 94,372 object instances in the training dataset, so each image
    # on average has 31.7 instances. 31.7 * 150 = 4755, so 4755 object instances will be annotated
    # per cycle, startinig with images_per_cycle = 150 fully annotated images on cycle 0.
    instances_per_cycle = 4755

    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = data_transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    if args.random_rotate > 0:
        t.append(data_transforms.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(data_transforms.RandomScale(args.random_scale))
    t.extend([data_transforms.RandomCrop(crop_size),
              data_transforms.RandomHorizontalFlip(),
              data_transforms.ToTensor(),
              normalize])
    dataset = SegList(data_dir, 'train', data_transforms.Compose(t),
                      list_dir=args.list_dir, 
                      include_instance_gt_masks=args.entropy_superpixels)
    training_dataset_no_augmentation = SegList(
        data_dir, 'train',
        data_transforms.Compose([data_transforms.ToTensor(), normalize]),
        list_dir=args.list_dir,
        include_instance_gt_masks=args.entropy_superpixels
    )

    # In case of working with fully annotated images.
    labeled_idx = []
    unlabeled_idx = list(range(len(dataset)))

    if args.entropy_superpixels:
        # In case of working with partially annotated images, I.E. only some superpixels
        # are annotated.
        labeled_superpixels = []
        unlabeled_superpixels = list(itertools.product(
            range(len(dataset)), range(args.superpixels_per_image)))
        #total_instances_count = count_total_instances_number(
        #        training_dataset_no_augmentation)
        # print("Total number of object instances in the training dataset is {}.".format(
        #    total_instances_count))

        # Precompute superpixels once, use later on.
        # all_superpixels = get_all_superpixels(
        #    training_dataset_no_augmentation, args.superpixels_per_image)
        # Save superpixels information to a file, so we will not re-create if every time.
        #pickle.dump(all_superpixels, open("all_superpixels_2975.pkl", "wb"))
        all_superpixels = pickle.load(open("all_superpixels_2975.pkl", "rb"))

    validation_accuracies = list()
    validation_mAPs = list()
    progress = tqdm.tqdm(range(10))
    model = None
    for cycle in progress:
        if args.entropy_superpixels:
            # Here we select superpixels, not images. Each entry of new_superpixels will be a tuple
            # of 2 values, (image_number, superpixel #). Images annotated on the first cycle
            # the first [150] will be completely annotated, I.E. all of their superpixels will be 
            # present in the list.
            if cycle == 0:
                images_chosen = rand_state.choice(
                        range(len(dataset)), images_per_cycle, replace=False)
                new_superpixels = list(itertools.product(
                    images_chosen,
                    range(args.superpixels_per_image)))
                entropies = None
            else:
                if args.random_superpixels:
                    new_superpixel_indices = rand_state.choice(
                        range(len(unlabeled_superpixels)), instances_per_cycle, replace=False)
                    new_superpixels = np.array(unlabeled_superpixels)[
                            new_superpixel_indices.tolist(), :].tolist()
                    entropies = None
                else:
                    new_superpixels, entropies = choose_superpixels_with_highest_entropies(
                        model, cycle, rand_state, unlabeled_superpixels,
                        training_dataset_no_augmentation,
                        device, criterion, all_superpixels, instances_per_cycle,
                        mc_dropout=args.mc_dropout)
            labeled_superpixels.extend(new_superpixels)
            unlabeled_superpixels = [x for x in unlabeled_superpixels if x not in new_superpixels]
        elif args.choose_images_with_highest_loss:
            # Choosing images based on the ground truth labels. 
            # We want to check if predicting loss with 100% accuracy would result to
            # a good active learning algorithm.
            new_indices, entropies = choose_new_labeled_indices_using_gt(
                    model, cycle, rand_state, unlabeled_idx, training_dataset_no_augmentation,
                    device, criterion, images_per_cycle)
            labeled_idx.extend(new_indices)
        elif args.choose_images_with_highest_uncertainty:
            new_indices, entropies = choose_new_labeled_indices_with_highest_uncertainty(
                    model, cycle, rand_state, unlabeled_idx, training_dataset_no_augmentation,
                    device, mc_dropout=True, images_per_cycle=images_per_cycle,
                    use_variance_as_uncertainty=args.use_variance_as_uncertainty)
            labeled_idx.extend(new_indices)
        else:
            new_indices, entropies = choose_new_labeled_indices(
                model, training_dataset_no_augmentation, cycle, rand_state,
                labeled_idx, unlabeled_idx, device, images_per_cycle,
                args.use_loss_prediction_al, args.use_discriminative_al, input_pickle_file=None)
            labeled_idx.extend(new_indices)

        if args.entropy_superpixels:
            print("Running on {} labeled superpixels.".format(len(labeled_superpixels)))
        else:
            print("Running on {} labeled images.".format(len(labeled_idx)))

        if args.output_superannotate_csv_file is not None:
            if entropies is None:
                entropies = np.zeros(new_indices.shape)
            # Write image paths to csv file which can be uploaded to annotate.online.
            write_entropies_csv(
                training_dataset_no_augmentation, new_indices,
                entropies, args.output_superannotate_csv_file)

        if args.entropy_superpixels:
            # For superpixel based algorithm we load all the images which have at least 1 superpixel
            # annotated.
            partially_labeled_idx = list(set(
                [image_id for (image_id, superpixel_id) in labeled_superpixels]))
            print("Running on {} partially labeled images".format(len(partially_labeled_idx)))
            dataset_local = SegList(
                data_dir, 'train',
                data_transforms.Compose([data_transforms.ToTensor()]),
                list_dir=args.list_dir,
                include_instance_gt_masks=True
            )
            # Run over the ground truths and remove all unlabeled objects.
            preprocess_data(dataset_local, all_superpixels, labeled_superpixels, al_cycle=cycle)

            # Now set al_cycle variable, so SegList will read proper ground truths.
            dataset_local_augmented = SegList(
                data_dir, 'train',
                data_transforms.Compose(t),
                list_dir=args.list_dir,
                include_instance_gt_masks=False,
                al_cycle=cycle
            )
            train_loader = torch.utils.data.DataLoader(
                data.Subset(dataset_local_augmented, partially_labeled_idx),
                batch_size=batch_size, shuffle=True, num_workers=num_workers,
                pin_memory=True, drop_last=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                data.Subset(dataset, labeled_idx),
                batch_size=batch_size, shuffle=True, num_workers=num_workers,
                pin_memory=True, drop_last=True
            )
        val_loader = torch.utils.data.DataLoader(
            SegList(data_dir, 'val', data_transforms.Compose([
                data_transforms.RandomCrop(crop_size),
                data_transforms.ToTensor(),
                normalize,
            ]), list_dir=args.list_dir),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, drop_last=True
        )

        # Reset the model. 
        single_model = DRNSeg(args.arch, args.classes, None,
                              pretrained=True, add_dropout=args.mc_dropout)
       
        if args.pretrained:
            single_model.load_state_dict(torch.load(args.pretrained))

        # Wrap our model in Active Learning Model.
        if args.use_loss_prediction_al:
            single_model = ActiveLearning(
                single_model, global_avg_pool_size=6, fc_width=256)
        elif args.use_discriminative_al:
            single_model = DiscriminativeActiveLearning(single_model)
        optim_parameters = single_model.optim_parameters()

        model = torch.nn.DataParallel(single_model, device_ids=[0]).cuda()

        # Don't apply a 'mean' reduction, we need the whole loss vector.
        criterion = nn.NLLLoss(ignore_index=255, reduction='none')

        criterion.cuda()

        # define loss function (criterion) and optimizer.
        optimizer = torch.optim.SGD(optim_parameters,
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        cudnn.benchmark = True
        best_prec1 = 0
        best_mAP = 0
        start_epoch = 0

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        if args.evaluate:
            validate(val_loader, model, criterion, eval_score=accuracy,
                     num_classes=args.classes,
                     use_loss_prediction_al=args.use_loss_prediction_al)
            return

        progress_epoch = tqdm.tqdm(range(start_epoch, args.epochs))
        for epoch in progress_epoch:
            lr = adjust_learning_rate(args, optimizer, epoch)
            logger.info('Cycle {0} Epoch: [{1}]\tlr {2:.06f}'.format(cycle, epoch, lr))
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch,
                  eval_score=accuracy, use_loss_prediction_al=args.use_loss_prediction_al, 
                  active_learning_lamda=args.lamda, entropy_superpixels=args.entropy_superpixels)

            # Save the model and reload to disable the inference time dropout.
            if args.mc_dropout:
                checkpoint_path = os.path.join(args.save_path, 'checkpoint_latest.pth.tar')
                torch.save(model.module.state_dict(), checkpoint_path)
                single_model_no_dropout = DRNSeg(
                    args.arch, args.classes, None,
                    pretrained=True, add_dropout=False)
                validation_model = torch.nn.DataParallel(
                    single_model, device_ids=[0]).cuda()
                validation_model.module.load_state_dict(torch.load(checkpoint_path))
            else:
                validation_model = model

            # evaluate on validation set
            prec1, mAP1 = validate(val_loader, validation_model, criterion, eval_score=accuracy,
                             num_classes=args.classes,
                             use_loss_prediction_al=args.use_loss_prediction_al)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_mAP = max(mAP1, best_mAP)
            checkpoint_path = os.path.join(args.save_path, 'checkpoint_latest.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_mAP': best_mAP,
            }, is_best, filename=checkpoint_path)
            if (epoch + 1) % args.save_iter == 0:
                history_path = os.path.join(args.save_path, 'checkpoint_{:03d}.pth.tar'.format(epoch + 1))
                shutil.copyfile(checkpoint_path, history_path)
        validation_accuracies.append(best_prec1)
        validation_mAPs.append(best_mAP)
        print("{} accuracies: {} mAPs {}".format(
            "Active Learning" if args.use_loss_prediction_al else "Random",
            str(validation_accuracies),
            str(validation_mAPs)))
        # Compute histogram of loss values for the unlabeled part of training dataset.
        # Uncomment next lines if you want to check the loss distribution.
        # loss_value_histogram(
        #     model, cycle, rand_state, unlabeled_idx,
        #     training_dataset_no_augmentation, device, criterion)
        # loss_value_min_max_average(
        #     model, cycle, rand_state, unlabeled_idx,
        #     dataset, device, criterion)


def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)


def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    for iter, (image, label, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)
        final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(
                pred, name, output_dir + '_color',
                TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    # workers = [threading.Thread(target=resize_one, args=(i, j))
    #            for i in range(tensor.size(0)) for j in range(tensor.size(1))]

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         out[i, j] = np.array(
    #             Image.fromarray(tensor_cpu[i, j]).resize(
    #                 (w, h), Image.BILINEAR))
    # out = tensor.new().resize_(*out.shape).copy_(torch.from_numpy(out))
    return out


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]
        h, w = input_item().size()[2:4]
        images = [input_item()]
        images.extend(input_data[-num_scales:])
        # pdb.set_trace()
        outputs = []
        for image in images:
            image_var = Variable(image, requires_grad=False, volatile=True)
            final = model(image_var)[0]
            outputs.append(final.data)
        final = sum([resize_4d_tensor(out, w, h) for out in outputs])
        # _, pred = torch.max(torch.from_numpy(final), 1)
        # pred = pred.cpu().numpy()
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALETTE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model, device_ids=[0]).cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = data_transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    if args.ms:
        dataset = SegListMS(data_dir, phase, data_transforms.Compose([
            data_transforms.ToTensor(),
            normalize,
        ]), scales, list_dir=args.list_dir)
    else:
        dataset = SegList(data_dir, phase, data_transforms.Compose([
            data_transforms.ToTensor(),
            normalize,
        ]), list_dir=args.list_dir, out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        mAP = test(test_loader, model, args.classes, save_vis=True,
                   has_gt=phase != 'test' or args.with_gt, output_dir=out_dir)
    logger.info('mAP: %f', mAP)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                        help='output path for training checkpoints')
    parser.add_argument('--save_iter', default=1, type=int,
                        help='number of training iterations between'
                             'checkpoint history saves')
    parser.add_argument('-j', '--workers', type=int, default=32)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    parser.add_argument('--use-loss-prediction-al',
                        dest='use_loss_prediction_al',
                        default=False, type=bool,
                        help='If True, will use loss prediction active learning algorithm.')
    parser.add_argument('--choose-images-with-highest-uncertainty',
                        dest='choose_images_with_highest_uncertainty',
                        default=False, type=bool,
                        help='If True, will use choose images with highest uncertainty scores.')
    parser.add_argument('--choose_images_with_highest_loss',
                        dest='choose_images_with_highest_loss',
                        default=False, type=bool,
                        help='If True, will use ground truth labels to select the images with highest loss.')
    parser.add_argument('--lamda', default=1, type=float,
                        help='Loss prediction active learning loss weight')
    parser.add_argument('--use-discriminative-al',
                        dest='use_discriminative_al',
                        default=False, type=bool,
                        help='If True, will use discriminative active learning algorithm.')
    parser.add_argument('--entropy-superpixels',
                        dest='entropy_superpixels',
                        default=False, type=bool,
                        help='If True, will select image superpixels based on entropy value.')
    parser.add_argument('--random-superpixels',
                        dest='random_superpixels',
                        default=False, type=bool,
                        help='If True, will select random superpixels instead of those with high entropy.')
    parser.add_argument('--mc-dropout',
                        dest='mc_dropout',
                        default=False, type=bool,
                        help='If True, will run inference multiple times and take variance as uncertainty score.')


    parser.add_argument('--superpixels-per-image',
                        dest='superpixels_per_image',
                        default=100, type=int,
                        help='Number of superpixels per image.')
    parser.add_argument('--output_superannotate_csv_file',
                        required=False,
                        type=str,
                        default=None,
                        help='Path to the output csv file with the selected indices. Can be uploaded to annotate.online.')
    parser.add_argument('--use-variance-as-uncertainty',
                        dest='use_variance_as_uncertainty',
                        default=True, type=bool,
                        help='If True, will use variance as uncertainty value, else will use entropy.')

    args = parser.parse_args()

    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args


def main():
    args = parse_args()
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)


if __name__ == '__main__':
    main()

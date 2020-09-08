''' Loss functions for active Learning.
'''

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LossPredictionLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(LossPredictionLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target):
        input = (
            input - input.flip(0)
        )[:len(input) // 2
         ]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()
        one = 2 * torch.sign(
            torch.clamp(target, min=0)
        ) - 1  # 1 operation which is defined by the authors

        loss = torch.sum(torch.clamp(self.margin - one * input, min=0))
        loss = loss / input.size(
            0
        )  # Note that the size of input is already halved
        return loss

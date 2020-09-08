''' Active learning wrapper model for pytoch Models.
    Implement function get_active_learning_layers() and pass your model to constructor of
    the ActiveLearning class.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActiveLearning(nn.Module):
    
    def __init__(self, base_model, global_avg_pool_size=1, fc_width=128):
        super(ActiveLearning, self).__init__()
        self.base_model = base_model
        self.channel_counts = base_model.get_active_learning_feature_channel_counts()
        self.fc = []
        for channels in self.channel_counts:
            self.fc.append(nn.Linear(
                channels * global_avg_pool_size * global_avg_pool_size, fc_width))

        self.fc = nn.ModuleList(self.fc)
        self.loss_pred = nn.Linear(len(self.channel_counts)*fc_width,1)
        self.avgpool = nn.AdaptiveAvgPool2d((global_avg_pool_size, global_avg_pool_size))

    def forward(self, x, detach_lp = False):
        out_p = self.base_model.forward(x)
        active_learning_features = self.base_model.get_active_learning_features()
        fc_outputs = []
        for idx, features in enumerate(active_learning_features):
            if detach_lp:
                features = features.detach()
            out = self.avgpool(features)
            out = torch.flatten(out, 1)
            out = self.fc[idx](out)
            fc_outputs.append(F.relu(out))
        fc_cat = torch.cat(fc_outputs, dim=1)
        loss_pred = self.loss_pred(fc_cat)
        return out_p, loss_pred

    # Used for segmentation with DRN only.
    def optim_parameters(self):
        for param in self.base_model.optim_parameters():
            yield param
        for param in self.fc.parameters():
            yield param
        for param in self.loss_pred.parameters():
            yield param


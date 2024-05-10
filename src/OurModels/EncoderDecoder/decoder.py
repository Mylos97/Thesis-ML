import torch
import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, TreeActivation, TreeLayerNorm)

class TreeDecoder(nn.Module):
    def __init__(self, output_dim, dropout_prob=0.1) -> None:
        super(TreeDecoder, self).__init__()

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(64, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, output_dim),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU())
        )

        self.linear = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Linear(16, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            MaxNormalize()
        )


    def forward(self, trees, indexes):
        x = self.linear(trees)
        x = x.view(x.shape[0], 64, 64)
        r = self.tree_conv((x, indexes))
        return r

class MaxNormalize(nn.Module):
    def __init__(self):
        super(MaxNormalize, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        max_val = torch.max(torch.abs(x))
        x = self.softmax(x / max_val)
        return x

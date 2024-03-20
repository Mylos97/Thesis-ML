import torch.nn as nn
from torch.nn import Sigmoid
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling, TreeActivation, TreeLayerNorm)

class TreeConvolution256(nn.Module):
    def __init__(self, in_dim) -> None:
        super(TreeConvolution256, self).__init__()
        self.in_dim = in_dim
        self.sigmoid = Sigmoid()
        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.in_dim, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, trees):
        y1 = self.tree_conv(trees[0])
        y2 = self.tree_conv(trees[1])
        diff = y1 - y2
        prob_y = self.sigmoid(diff)
        return prob_y
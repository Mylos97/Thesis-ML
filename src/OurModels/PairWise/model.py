import torch.nn as nn
from torch.nn import Sigmoid
import torch
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling, TreeActivation, TreeLayerNorm)

class Pairwise(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob) -> None:
        super(Pairwise, self).__init__()
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
            nn.Dropout(dropout_prob),
            nn.Linear(32, 1)
        )

    def forward(self, trees):
        tree1, tree2 = trees
        y1 = self.tree_conv(tree1)
        y2 = self.tree_conv(tree2)
        diff = y1 - y2
        prob_y = self.sigmoid(diff)
        prob_y = torch.squeeze(prob_y, dim=1)
        return prob_y
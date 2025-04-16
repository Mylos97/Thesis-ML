import torch
import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)

class CostModel(nn.Module):
    def __init__(self, in_dim, **args) -> None:
        super(CostModel, self).__init__()
        self.tree_conv = nn.Sequential(
            BinaryTreeConv(in_dim, 256),
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
        x = self.tree_conv(trees)
        x = torch.squeeze(x, dim=1)
        return x

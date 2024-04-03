import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)

class TreeConvolution256(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(TreeConvolution256, self).__init__()
        self.tree_conv = nn.Sequential (
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
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, trees):
        return self.tree_conv(trees)
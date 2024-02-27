import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)

class TreeConvolution256(nn.Module):
    def __init__(self, input_feature_dim) -> None:
        super(TreeConvolution256, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.cuda = False
        self.device = None

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim, 256),
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
        return self.tree_conv(trees)
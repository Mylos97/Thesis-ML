import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)

class TreeEncoder(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.1) -> None:
        super(TreeEncoder, self).__init__()

        self.binary_conv = nn.Sequential (
            BinaryTreeConv(input_dim, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
        )

        self.linear = nn.Sequential(
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, 16),
            nn.Dropout(dropout_prob),
        )

    def forward(self, trees):
        x = self.binary_conv(trees)
        y = self.linear(x)
        return y, x[1]
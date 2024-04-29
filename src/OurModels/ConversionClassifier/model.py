import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling, TreeActivation, TreeLayerNorm)

class ConversionClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob) -> None:
        super(ConversionClassifier, self).__init__()
        self.in_dim = in_dim
        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.in_dim, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(64, 32),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(16, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, trees):
        return self.tree_conv(trees)
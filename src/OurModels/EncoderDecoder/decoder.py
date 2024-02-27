import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv,
                                  TreeActivation, TreeLayerNorm)

class TreeDecoder(nn.Module):
    def __init__(self, output_dim) -> None:
        super(TreeDecoder, self).__init__()
        self.output_dim = output_dim

        self.tree_conv = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            BinaryTreeConv(64, 128),
            TreeActivation(nn.LeakyReLU()),
            TreeLayerNorm(),
            BinaryTreeConv(128, 256),
            TreeActivation(nn.LeakyReLU()),
            TreeLayerNorm(),
            BinaryTreeConv(256, self.output_dim)
        )
        

    def forward(self, trees):
        return self.tree_conv(trees)
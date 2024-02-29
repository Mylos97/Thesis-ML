import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv,
                                  TreeActivation, TreeLayerNorm)

class TreeDecoder(nn.Module):
    def __init__(self, output_dim) -> None:
        super(TreeDecoder, self).__init__()
        self.tree_conv = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            BinaryTreeConv(64, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, output_dim), # this prolly does not work atm
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU())
        )
        
    def forward(self, trees):
        return self.tree_conv(trees)
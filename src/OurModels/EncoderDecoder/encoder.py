import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)

class TreeEncoder(nn.Module):
    def __init__(self, input_dim) -> None:
        super(TreeEncoder, self).__init__()
        self.tree_conv = nn.Sequential (
            BinaryTreeConv(input_dim, 256),
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
        )

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


    def forward(self, trees):
        x = self.tree_conv(trees)
        y = self.binary_conv(trees)
        #print("i am from ENCODEr", type(y[0]), type(y[1]))
        #print("i am from ENCODEr", y[0], y[1])
        
        return x, y[1]
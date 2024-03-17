import torch
import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, TreeActivation, TreeLayerNorm)

class TreeDecoder(nn.Module):
    def __init__(self, output_dim) -> None:
        super(TreeDecoder, self).__init__()

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(64, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, output_dim),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU())
        )

        self.linear = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
        )
        
    def forward(self, trees, indexes):
        x = self.linear(trees)
        x = x.view(x.shape[0], 64, 64)
        return self.tree_conv((x, indexes))
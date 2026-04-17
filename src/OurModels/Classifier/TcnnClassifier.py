import torch
import torch.nn as nn

from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)

class TcnnClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, **args) -> None:
        super().__init__()

        self.binary_conv = nn.Sequential(
            BinaryTreeConv(in_dim, 512),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(512, 512),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(512, 512),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(512, 512),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(512, out_dim))

    def forward(self, trees):
        return self.binary_conv(trees)

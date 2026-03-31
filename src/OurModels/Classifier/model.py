import torch
import torch.nn as nn

from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)

class CnnClassifier(nn.module):
    def __init__(self, in_channels, out_channels, **args) -> None:
        self.encoder = Encoder()
        self.head = ClassificationHead(pass, out_channels)

    def forward(self, trees):

class ClassificationHead:
    def __init__(self, in_channels, out_channels, **args) -> None:
        super.__init__(
            DynamicPooling()
            nn.LayerNorm(in_channels)
            nn.Linear(in_channels, out_channels)
        )

class Encoder:  
    def __init__(self, in_channels, out_channels, **args) -> None:
        super.__init__(
            #stem
            BinaryTreeConv(in_channels, out_channels, kernel_size=3)
            TreeLayerNorm()
            #stage
            DownSampler()
            Block()
            Block()
            Block()
            Block()
            Block()
            Block()
            Block()
            Block()
            Block()
        )

class DownSampler:
    def __init__(self, in_channels, out_channels, **args) -> None:
        super.__init__(
            TreeLayerNorm()
            BinaryTreeConv(in_channels, out_channels, kernel_size=3)
        )


class Block:
    def __init__(self, in_channels, out_channels, **args) -> None:
        super.__init__(
            BinaryTreeConv(in_channels, in_channels, kernel_size=3)
            TreeLayerNorm()
            BinaryTreeConv(in_channels, out_channels * 4, kernel_size=1) #pw widening 
            TreeActivation(nn.Mish())
            BinaryTreeConv(out_channels*4, out_channels, kernel_size=1) #pw 
        )

    def forward(x)
        output = self.block(x) + x #add residual
        return output
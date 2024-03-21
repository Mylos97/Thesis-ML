import torch
import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, TreeActivation, TreeLayerNorm)

class TreeDecoder(nn.Module):
    def __init__(self, output_dim, dropout_prob=0.1) -> None:
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

        linear_layers = []
        prev_dim = 16
        num_layers = 8
        hidden_dim = 32
        for _ in range(num_layers):
            linear_layers.append(nn.Linear(prev_dim, hidden_dim))
            linear_layers.append(nn.LeakyReLU())
            linear_layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dim
            hidden_dim *= 2


        self.linear = nn.Sequential(*linear_layers)


    def forward(self, trees, indexes):
        x = self.linear(trees)
        x = x.view(x.shape[0], 64, 64)
        return self.tree_conv((x, indexes))
import torch
import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, TreeActivation, TreeLayerNorm, DynamicPooling)

class TreeDecoder(nn.Module):
    linear_layer_size: int = 4096

    def __init__(self, output_dim, dropout_prob, z_dim) -> None:
        super(TreeDecoder, self).__init__()

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(16, 32),
            TreeLayerNorm(),
            BinaryTreeConv(32, 64),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(64, 128),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(128, 256),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(256, 512),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(512, output_dim),
            TreeLayerNorm(),
            TreeActivation(nn.Mish())
        )

        self.linear = nn.Sequential(
            nn.Linear(z_dim, self.linear_layer_size),
            nn.BatchNorm1d(self.linear_layer_size),
            nn.Mish(),
            nn.Dropout(dropout_prob),
        )

    def forward(self, trees, indexes):
        max_dim_tree = torch.max(indexes)
        next_pow_2 = 1<<(max_dim_tree).item().bit_length()
        x = self.linear(trees)
        assert next_pow_2 != 0
        x = x.view(x.shape[0], int(self.linear_layer_size / next_pow_2), next_pow_2)
        r = self.tree_conv((x, indexes))

        return r


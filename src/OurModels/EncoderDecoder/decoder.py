import torch
import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, TreeActivation, TreeLayerNorm, DynamicPooling)

class TreeDecoder(nn.Module):
    def __init__(self, output_dim, dropout_prob, z_dim) -> None:
        super(TreeDecoder, self).__init__()

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(32, 64),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(64, 128),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(128, 256),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(256, output_dim),
            TreeLayerNorm(),
            TreeActivation(nn.Mish())
        )

        """
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.Mish(),
            nn.Dropout(dropout_prob),
        )

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(32, 64),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(64, 128),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(128, 256),
            TreeLayerNorm(),
            TreeActivation(nn.Mish()),
            BinaryTreeConv(256, output_dim),
            TreeLayerNorm(),
            TreeActivation(nn.Mish())
        )
        """

        self.linear = nn.Sequential(
            nn.Linear(z_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.Mish(),
            nn.Dropout(dropout_prob),
        )

    def forward(self, trees, indexes):
        x = self.linear(trees)
        x = x.view(x.shape[0], 64, 64)
        r = self.tree_conv((x, indexes))

        return r


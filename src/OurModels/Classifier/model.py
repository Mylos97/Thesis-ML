import torch.nn as nn
from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)
from TreeConvolution.util import prepare_trees
from Models.PairWise.helper import (transformer, left_child, right_child)


class TreeConvolution256(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(TreeConvolution256, self).__init__()
        self.input_feature_dim = input_dim
        self.output_dim = output_dim

        self.tree_conv = nn.Sequential (
            BinaryTreeConv(self.input_feature_dim, 256),
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
            nn.LeakyReLU(),
            nn.Linear(16, self.output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, trees):
        return self.tree_conv(trees)

    def build_trees(self, feature):
        return prepare_trees(feature, transformer, left_child, right_child, cuda=self.cuda, device=self.device)
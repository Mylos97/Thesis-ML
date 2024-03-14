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
            BinaryTreeConv(256, output_dim), # this prolly does not work atm
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU())
        )

        self.linear_boi = nn.Sequential(
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
        linear_boi = self.linear_boi(trees)
        l = []

        for vector in linear_boi:
            sublists = [vector[i:i+64].tolist() for i in range(0, len(vector), 64)]
            l.append(sublists)
        
        l = torch.tensor(l)
        return self.tree_conv((l, indexes))
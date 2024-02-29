import torch.nn as nn
from encoder import TreeEncoder
from decoder import TreeDecoder

class TreeAutoEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = TreeEncoder(dim)
        self.decoder = TreeDecoder(dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
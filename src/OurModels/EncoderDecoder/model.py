import torch.nn as nn
from OurModels.EncoderDecoder.encoder import TreeEncoder
from OurModels.EncoderDecoder.decoder import TreeDecoder

class TreeAutoEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = TreeEncoder(dim)
        self.decoder = TreeDecoder(dim)

    def forward(self, x):
        x, indexes = self.encoder(x)
        x = self.decoder(x, indexes)
        return x[0]
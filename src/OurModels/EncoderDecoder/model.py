import torch.nn as nn
from OurModels.EncoderDecoder.encoder import TreeEncoder
from OurModels.EncoderDecoder.decoder import TreeDecoder

class TreeAutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = TreeEncoder(in_dim)
        self.decoder = TreeDecoder(out_dim)

    def forward(self, x, indexes):
        x = (x, indexes)
        xs, indexes = self.encoder(x)
        xt = self.decoder(xs, indexes)

        return xt[0]
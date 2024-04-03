import torch.nn as nn
from OurModels.EncoderDecoder.encoder import TreeEncoder
from OurModels.EncoderDecoder.decoder import TreeDecoder

class TreeAutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob):
        super().__init__()
        self.encoder = TreeEncoder(in_dim, dropout_prob)
        self.decoder = TreeDecoder(out_dim, dropout_prob)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_prob = dropout_prob

    def forward(self, x):
        xs, indexes = self.encoder(x)
        xt = self.decoder(xs, indexes)

        return xt[0]
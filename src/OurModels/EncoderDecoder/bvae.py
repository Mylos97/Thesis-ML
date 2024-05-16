import torch
import torch.nn as nn
from OurModels.EncoderDecoder.encoder import TreeEncoder
from OurModels.EncoderDecoder.decoder import TreeDecoder

class BVAE(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob, z_dim):
        super().__init__()
        self.mu = nn.Linear(z_dim, z_dim)
        self.log_var = nn.Linear(z_dim, z_dim)
        self.encoder = TreeEncoder(in_dim, dropout_prob)
        self.decoder = TreeDecoder(out_dim, dropout_prob)
        self.training = False
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if not self.training:
            encoded, indexes = self.encoder(x)
            z = self.mu(encoded)
            decoded = self.decoder(z, indexes)
            x = self.softmax(decoded[0])
            return x
        else:
            encoded, indexes = self.encoder(x)
            mean = self.mu(encoded)
            log_var = self.log_var(encoded)
            batch, dim = mean.shape
            epsilon = torch.randn(batch, dim)
            z = mean + torch.exp(0.5 * log_var) * epsilon
            decoded = self.decoder(z, indexes)
            x = self.softmax(decoded[0])
            return [x, mean, log_var]
import torch
import torch.nn as nn
from OurModels.EncoderDecoder.encoder import TreeEncoder
from OurModels.EncoderDecoder.decoder import TreeDecoder

class VAE(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob):
        super().__init__()            
        self.num_hidden = 16
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_prob = dropout_prob
        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Linear(self.num_hidden, self.num_hidden)
        self.encoder = TreeEncoder(in_dim, dropout_prob)
        self.decoder = TreeDecoder(out_dim, dropout_prob)
        self.is_done = False

    def forward(self, x):
        if self.is_done:
            encoded, indexes = self.encoder(x)
            z = self.mu(encoded)
            decoded = self.decoder(z, indexes)

            return decoded[0]

        encoded, indexes = self.encoder(x)
        mean = self.mu(encoded)
        log_var = self.log_var(encoded)
        batch, dim = mean.shape
        epsilon = torch.randn(batch, dim)
        z = mean + torch.exp(0.5 * log_var) * epsilon
        decoded = self.decoder(z, indexes)

        return decoded[0]
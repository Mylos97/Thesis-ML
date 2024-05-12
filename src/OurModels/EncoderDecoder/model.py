import torch
import torch.nn as nn
from OurModels.EncoderDecoder.encoder import TreeEncoder
from OurModels.EncoderDecoder.decoder import TreeDecoder

class MaxNormalize(nn.Module):
    def __init__(self):
        super(MaxNormalize, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        max_val = torch.max(torch.abs(x))
        x = self.softmax(x / max_val)
        return x

class VAE(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob):
        super().__init__()
        self.num_hidden = 16
        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU()
        )
        self.encoder = TreeEncoder(in_dim, dropout_prob)
        self.decoder = TreeDecoder(out_dim, dropout_prob)
        self.training = True
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
            return x

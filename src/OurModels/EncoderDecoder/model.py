import torch
import torch.nn as nn
from OurModels.EncoderDecoder.encoder import TreeEncoder
from OurModels.EncoderDecoder.decoder import TreeDecoder


class VAE(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob):
        super().__init__()
        self.num_hidden = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mu = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_hidden), nn.ReLU()
        )
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
            epsilon = torch.randn(batch, dim).to(self.device)
            z = mean + torch.exp(0.5 * log_var) * epsilon
            decoded = self.decoder(z, indexes)
            x = self.softmax(decoded[0])
            return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from OurModels.EncoderDecoder.encoder import TreeEncoder
from OurModels.EncoderDecoder.decoder import TreeDecoder


class BVAE(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob, z_dim):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.mu = nn.Linear(z_dim, z_dim)
        self.log_var = nn.Linear(z_dim, z_dim)
        self.encoder = TreeEncoder(in_dim, dropout_prob, z_dim)
        self.decoder = TreeDecoder(out_dim, dropout_prob, z_dim)
        self.training = False
        self.softmax = nn.Softmax(dim=1)
        #self.logger = logging.getLogger(__name__)
        #logging.basicConfig(filename='src/Logs/bvae.log', level=logging.INFO)


    def forward(self, x):
        if not self.training:
            # remove the padding from ONNX
            if x[1].shape[1] < x[0].shape[2]:
                x[0] = x[0][:, :, :x[1].shape[1]]
            print('BVAE started inference')
            encoded, indexes = self.encoder(x)
            z = self.mu(encoded)
            decoded = self.decoder(z, indexes)
            print(f"Indexes: {indexes[0].shape}")

            pad_upper = x[0].shape[2]
            pad_lower = decoded[0].shape[2]
            x = decoded[0]

            if pad_lower < pad_upper:
                pad_size = pad_upper - pad_lower

                x = F.pad(decoded[0], (0, pad_size))
                print(f"decoded after padding: {x.shape}")


            x = self.softmax(x)
            print('BVAE finished')

            return x
        else:
            print('BVAE started training')
            encoded, indexes = self.encoder(x)
            print(f"Indexes: {indexes[0]}")
            mean = self.mu(encoded)
            log_var = self.log_var(encoded)
            batch, dim = mean.shape
            epsilon = torch.randn(batch, dim).to(self.device)
            z = mean + torch.exp(0.5 * log_var) * epsilon
            decoded = self.decoder(z, indexes)
            print(f"input: {x[0].shape}")
            print(f"decoded: {decoded[0].shape}")

            pad_upper = x[0].shape[2]
            pad_lower = decoded[0].shape[2]

            x = decoded[0]

            if pad_lower < pad_upper:
                pad_size = pad_upper - pad_lower

                x = F.pad(decoded[0], (0, pad_size))
                print(f"decoded after padding: {x.shape}")


            x = self.softmax(x)
            print('BVAE finished training')


            return [x, mean, log_var]

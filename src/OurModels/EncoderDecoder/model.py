import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        x = self.encoder(src)
        x = self.decoder(x)

        return x
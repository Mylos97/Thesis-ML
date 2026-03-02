import torch
import torch.nn as nn
import torch.nn.functional as F

from OurModels.EncoderDecoder.betaCVAE.encoder import (TreeEncoder, CVAEEncoder)
from OurModels.EncoderDecoder.betaCVAE.decoder import TreeDecoder

class BetaCVAE(nn.Module):
    '''
    logical_dim: nr. of features in logical op. node vector.
    physical_dim: nr. of features in physical op. node vector.
    hidden_dim: the size of the internal tree embedding vectors.
    num_phys_ops: output classes per node (aka choices).
    '''
    def __init__(
        self,
        logical_dim: int,
        physical_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_phys_ops: int,
        beta=1.0
    ):
        super().__init__()

        self.encoder = CVAEEncoder(
            logical_dim,
            physical_dim,
            hidden_dim,
            latent_dim
        )

        self.decoder = TreeDecoder(
            logical_dim,
            hidden_dim,
            latent_dim,
            num_phys_ops
        )

        self.beta = beta

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    """
    logical_tree: (node_feats, children)
    z: [latent_dim] or [1, latent_dim]
    temperature: softmax temperature
    stochastic: if False, use argmax

    Returns:
    phys_labels: [N] integer tensor
    """
    def sample(
        self,
        logical_tree: torch.Tensor,
        z: torch.Tensor,
        temperature: float = 1.0,
        stochastic: bool = True
    ):
        logits = self.decoder(logical_tree, z)
        # ---- 4. Temperature scaling ----
        logits = logits / temperature

        if stochastic:
            probs = F.softmax(logits, dim=-1)
            return probs
            """
            dist = torch.distributions.Categorical(probs)
            phys_labels = dist.sample()  # [N]
            """
        else:
            phys_labels = torch.argmax(logits, dim=-1)

        return phys_labels

    def forward(
        self,
        logical_tree: torch.Tensor,
        physical_tree: torch.Tensor
    ):
        mu, logvar = self.encoder(logical_tree, physical_tree)
        z = self.reparameterize(mu, logvar)

        logits = self.decoder(logical_tree, z)

        return logits, mu, logvar

class Loss(torch.nn.Module):
    def __init__(self, beta: float = 1.0):
        super(Loss, self).__init__()
        self.beta = beta

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ):
        # Reconstruction loss (per-node cross-entropy)
        targets = targets.permute(0, 2, 1)

        recon = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.permute(0, 2, 1).view(-1, targets.size(-1)),
            reduction='mean'
        )

        # KL divergence
        kl = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return recon + self.beta * kl, recon, kl

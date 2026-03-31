import torch
import torch.nn as nn
import torch.nn.functional as F

from OurModels.EncoderDecoder.betaCVAE.encoder import (TreeEncoder, CVAEEncoder)
from OurModels.EncoderDecoder.betaCVAE.decoder import TreeDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CarbVAE(nn.Module):

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
        dropout: float,
        beta: float = 1.0,
        gamma: float =1.0,
        delta: float =1.0
    ):
        super().__init__()

        self.encoder = CVAEEncoder(
            logical_dim,
            physical_dim,
            hidden_dim,
            latent_dim,
            dropout
        )

        self.decoder = TreeDecoder(
            self.encoder.logical_encoder,
            logical_dim,
            hidden_dim,
            latent_dim,
            num_phys_ops,
            dropout
        )

        self.beta = beta
        self.gamma = gamma
        self.delta = delta

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
        physical_tree: torch.Tensor,
    ):
        mu, logvar = self.encoder(logical_tree, physical_tree)
        z = self.reparameterize(mu, logvar)

        logits = self.decoder(logical_tree, z)

        return logits, mu, logvar, z

class Loss(torch.nn.Module):
    def __init__(self, beta: float = 1.0, gamma: float = 1.0, delta: float = 1.0):
        super(Loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        latency: torch.Tensor
    ):
        # Reconstruction loss (per-node cross-entropy)
        targets = targets.permute(0, 2, 1).contiguous()  # permute once, use consistently
        recon = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1, targets.size(-1))  # no second permute
        )

        # KL divergence
        kl = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # attribute regularization - requires normalized latency
        #attr_loss = F.mse_loss(z[:, 0].to(device), latency.to(device)).to(device)
        attr_loss = self.reg_loss(z, latency, self.delta)

        print(f"recon: {recon.item():.4f}, kl: {kl.item():.4f}, attr: {attr_loss.item():.4f}")

        return recon + self.beta * kl + self.gamma * attr_loss, recon, kl

    def reg_loss(self, z: torch.Tensor, latencies: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
        """
        Computes the AR-VAE ordinal regularization loss for z[0] and latency.
        Enforces that the ordering of z[0] matches the ordering of latencies.

        Args:
            z:         latent vectors, shape [batch_size, z_dim]
            latencies: normalized latency values, shape [batch_size]
            factor:    scaling factor for tanh, controls sensitivity to small differences
        Returns:
            scalar loss
        """
        latent_code = z[:, 0]  # shape: [batch_size]

        # Compute pairwise latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # Compute pairwise attribute distance matrix
        attribute = latencies.view(-1, 1).repeat(1, latencies.shape[0])
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # Ordinal loss: tanh of latent differences should match sign of latency differences
        lc_tanh = torch.tanh(lc_dist_mat * factor)
        attribute_sign = torch.sign(attribute_dist_mat)

        return F.l1_loss(lc_tanh, attribute_sign.float())


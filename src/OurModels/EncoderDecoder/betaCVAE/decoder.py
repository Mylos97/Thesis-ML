import torch
import torch.nn as nn
import torch.nn.functional as F

from OurModels.EncoderDecoder.betaCVAE.encoder import TreeEncoder

class TreeDecoder(nn.Module):
    def __init__(self, logical_dim, hidden_dim, latent_dim, num_phys_ops):
        super().__init__()

        self.logical_encoder = TreeEncoder(logical_dim, hidden_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_phys_ops)
        )

    def forward(self, logical_tree, z):
        node_feats, children = logical_tree

        # Per-node embeddings
        x = self.logical_encoder.conv1([node_feats, children])
        x = F.relu(self.logical_encoder.norm1(x)[0])

        x = self.logical_encoder.conv2([x, children])
        x = F.relu(self.logical_encoder.norm2(x)[0])

        # Broadcast latent to each node
        z_expanded = z.unsqueeze(2).expand(-1, -1, x.size(2))

        xz = torch.cat([x.float(), z_expanded.float()], dim=1)
        xz = xz.permute(0, 2, 1)

        logits = self.node_mlp(xz)

        return logits  # per-node operator logits

import torch
import torch.nn as nn
import torch.nn.functional as F

from TreeConvolution.tcnn import (BinaryTreeConv, DynamicPooling,
                                  TreeActivation, TreeLayerNorm)

class TreeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()

        self.conv1 = BinaryTreeConv(in_dim, hidden_dim)
        self.norm1 = TreeLayerNorm()

        self.conv2 = BinaryTreeConv(hidden_dim, hidden_dim)
        self.norm2 = TreeLayerNorm()

        self.pool = DynamicPooling()  # global tree embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feats, children):
        x = self.conv1([node_feats, children])
        x = F.relu(self.norm1(x)[0])

        x = self.conv2([x, children])
        x = self.dropout(F.relu(self.norm2(x)[0]))

        pooled = self.pool([x, children])  # shape: [batch, hidden_dim]

        return pooled

class CVAEEncoder(nn.Module):
    def __init__(self, logical_dim, physical_dim, hidden_dim, latent_dim, dropout):
        super().__init__()

        self.logical_encoder = TreeEncoder(logical_dim, hidden_dim, dropout)
        self.physical_encoder = TreeEncoder(physical_dim, hidden_dim, dropout)

        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, logical_tree, physical_tree):
        logical_tree, log_children = logical_tree
        l_embed = self.logical_encoder(logical_tree, log_children)
        p_embed = self.physical_encoder(physical_tree, log_children)

        combined = torch.cat([l_embed, p_embed], dim=-1)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)

        return mu, logvar

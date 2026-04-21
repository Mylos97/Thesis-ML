import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GatClassifier(nn.Module):
    # forward(batch) -> (logits,)  where logits: [total_nodes_in_batch, out_dim]
    # batch is a torch_geometric.data.Batch produced by loader.make_gat_dataloader
    def __init__(self, in_dim, out_dim, hidden=256, heads=4, dropout_prob=0.0, **kwargs):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=True, dropout=dropout_prob)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, concat=True, dropout=dropout_prob)
        self.conv3 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout_prob)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden * heads)
        self.norm3 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)
        self.act = nn.Mish()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.norm1(self.act(self.conv1(x, edge_index)))
        x = self.norm2(self.act(self.conv2(x, edge_index)))
        x = self.norm3(self.act(self.conv3(x, edge_index)))
        return (self.head(x),)

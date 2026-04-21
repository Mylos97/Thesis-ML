import ast
import re
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from helper import generate_tree_latency_list
from TreeConvolution.util import is_null_operator


def _tree_to_pyg(input_node, target_node, node_features, node_targets, edge_src, edge_dst, parent_idx):
    # Skip binary-tree padding nodes that TCNN required but GAT does not
    if is_null_operator(input_node):
        return

    my_idx = len(node_features)
    node_features.append(np.array(input_node[0], dtype=np.float32))
    node_targets.append(np.array(target_node[0], dtype=np.float32))

    if parent_idx is not None:
        # Bidirectional edges so attention can flow both bottom-up and top-down
        edge_src.extend([parent_idx, my_idx])
        edge_dst.extend([my_idx, parent_idx])

    left_in  = input_node[1]  if len(input_node)  > 1 else None
    right_in = input_node[2]  if len(input_node)  > 2 else None
    left_tgt = target_node[1] if len(target_node) > 1 else None
    right_tgt= target_node[2] if len(target_node) > 2 else None

    if left_in is not None:
        _tree_to_pyg(left_in, left_tgt, node_features, node_targets, edge_src, edge_dst, my_idx)
    if right_in is not None:
        _tree_to_pyg(right_in, right_tgt, node_features, node_targets, edge_src, edge_dst, my_idx)


def tree_pair_to_data(input_tree, target_tree):
    node_features, node_targets, edge_src, edge_dst = [], [], [], []
    _tree_to_pyg(input_tree, target_tree, node_features, node_targets, edge_src, edge_dst, parent_idx=None)

    x = torch.tensor(np.stack(node_features), dtype=torch.float32)
    y = torch.tensor(np.stack(node_targets), dtype=torch.float32)
    y = torch.clamp(y, 0.0, 1.0)

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def load_gat_data(path, num_ops=43, num_platfs=9):
    regex_pattern = r'\(((?:[+,-]?\d+(?:,[+,-]?\d+)*)(?:\s*,\s*\(.*?\))*)\)'

    def platform_encodings(optimal_tree):
        for match in re.finditer(regex_pattern, optimal_tree):
            in_parens = match.group()
            values = [int(v.strip()) for v in in_parens.strip('()').split(',')]
            replacement = ','.join(map(str, values[num_ops:num_ops + num_platfs]))
            optimal_tree = optimal_tree.replace(in_parens, f"({replacement})", 1)
        return optimal_tree

    graphs = []
    in_dim = out_dim = None

    for tup in generate_tree_latency_list(path):
        input_tree  = ast.literal_eval(tup[0])
        target_tree = ast.literal_eval(platform_encodings(tup[1]))

        if in_dim is None:
            in_dim  = len(input_tree[0])
            out_dim = len(target_tree[0])

        graphs.append(tree_pair_to_data(input_tree, target_tree))

    print(f'Successfully loaded {len(graphs)} plans')
    return graphs, in_dim, out_dim


def make_gat_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_gat_data_loaders(train_path, test_path, val_path, batch_size, num_ops=43, num_platfs=9):
    train_data, in_dim, out_dim = load_gat_data(train_path, num_ops, num_platfs)
    val_data,   _,      _       = load_gat_data(val_path,   num_ops, num_platfs)
    test_data,  _,      _       = load_gat_data(test_path,  num_ops, num_platfs)

    train_loader = make_gat_dataloader(train_data, batch_size, shuffle=True)
    val_loader   = make_gat_dataloader(val_data,   batch_size, shuffle=False)
    test_loader  = make_gat_dataloader(test_data,  batch_size, shuffle=False)

    print(f'Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}')
    return train_loader, val_loader, test_loader, in_dim, out_dim

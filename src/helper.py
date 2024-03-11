import numpy as np
import torch
from torch.utils.data import DataLoader
from TreeConvolution.util import prepare_trees

def collate_pairwise_fn(x):
    trees1 = []
    trees2 = []
    labels = []

    for tree1, tree2, label in x:
        trees1.append(tree1)
        trees2.append(tree2)
        labels.append(label)
    return trees1, trees2, labels

def left_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        return None
    return x[1]

def right_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        return None
    return x[2]

def transformer(x):
    return np.array(x[0])

def make_dataloader_pairwise(x, batch_size):
    dataset = DataLoader(x,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_pairwise_fn)
    return dataset

def make_dataloader(x, batch_size, device): # We dont know the structure yet træ, cost
    dataset = DataLoader(x,
                batch_size=batch_size,
                shuffle=True,
                transform=get_trees_and_labels(device))
    return dataset

def get_trees_and_labels(x, device):
    trees = []
    targets = []
    for tree, target in x:
        trees.append(tree)
        targets.append(target)
    
    return build_trees(trees, device=device), torch.tensor(targets).to(device=device)

def build_trees(feature, device):
    return prepare_trees(feature, transformer, left_child, right_child, device=device)

def make_pairs(X1,X2,Y1,Y2) ->  list[(tuple, tuple, tuple)]:
    assert len(X1) == len(X2) and len(Y1) == len(Y2) and len(X1) == len(Y1)
    if isinstance(Y1, list):
        Y1 = np.array(Y1)
        Y1 = Y1.reshape(-1, 1)
    if isinstance(Y2, list):
        Y2 = np.array(Y2)
        Y2 = Y2.reshape(-1, 1)

    pairs = []
    for i in range(len(X1)):
        pairs.append((X1[i], X2[i], 1.0 if Y1[i] >= Y2[i] else 0.0))
    return pairs
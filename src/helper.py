import ast
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
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

def make_dataloader(x, batch_size): # We dont know the structure yet træ, cost
    dataset = DataLoader(x,
                batch_size=batch_size,
                shuffle=True)
    return dataset

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

def load_autoencoder_data(device):
    trees = []
    targets = []
    with open('Data/autoencoder.txt', 'r') as f:
        for l in f:
            vector, cost = l.split(':')
            vector, cost = vector.strip(), cost.strip()
            vector, cost = ast.literal_eval(vector), ast.literal_eval(cost) 
            trees.append(vector)
            targets.append(cost)
    assert len(trees) == len(targets)
    x = []
    in_trees = build_trees(trees, device=device)
    target_trees = build_trees(targets, device=device)
    for i, tree in enumerate(in_trees[0]):
        x.append(((tree, in_trees[1][i]), target_trees[0][i]))
        
    return TreeVectorDataset(x)

class TreeVectorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vector, cost = self.data[idx]
        return vector, cost
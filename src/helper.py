import ast
import numpy as np
import onnx
import torch
import os
from TreeConvolution.util import prepare_trees
from onnx import numpy_helper
from torch.utils.data import DataLoader, Dataset
from itertools import combinations

def get_relative_path(file_name, dir):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, dir, file_name)
    return file_path

def to_device(vector, target, device):
    if len(vector) == 2:
        return [vector[0].to(device), vector[1].to(device)], target.to(device)
    
    return None

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

def make_dataloader(x, batch_size, num_workers):
    dataset = DataLoader(x,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers)
    return dataset

def build_trees(feature, device):
    return prepare_trees(feature, transformer, left_child, right_child, device=device)

def load_autoencoder_data(device, path):
    trees = []
    targets = []
    with open(path, 'r') as f:
        for l in f:
            vector, cost = l.split(':')
            vector, cost = vector.strip(), cost.strip()
            vector, cost = ast.literal_eval(vector), ast.literal_eval(cost) 
            trees.append(vector)
            targets.append(cost)
    
    assert len(trees) == len(targets)
    in_dim, out_dim = len(vector[0]), len(cost[0])
    x = []
    in_trees = build_trees(trees, device=device)
    target_trees = build_trees(targets, device=device)
    for i, tree in enumerate(in_trees[0]):
        x.append(((tree, in_trees[1][i]), target_trees[0][i]))
    return TreeVectorDataset(x), in_dim, out_dim
    
def load_pairwise_data():
    def generate_unique_pairs(lst):
        return list(combinations(lst, 2))

    with open(get_relative_path('pairwise.txt', 'Data'), 'r') as f:
        vectors = []
        x = []
        for l in f:
            tree, cost = l.split(":")
            vectors.append({"tree": tree, "cost":cost}) # need make into a string
        
        pairs_trees = generate_unique_pairs(vectors)

        for tree1, tree2 in pairs_trees:
            label = 0.0 if tree1["cost"] < tree2["cost"] else 1.0
            x.append((tree1, tree2), label)
            
    in_dim = len(tree1[0])
    return TreeVectorDataset(x), in_dim

def load_classifier_data():
    raise Exception("IMplement classifier loader")
    pass

def get_weights_of_model(modelname):
    onnx_model   = onnx.load(get_relative_path(f'{modelname}.onnx','Models'))
    INTIALIZERS  = onnx_model.graph.initializer
    onnx_weights = {}
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = W
    return onnx_weights

def set_weights(weights, model):
    for name, param in model.named_parameters():
        if name in weights:
            param.data = torch.tensor(weights[name].copy())
    return model

class TreeVectorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vector, cost = self.data[idx]
        return vector, cost
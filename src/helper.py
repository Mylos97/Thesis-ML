import ast
import numpy as np
import onnx
import torch
import os
import json
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
from TreeConvolution.util import prepare_trees
from onnx import numpy_helper
from torch.utils.data import DataLoader, Dataset
from itertools import combinations


class TreeVectorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vector, cost = self.data[idx]
        return vector, cost

def get_relative_path(file_name:str, dir:str) -> str:
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, dir, file_name)
    return file_path

def to_device(vector: torch.Tensor, target: torch.Tensor, device:str) -> tuple[list[torch.Tensor], torch.Tensor]:
    if len(vector) == 2:
        return [vector[0].to(device), vector[1].to(device)], target.to(device)

    return None

def left_child(x:tuple) -> tuple:
    assert isinstance(x, tuple)
    if len(x) == 1:
        return None
    return x[1]

def right_child(x:tuple) -> tuple:
    assert isinstance(x, tuple)
    if len(x) == 1:
        return None
    return x[2]

def transformer(x:tuple) -> np.array:
    return np.array(x[0])

def make_dataloader(x:Dataset, batch_size:int) -> DataLoader:
    dataloader = DataLoader(x,
                batch_size=batch_size,
                drop_last=True,
                shuffle=True)
    return dataloader

def build_trees(feature:list[tuple[torch.Tensor, torch.Tensor]], device:str) -> tuple[torch.Tensor, torch.Tensor]:
    return prepare_trees(feature, transformer, left_child, right_child, device=device)

def load_autoencoder_data(device:str, path:str) -> tuple[TreeVectorDataset, int, int]:
    trees = []
    targets = []
    with open(path, 'r') as f:
        for _ in range(1024):
            l = f.readline()
            s = l.split(':')
            tree, optimal_tree = s[0], s[1]
            tree, optimal_tree = tree.strip(), optimal_tree.strip()
            tree, optimal_tree = ast.literal_eval(tree), ast.literal_eval(optimal_tree)
            trees.append(tree)
            targets.append(optimal_tree)

    assert len(trees) == len(targets)
    in_dim, out_dim = len(tree[0]), len(optimal_tree[0])
    print("in_dim ", in_dim, " out_dim ", out_dim)
    x = []
    in_trees = build_trees(trees, device=device)
    target_trees = build_trees(targets, device=device)
    for i, tree in enumerate(in_trees[0]):
        x.append(((tree, in_trees[1][i]), target_trees[0][i]))
    
    print(f"Succesfully loaded {len(x)} plans")
    return TreeVectorDataset(x), in_dim, out_dim

def load_pairwise_data(device:str, path:str) -> tuple[TreeVectorDataset, int, None]:
    def generate_unique_pairs(best_plan, lst):
        return [(best_plan, x) for x in lst]

    with open(get_relative_path('pairwise-encodings.txt', 'Data'), 'r') as f:
        wayangPlans = {}
        trees = []
        x = []
        pairs_trees = {}
        for l in f:
            s = l.split(":")
            wayangPlan, executionPlan, cost = s[0].strip(), s[1].strip(), int(s[2].strip())
            executionPlan = ast.literal_eval(executionPlan)
            trees.append(executionPlan)
            wayangPlans.setdefault(wayangPlan, []).append((len(trees) - 1, cost))

        print(f"Read {len(wayangPlans)} different WayangPlans")
        in_dim = len(executionPlan[0])
        in_trees = build_trees(trees, device=device)

        for wayangPlan, exTuple in wayangPlans.items():
            best_plan = min(exTuple, key=lambda x: x[1])

            for i, cost in exTuple:
                if i != best_plan[0]:
                    x.append(((in_trees[0][i], in_trees[1][i]), cost))

            best_plan_tuple = ((in_trees[0][best_plan[0]], in_trees[1][best_plan[0]]), best_plan[1])
            pairs_trees[wayangPlan] = generate_unique_pairs(best_plan_tuple, x)

        pairs = []

        for wayangPlan, pair in pairs_trees.items():
            for tree1, tree2 in pair:
                tree1, cost1 = tree1
                tree2, cost2 = tree2
                label = 0.0 if cost1 < cost2 else 1.0
                pairs.append(((tree1, tree2), label))

    return TreeVectorDataset(pairs), in_dim, None

def load_costmodel_data(path, device:str):
    trees = []
    costs = []

    with open(get_relative_path('pairwise-encodings.txt', 'Data'), 'r') as f:
        for l in f:
            s = l.split(':')
            executionPlan, cost = s[1].strip(), int(s[2].strip())
            executionPlan = ast.literal_eval(executionPlan)
            trees.append(executionPlan)
            costs.append(cost)

    in_dim, out_dim = len(executionPlan[0]), None
    x = []
    in_trees = build_trees(trees, device=device)

    for i, tree in enumerate(in_trees[0]):
        x.append(((tree, in_trees[1][i]), costs[i]))

    return TreeVectorDataset(x), in_dim, out_dim

def get_weights_of_model(modelname:str) -> dict:
    onnx_model   = onnx.load(get_relative_path(f'{modelname}.onnx','Models'))
    INTIALIZERS  = onnx_model.graph.initializer
    onnx_weights = {}
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = W
    return onnx_weights

def set_weights(weights:dict, model:torch.nn.Module) -> torch.nn.Module:
    for name, param in model.named_parameters():
        if name in weights:
            param.data = torch.tensor(weights[name].copy())
    return model

def convert_to_json(plans) -> None:
    l = []
    for plan in plans:
        current_plan = {}
        current_plan['values'] = plan[0].tolist()
        current_plan['indexes'] = plan[1].tolist()
        l.append(current_plan)
    json_data = json.dumps(l)
    relative_path = get_relative_path('json-plans', "Data")
    with open(f'{relative_path}.txt', "w") as file:
        file.write(json_data)


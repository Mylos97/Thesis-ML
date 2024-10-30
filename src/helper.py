import ast
import numpy as np
import onnx
import torch
import os
import json
import random
import torch.nn.intrinsic
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
import re
import torch.nn.functional as F
import torch.nn
from TreeConvolution.util import prepare_trees
from onnx import numpy_helper
from torch.utils.data import DataLoader, Dataset


class TreeVectorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vector, cost = self.data[idx]
        return vector, cost

    def append(self, item):
        self.data.append(item)


def get_relative_path(file_name: str, dir: str) -> str:
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, dir, file_name)
    return file_path


def left_child(x: tuple) -> tuple:
    assert isinstance(x, tuple)
    if len(x) == 1:
        return None
    return x[1]


def right_child(x: tuple) -> tuple:
    assert isinstance(x, tuple)
    if len(x) == 1:
        return None
    return x[2]


def transformer(x: tuple) -> np.array:
    return np.array(x[0])


def make_dataloader(x: Dataset, batch_size: int) -> DataLoader:
    dataloader = DataLoader(x, batch_size=batch_size, drop_last=True, shuffle=True)
    return dataloader


def build_trees(
    feature: list[tuple[torch.Tensor, torch.Tensor]], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return prepare_trees(feature, transformer, left_child, right_child, device=device)


def remove_operator_ids(tree: str):
    regex_pattern = r'\(((?:[+,-]?\d+(?:,[+,-]?\d+)*)(?:\s*,\s*\(.*?\))*)\)'
    matches_iterator = re.finditer(regex_pattern, tree)

    for match in matches_iterator:
        find = match.group().strip('(').strip(')')
        values = [int(num.strip()) for num in find.split(',')]
        values[0] = 0
        replacement = ','.join(map(str, values))
        tree = tree.replace(find, replacement)

    return tree


def generate_latency_map_intersect(path, old_tree_latency_map):
    new_tree_latency_map = generate_tree_latency_map(path)

    intersect_latency_map = {}
    key_intersection = {k: old_tree_latency_map[k] for k in old_tree_latency_map if k in new_tree_latency_map}

    for key in key_intersection:
        if old_tree_latency_map[key][1] > new_tree_latency_map[key][1]:
            intersect_latency_map[key] = new_tree_latency_map[key]
        else:
            intersect_latency_map[key] = old_tree_latency_map[key]

    return intersect_latency_map


def load_autoencoder_data(device: str, path: str, retrain_path: str = "", num_ops: int = 43, num_platfs: int = 9) -> tuple[TreeVectorDataset, int, int]:
    regex_pattern = r'\(((?:[+,-]?\d+(?:,[+,-]?\d+)*)(?:\s*,\s*\(.*?\))*)\)'
    path = get_relative_path('naive-lsbo.txt', 'Data') if path == None else path

    def platform_encodings(optimal_tree: str):
        matches_iterator = re.finditer(regex_pattern, optimal_tree)

        for match in matches_iterator:
            find = match.group().strip('(').strip(')')
            values = [int(num.strip()) for num in find.split(',')]
            replacement = ','.join(map(str, values[num_ops:num_ops+num_platfs]))
            optimal_tree = optimal_tree.replace(find, replacement)

        return optimal_tree

    trees = []
    targets = []
    # structure tree -> (exec-plan, latency)
    tree_latency_map = generate_tree_latency_map(path)

    if retrain_path != "":
        tree_latency_map = generate_latency_map_intersect(retrain_path, tree_latency_map)

    for tree, tup in tree_latency_map.items():
        optimal_tree = platform_encodings(tup[0])
        tree, optimal_tree = ast.literal_eval(tree), ast.literal_eval(optimal_tree)
        trees.append(tree)
        targets.append(optimal_tree)

    print(f"Tree size: {len(trees)}")
    print(f"Targets size: {len(targets)}")

    assert len(trees) == len(targets)
    in_dim, out_dim = len(tree[0]), len(optimal_tree[0])
    x = []
    trees, indexes = build_trees(trees, device=device)
    target_trees, _ = build_trees(targets, device=device)
    target_trees = torch.where((target_trees > 1) | (target_trees < 0), 0, target_trees)

    for i, tree in enumerate(trees):
        x.append(((tree, indexes[i]), target_trees[i]))

    print(f'Succesfully loaded {len(x)} plans', flush=True)
    return TreeVectorDataset(x), in_dim, out_dim


def load_autoencoder_data_from_str(device: str, data: str, num_ops: int = 43, num_platfs: int = 9) -> tuple[TreeVectorDataset, int, int]:
    regex_pattern = r'\(((?:[+,-]?\d+(?:,[+,-]?\d+)*)(?:\s*,\s*\(.*?\))*)\)'

    def platform_encodings(optimal_tree: str):
        matches_iterator = re.finditer(regex_pattern, optimal_tree)

        for match in matches_iterator:
            find = match.group().strip('(').strip(')')
            values = [int(num.strip()) for num in find.split(',')]
            replacement = ','.join(map(str, values[num_ops:num_ops+num_platfs]))
            optimal_tree = optimal_tree.replace(find, replacement)

        return optimal_tree

    trees = []
    targets = []
    # structure tree -> (exec-plan, latency)
    tree_latency_map = generate_tree_latency_map_from_str(data)

    for tree, tup in tree_latency_map.items():
        optimal_tree = platform_encodings(tup[0])
        tree, optimal_tree = ast.literal_eval(tree), ast.literal_eval(optimal_tree)
        trees.append(tree)
        targets.append(optimal_tree)

    print(f"Tree size: {len(trees)}")
    print(f"Targets size: {len(targets)}")

    assert len(trees) == len(targets)
    in_dim, out_dim = len(tree[0]), len(optimal_tree[0])
    x = []
    trees, indexes = build_trees(trees, device=device)
    target_trees, _ = build_trees(targets, device=device)
    target_trees = torch.where((target_trees > 1) | (target_trees < 0), 0, target_trees)

    for i, tree in enumerate(trees):
        x.append(((tree, indexes[i]), target_trees[i]))

    print(f'Succesfully loaded {len(x)} plans', flush=True)
    return TreeVectorDataset(x), in_dim, out_dim

def generate_tree_latency_map(path):
    tree_latency_map = {}
    with open(path, 'r') as f:
        for l in f:
            s = l.split(':')
            tree, optimal_tree, latency = s[0], s[1], int(s[2].strip())
            tree, optimal_tree = (
                remove_operator_ids(tree.strip()),
                remove_operator_ids(optimal_tree.strip()),
            )

            if tree in tree_latency_map:
               if tree_latency_map[tree][1] > latency:
                   tree_latency_map[tree] = (optimal_tree, latency)
            else:
                tree_latency_map[tree] = (optimal_tree, latency)
    return tree_latency_map

def generate_tree_latency_map_from_str(plans: str):
    tree_latency_map = {}
    for plan in plans:
        s = plan.split(':')
        tree, optimal_tree, latency = s[0], s[1], int(s[2].strip())
        tree, optimal_tree = (
            remove_operator_ids(tree.strip()),
            remove_operator_ids(optimal_tree.strip()),
        )

        if tree in tree_latency_map:
           if tree_latency_map[tree][1] > latency:
               tree_latency_map[tree] = (optimal_tree, latency)
        else:
           tree_latency_map[tree] = (optimal_tree, latency)

    return tree_latency_map


def load_pairwise_data(device: str, path: str) -> tuple[TreeVectorDataset, int, None]:
    path = get_relative_path('pairwise-encodings.txt', 'Data') if path == None else path

    with open(path, 'r') as f:
        wayangPlans = {}
        trees = []
        pairs_trees = {}
        for l in f:
            s = l.split(':')
            wayangPlan, executionPlan, cost = (
                remove_operator_ids(s[0].strip()),
                remove_operator_ids(s[1].strip()),
                int(s[2].strip()),
            )
            executionPlan = ast.literal_eval(executionPlan)
            trees.append(executionPlan)
            wayangPlans.setdefault(wayangPlan, []).append((len(trees) - 1, cost))

        print(f'Read {len(wayangPlans)} different WayangPlans', flush=True)
        in_dim = len(executionPlan[0])
        trees, indexes = build_trees(trees, device=device)

        for wayangPlan, exTuple in wayangPlans.items():
            best_plan_index, best_cost = min(exTuple, key=lambda x: x[1])
            best_tree = ((trees[best_plan_index], indexes[best_plan_index]), best_cost)
            tuples = []
            for i, cost in exTuple:
                if i == best_plan_index:
                    continue
                current_tree = ((trees[i], indexes[i]), cost)
                pair = [best_tree, current_tree]
                random.shuffle(pair)
                tuples.append(pair)

            pairs_trees[wayangPlan] = tuples

        pairs = []
        labels = [0,0]
        for wayangPlan, pair in pairs_trees.items():
            for tree1, tree2 in pair:
                tree1, cost1 = tree1
                tree2, cost2 = tree2
                label = 0.0 if cost1 < cost2 else 1.0
                labels[int(label)] += 1
                pairs.append(((tree1, tree2), torch.tensor(label).to(device)))

    print(f'Found {len(pairs)} different Wayang pairs and labels {labels}')

    return TreeVectorDataset(pairs), in_dim, None


def load_costmodel_data(device: str, path: str) -> tuple[TreeVectorDataset, int, None]:
    path = get_relative_path('full-encodings.txt', 'Data') if path == None else path
    trees = []
    costs = []

    with open(path, 'r') as f:
        for l in f:
            s = l.split(':')
            executionPlan, cost = remove_operator_ids(s[1].strip()), int(s[2].strip())
            executionPlan = ast.literal_eval(executionPlan)
            trees.append(executionPlan)
            costs.append(cost)
    print(f'Loaded {len(trees)} different trees')
    in_dim, out_dim = len(executionPlan[0]), None
    x = []
    trees, indexes = build_trees(trees, device=device)

    costs = torch.tensor(costs).to(device)

    for i, tree in enumerate(trees):
        x.append(((tree, indexes[i]), costs[i]))

    return TreeVectorDataset(x), in_dim, out_dim


def get_weights_of_model(modelname: str) -> dict:
    onnx_model = onnx.load(get_relative_path(f'{modelname}.onnx', 'Models'))
    INTIALIZERS = onnx_model.graph.initializer
    onnx_weights = {}
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = W
    return onnx_weights

def get_weights_of_model_by_path(path: str,) -> dict:
    onnx_model = onnx.load(path)
    INTIALIZERS = onnx_model.graph.initializer
    onnx_weights = {}
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = W
    return onnx_weights


def set_weights(weights: dict, model: torch.nn.Module, device: str) -> torch.nn.Module:
    for name, param in model.named_parameters():
        if name in weights:
            param.data = torch.tensor(weights[name].copy())
    return model


def get_data_loaders(data, batch_size):
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        data, [0.8, 0.1, 0.1]
    )
    train_loader = make_dataloader(x=train_dataset, batch_size=batch_size)
    val_loader = make_dataloader(x=val_dataset, batch_size=batch_size)
    test_loader = make_dataloader(x=test_dataset, batch_size=batch_size)
    print(f"Train set len: {len(train_loader)}")
    print(f"Val set len: {len(val_loader)}")
    print(f"Test set len: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def convert_to_json(plans) -> None:
    l = []
    for plan in plans:
        current_plan = {}
        current_plan['values'] = plan[0].tolist()
        current_plan['indexes'] = plan[1].tolist()
        l.append(current_plan)
    json_data = json.dumps(l)
    relative_path = get_relative_path('json-plans', 'Data')
    with open(f'{relative_path}.txt', 'w') as file:
        file.write(json_data)

class Beta_Vae_Loss(torch.nn.Module):
    def __init__(self, beta=1.0):
        super(Beta_Vae_Loss, self).__init__()
        self.beta = beta

    def forward(self, prediction, target):
        recon_x, mu, logvar = prediction
        #recon_loss = F.cross_entropy(recon_x, target)
        recon_loss = F.binary_cross_entropy(recon_x, target, reduction='sum')
        loss_reg = (-0.5 * (1 + logvar - mu**2 - logvar.exp())).mean(dim=0).sum()
        total_kld = loss_reg * 0.0001

        return recon_loss + total_kld * self.beta

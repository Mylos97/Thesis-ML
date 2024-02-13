import numpy as np

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
import torch
import numpy as np
import torch.nn as nn
from TreeConvolution.util import prepare_trees

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
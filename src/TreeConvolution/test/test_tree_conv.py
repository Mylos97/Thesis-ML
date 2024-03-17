import sys
sys.path.append("..")  # Add parent directory to the sys.path
sys.path.append("../../")  # Add parent directory to the sys.path
import unittest
import numpy as np
from torch import nn
from util import prepare_trees
from exporter import export_model
import tcnn
import ast

class TestTreeConvolution(unittest.TestCase):
    def test_example(self):
        tree1 = (
            (0, 1,2),
            ((3, 4,5), ((6, 7,8),), ((9, 10,11),)),
            ((12, 13,14), ((15, 16, 17),), ((18,19,20),))
        )

        trees = [tree1]
        
        # function to extract the left child of a node
        def left_child(x):
            assert isinstance(x, tuple)
            if len(x) == 1:
                # leaf.
                return None
            return x[1]

        # function to extract the right child of node
        def right_child(x):
            assert isinstance(x, tuple)
            if len(x) == 1:
                # leaf.
                return None
            return x[2]

        # function to transform a node into a (feature) vector,
        # should be a numpy array.
        def transformer(x):
            return np.array(x[0])


        prepared_trees = prepare_trees(trees, transformer, left_child, right_child)
        net = nn.Sequential(
            tcnn.BinaryTreeConv(3, 16),
            tcnn.TreeLayerNorm(),
            tcnn.TreeActivation(nn.ReLU()),
            tcnn.BinaryTreeConv(16, 8),
            tcnn.TreeLayerNorm(),
            tcnn.TreeActivation(nn.ReLU()),
            tcnn.BinaryTreeConv(8, 4),
            tcnn.TreeLayerNorm(),
            tcnn.TreeActivation(nn.ReLU()),
            tcnn.DynamicPooling()
        )
        print(prepared_trees)
        shape = tuple(net(prepared_trees).shape)
        self.assertEqual(shape, (2, 4))

if __name__ == '__main__':
    unittest.main()

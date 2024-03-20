import sys
sys.path.append("..")  # Add parent directory to the sys.path
sys.path.append("../../")  # Add parent directory to the sys.path
import unittest
import numpy as np
from torch import nn
from util import prepare_trees
import tcnn

class TestTreeConvolution(unittest.TestCase):
    def test_example(self):
        # First tree:
        #               (0, 1)
        #       (1, 2)        (-3, 0)
        #   (0, 1) (-1, 0)  (2, 3) (1, 2)

        tree1 = (
            (0, 1),
            ((1, 2), ((0, 1),), ((-1, 0),)),
            ((-3, 0), ((2, 3),), ((1, 2),))
        )

        # Second tree:
        #               (16, 3)
        #       (0, 1)         (2, 9)
        #   (5, 3)  (2, 6)

        tree2 = (
            (16, 3),
            ((0, 1), ((5, 3),), ((2, 6),)),
            ((2, 9),)
        )
        tree3 = ((0,1,2,3),((4,5,6,7), ((8,9,10,11),((12,13,14,15),((16,17,18,19),((20,21,22,23),((24,25,26,27),),((28,29,30,31),)),((32,33,34,35),)),((36,37,38,39),)),((40,41,42,43),)),((44,45,46,47),)),((48,49,50,51),))
        trees = [tree3]
        
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
            tcnn.BinaryTreeConv(2, 16),
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
        shape = tuple(net(prepared_trees).shape)
        self.assertEqual(shape, (1, 4))

if __name__ == '__main__':
    unittest.main()

import torch
import torch.nn as nn
import numpy as np
from TreeModel import LeroNet
from torch.utils.data import DataLoader
from HelperFunctions import collate_pairwise_fn

class PairWise:

    def __init__(self, input_dim):
        self.net = LeroNet(input_dim)

    def fit(self, X1, X2, Y1, Y2):
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

        batch_size = 64
        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_pairwise_fn)

        optimizer = torch.optim.Adam(self.net.parameters())
        bce_loss_fn = torch.nn.BCELoss()

        losses = []
        sigmoid = nn.Sigmoid()
        for epoch in range(100):
            loss_accum = 0
            for x1, x2, label in dataset:

                tree_x1 = self.net.build_trees(x1)
                tree_x2 = self.net.build_trees(x2)

                # pairwise
                y_pred_1 = self.net(tree_x1)
                y_pred_2 = self.net(tree_x2)
                diff = y_pred_1 - y_pred_2
                prob_y = sigmoid(diff)

                label_y = torch.tensor(np.array(label).reshape(-1, 1))

                loss = bce_loss_fn(prob_y, label_y)
                loss_accum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print("Epoch", epoch, "training loss:", loss_accum)
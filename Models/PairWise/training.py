import torch
import torch.nn as nn
import numpy as np
from Models.PairWise.treenetwork import TreeConvolution256
from Models.PairWise.helper import make_dataloader, make_pairs
from Exporter.exporter import export_model

def fit(X1, X2, Y1, Y2):
    pairs = make_pairs(X1,X2,Y1,Y2)

    batch_size = 64
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(pairs, [0.8, 0.1, 0.1])
    
    train_dataloader = make_dataloader(train_dataset, batch_size)
    val_dataloader = make_dataloader(val_dataset, batch_size)
    test_dataloader = make_dataloader(test_dataset, batch_size)

    model = TreeConvolution256(len(X1[1])) # Maybe not correct we need to find the length of each tuple
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    bce_loss_fn = torch.nn.BCELoss()

    losses = []
    sigmoid = nn.Sigmoid()
    for epoch in range(100):
        loss_accum = 0
        test_loss = 0
        for x1, x2, label in train_dataloader:
            tree_x1 = model.build_trees(x1)
            tree_x2 = model.build_trees(x2)

            y_pred_1 = model(tree_x1)
            y_pred_2 = model(tree_x2)
            diff = y_pred_1 - y_pred_2
            prob_y = sigmoid(diff)

            label_y = torch.tensor(np.array(label).reshape(-1, 1))

            loss = bce_loss_fn(prob_y, label_y)
            loss_accum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_accum /= len(train_dataloader)
        losses.append(loss_accum)

        with torch.no_grad():
            model.eval()
            for x1, x2, label in test_dataloader:
                tree_x1 = model.build_trees(x1)
                tree_x2 = model.build_trees(x2)

                y_pred_1 = model(tree_x1)
                y_pred_2 = model(tree_x2)
                diff = y_pred_1 - y_pred_2
                prob_y = sigmoid(diff)

                label_y = torch.tensor(np.array(label).reshape(-1, 1))

                test_loss = bce_loss_fn(prob_y, label_y)
                test_loss += loss.item()


        print("Epoch", epoch, "training loss:", loss_accum, "test loss:", test_loss)
        
    export_model(model, x1)
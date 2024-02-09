import os
import joblib
import torch
import numpy as np
from torch.utils.data import DataLoader
from TreeNetwork import LeroNet
from helperfunctions import (_nn_path, _feature_generator_path, 
                             _input_feature_dim_path, collate_fn)

class LeroModel():
    def __init__(self, feature_generator) -> None:
        self._net = None
        self._feature_generator = feature_generator
        self._input_feature_dim = None
        self._model_parallel = None

    def load(self, path):
        with open(_input_feature_dim_path(path), "rb") as f:
            self._input_feature_dim = joblib.load(f)

        self._net = LeroNet(self._input_feature_dim)

        self._net.load_state_dict(torch.load(_nn_path(path), map_location=torch.device('cpu')))
        self._net.eval()
        with open(_feature_generator_path(path), "rb") as f:
            self._feature_generator = joblib.load(f)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self._net.state_dict(), _nn_path(path))

        with open(_feature_generator_path(path), "wb") as f:
            joblib.dump(self._feature_generator, f)
        with open(_input_feature_dim_path(path), "wb") as f:
            joblib.dump(self._input_feature_dim, f)

    def fit(self, X, Y, pre_training=False):
        if isinstance(Y, list):
            Y = np.array(Y)
            Y = Y.reshape(-1, 1)

        batch_size = 64
        pairs = []
        for i in range(len(Y)):
            pairs.append((X[i], Y[i]))
        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)

        if not pre_training:
            # # determine the initial number of channels
            input_feature_dim = len(X[0].get_feature())
            print("input_feature_dim:", input_feature_dim)

            self._net = LeroNet(input_feature_dim)
            self._input_feature_dim = input_feature_dim

            optimizer = torch.optim.Adam(self._net.parameters())

        loss_fn = torch.nn.MSELoss()
        losses = []
        for epoch in range(100):
            loss_accum = 0
            for x, y in dataset:
                tree = self._net.build_trees(x)

                y_pred = self._net(tree)
                loss = loss_fn(y_pred, y)
                loss_accum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print("Epoch", epoch, "training loss:", loss_accum)

    def predict(self, x):
        if not isinstance(x, list):
            x = [x]
        
        tree = self._net.build_trees(x)
        pred = self._net(tree).cpu().detach().numpy()
        
        return pred
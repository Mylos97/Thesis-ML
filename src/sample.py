import os
import sys
import torch
import torch.onnx
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
from subprocess import PIPE, Popen, TimeoutExpired
import signal
import threading
import random
import math
import datetime
import onnx
import onnxruntime

from OurModels.EncoderDecoder.carbVAE.model import CarbVAE
from helper import get_weights_of_model_by_path, set_weights, load_autoencoder_data_from_str, load_autoencoder_data, load_autoencoder_carb_data, get_relative_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def load_model(model_path: str, parameters_path: str, in_dim: int, out_dim: int) -> CarbVAE:
    # find model parameters
    with open(parameters_path) as file:
        parameters = json.load(file)
        lr = parameters.get("lr", 0.001)
        gradient_norm = parameters.get("gradient_norm", 1.0)
        dropout = parameters.get("dropout", 0.1)
        z_dim = parameters.get("z_dim", 1)
        weights = get_weights_of_model_by_path(model_path)


        model = CarbVAE(
            logical_dim=in_dim,
            physical_dim=out_dim,
            hidden_dim=128,
            latent_dim=z_dim,
            num_phys_ops=out_dim,
            dropout=dropout,
            beta=parameters.get('beta', 1.0),
            gamma=parameters.get('gamma', 1.0),
            delta=parameters.get('delta', 1.0)
        )

        if weights:
            set_weights(weights=weights, model=model, device=device)

        # load model
        model.to(device)
        model.eval()

        return model

def load_data(path: str) -> DataLoader:
    data, in_dim, out_dim, mean, std = load_autoencoder_carb_data(device=device, path=path)

    return DataLoader(data, batch_size=64, drop_last=True, shuffle=True), in_dim, out_dim


def encode_training_data(model: CarbVAE, dataloader: DataLoader):
    with torch.no_grad():
        for logical, physical, latency in dataloader:
            mu, logvar = model.encoder(logical, physical)
            print(f"mu mean: {mu.mean(dim=0)}")
            print(f"mu std:  {mu.std(dim=0)}")
            print(f"logvar mean: {logvar.mean():.4f}")

def main():
    model_path = get_relative_path("carbvae.onnx", "Models/imdb/")
    parameters_path = get_relative_path("CarbVAE.json", "HyperparameterLogs/imdb/")
    data_path = get_relative_path("train.txt", "Data/splits/imdb/training/carbvae")

    dataloader, in_dim, out_dim = load_data(data_path)
    model = load_model(model_path, parameters_path, in_dim, out_dim)

    encode_training_data(model, dataloader)


if __name__ == "__main__":
    main()

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import sys
import torch
import torch.onnx
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
from subprocess import PIPE, Popen
import signal
import threading
import random
import math
import argparse
import onnx
import onnxruntime
import numpy as np
import umap
import matplotlib.pyplot as plt

from helper import get_weights_of_model_by_path, set_weights, load_autoencoder_data, get_relative_path

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize, Normalize
from botorch.models.transforms.input import InputStandardize
from botorch import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.optim import optimize_acqf, gen_batch_initial_conditions
from botorch.generation.gen import get_best_candidates, gen_candidates_torch
from OurModels.EncoderDecoder.model import VAE
from OurModels.EncoderDecoder.bvae import BVAE
from Util.communication import read_int, UTF8Deserializer, dump_stream, open_connection
from LSBO.criteria import StoppingCriteria

TIMEOUT = float(60 * 60 * 60)
PLAN_CACHE = set()
best_plan_data = None
z_dim = 31
distinct_choices = []
PLAN_SIZE = 0

def main(args):
    # set some defaults, highly WIP
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path=args.model_path
    parameters_path=args.parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    data, in_dim, out_dim = load_autoencoder_data(path=get_relative_path('test.txt', 'Data/splits/imdb/training'), retrain_path=args.retrain, device=device, num_ops=args.operators, num_platfs=args.platforms)
    #data, in_dim, out_dim = load_autoencoder_data(path=get_relative_path('g0.txt', 'Data/splits/imdb/training'), retrain_path=args.retrain, device=device, num_ops=args.operators, num_platfs=args.platforms)

    # find model parameters
    weights = get_weights_of_model_by_path(model_path)

    # find model parameters
    with open(parameters_path) as file:
        parameters = json.load(file)
        lr = parameters.get("lr", 0.001)
        gradient_norm = parameters.get("gradient_norm", 1.0)
        dropout = parameters.get("dropout", 0.1)
        z_dim = parameters.get("z_dim", 0.1)
        weights = get_weights_of_model_by_path(model_path)

        #best_model, x = do_hyperparameter_BO(model_class=model_class, data=data, in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device, lr=lr, weights=weights, epochs=epochs, trials=trials, plots=args.plots)
        model = BVAE(
            in_dim=in_dim,
            out_dim=out_dim,
            dropout_prob=dropout,
            z_dim=z_dim,
        )


        if weights:
            set_weights(weights=weights, model=model, device=device)

        # load model
        model.to(device)
        model.train()
        model.isTraining = True

        dataloader = DataLoader(data, batch_size=parameters.get("batch_size", 1), drop_last=False, shuffle=False)

        dtype = torch.float64
        latent_target = None

        # Assuming `vae` is your trained model and `data_loader` is your input data
        latent_vectors = []
        labels = []
        #model = model.to("cpu")
        with torch.no_grad():
            for i, (tree,target) in enumerate(dataloader):
                torch.set_printoptions(profile="full")
                model.train()
                model.isTraining = True
                predictions, targets = model.encoder(tree)
                print(predictions)
                latent_vectors.append(predictions.detach().cpu().numpy())
                labels.append([i for i in range(len(predictions))])
                print(labels)
                break

        print(len(latent_vectors))
        latent_vectors = np.concatenate(latent_vectors)
        #latent_vectors = np.concatenate(latent_vectors)
        print(f"Latent vector.shape {latent_vectors.shape}")

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='euclidean')
        embedding = reducer.fit_transform(latent_vectors)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
        plt.colorbar(scatter)
        plt.title("UMAP projection of VAE latent space")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.show()
        plt.savefig("./umap_plot.png", dpi=300)
        print("Plot saved as umap_plot.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vae')
    parser.add_argument('--model-path', default='./src/Data/vae.onnx')
    parser.add_argument('--parameters', default='./src/HyperparameterLogs/BVAE.json')
    parser.add_argument('--retrain', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--lr', type=str, default='[1e-6, 0.1]')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--plots', type=bool, default=False)
    parser.add_argument('--platforms', type=int, default=9)
    parser.add_argument('--operators', type=int, default=43)
    args = parser.parse_args()
    main(args)

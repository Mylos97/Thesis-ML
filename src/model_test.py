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
import re
import os

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

    data, in_dim, out_dim = load_autoencoder_data(path=get_relative_path('30.txt', 'Data/splits/imdb/training'), retrain_path=args.retrain, device=device, num_ops=args.operators, num_platfs=args.platforms)
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
        model.eval()

        dataloader = DataLoader(data, batch_size=1, drop_last=False, shuffle=False)

        dtype = torch.float64
        latent_target = None
        results = []
        with torch.no_grad():
            for tree,target in dataloader:
                model.training = False
                model.eval()
                encoded_plan = model.encoder(tree)
                latent_target = target
                latent_vector = encoded_plan[0]
                indexes = encoded_plan[1]
                d = latent_vector.shape[1]

                decoded = model.decoder(encoded_plan[0].float(), indexes)

                platform_choices = list(
                    map(
                        lambda x: [int(v == max(x)) for v in x],
                        decoded[0][0].detach().clone().transpose(0, 1)
                    )
                )



                for choice in platform_choices:
                    print(choice)

                results.append(platform_choices)

        return results

def main_onnx(args):
    # set some defaults, highly WIP
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path=args.model_path
    parameters_path=args.parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, in_dim, out_dim = load_autoencoder_data(path=get_relative_path('30.txt', 'Data/splits/imdb/training'), retrain_path=args.retrain, device=device, num_ops=args.operators, num_platfs=args.platforms)

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

        onnx_model = onnx.load(model_path)

        ort_session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        options = ort_session.get_session_options()

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        dataloader = DataLoader(data, batch_size=1, drop_last=False, shuffle=False)

        results = []

        with torch.no_grad():
            for tree,target in dataloader:
                torch.set_printoptions(profile="full")
                input_value_name = ort_session.get_inputs()[0].name
                input_index_name = ort_session.get_inputs()[1].name
                output_name = ort_session.get_outputs()[0].name
                decoded = ort_session.run([output_name], {input_value_name: to_numpy(tree[0]), input_index_name: to_numpy(tree[1])})

                platform_choices = list(
                    map(
                        lambda x: [int(v == max(x)) for v in x],
                        torch.from_numpy(decoded[0][0]).detach().clone().transpose(0, 1)
                    )
                )

                results.append(platform_choices)
        return results

def get_platform_choices(file_path: str, line: int):
    regex_pattern = r'\(((?:[+,-]?\d+(?:,[+,-]?\d+)*)(?:\s*,\s*\(.*?\))*)\)'
    lines = []

    print(file_path)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    input, exec_plan, latency = lines[line].split(":")
    matches_iterator = re.finditer(regex_pattern, exec_plan)
    for match in matches_iterator:
        in_paranthesis = match.group()
        find = in_paranthesis.strip('(').strip(')')
        values = [int(num.strip()) for num in find.split(',')]
        platform_choices = values[43:43+9]
        #if sum(platform_choices) > 0:
        print(platform_choices)



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
    #onnx_plats = main_onnx(args)
    torch_plats = main(args)
    file_path = get_relative_path("30.txt", "Data/splits/imdb/training/")
    #clean_duplicate_platforms(file_path)
    get_platform_choices(file_path, 0)

    #assert onnx_plats == torch_plats
    #print(f"Is equal: {onnx_plats == torch_plats}")

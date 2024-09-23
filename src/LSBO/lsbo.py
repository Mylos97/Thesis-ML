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
import torch
import torch.onnx
from torch.utils.data import DataLoader
import json
from subprocess import PIPE, Popen

from helper import get_weights_of_model_by_path, set_weights, load_autoencoder_data_from_str

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize
from botorch import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from OurModels.EncoderDecoder.model import VAE
from OurModels.EncoderDecoder.bvae import BVAE
from Util.communication import read_int, UTF8Deserializer, dump_stream, open_connection

TIMEOUT = float(60 * 60 * 60)

class LSBOResult:
    def __init__(
        self,
        ml_model,
        model,
        model_results,
        tree,
        train_x,
        train_obj,
        state_dict,
        best_values
    ):
        self.ml_model = ml_model
        self.model = model
        self.model_results = model_results
        self.tree = tree
        self.train_x = train_x
        self.train_obj = train_obj
        self.state_dict = state_dict
        self.best_values = best_values

    def update(self, new_x, new_obj):
         # update training points
        self.train_x = torch.cat((self.train_x, new_x))
        self.train_obj = torch.cat((self.train_obj, new_obj))

        # update progress
        best_value = self.train_obj.max().item()
        self.best_values.append(best_value)

        self.state_dict = self.model.state_dict()

def latent_space_BO(ML_model, device, plan, args, previous: LSBOResult = None) -> LSBOResult:
    global initial_latency
    print('Running latent space Bayesian Optimization', flush=True)
    dtype = torch.float64
    for tree,target in plan:
        encoded_plan = ML_model.encoder(tree)
    latent_vector = encoded_plan[0]
    indexes = encoded_plan[1]
    d = latent_vector.shape[1]
    N_BATCH = 25
    BATCH_SIZE = 1
    NUM_RESTARTS = 10
    RAW_SAMPLES = 256
    MC_SAMPLES = 2000
    initial_latency = 0
    seed = 42
    is_initial_run = False
    latent_vector_sample = latent_vector[0].max().item()

    bounds = torch.tensor([[-(latent_vector_sample * 25_000)] * d, [latent_vector_sample * 25_000] * d], device=device, dtype=dtype)

    def get_latencies(plans) -> list[torch.Tensor]:
        global initial_latency
        results = []
        for plan in plans:
            #results.append(plan[0].sum().item())
            if initial_latency == 0:
                latency = get_plan_latency(args, plan)
            else:
                latency = initial_latency - get_plan_latency(args, plan)
            results.append(latency)

        #convert_to_json(plans)
        return results

    def objective_function(X):
        # Move the prediction made in latent_vector by some random v
        v_hat = [torch.add(v, latent_vector) for v in X]
        model_results = []

        for v in v_hat:
            decoded = ML_model.decoder(v.float(), indexes)
            model_results.append(decoded)
        results = get_latencies(model_results)

        return torch.tensor(results)

    def gen_initial_data(n: int = 1):
        global initial_latency
        print(f"Generating {n} initial samples")
        train_x = unnormalize(
            torch.rand(n, d, device=device, dtype=dtype),
            bounds=bounds)
        train_obj = objective_function(train_x).unsqueeze(-1)
        best_observed_value = train_obj.max().item()
        train_obj = torch.tensor([[0]])

        initial_latency = best_observed_value
        print(f"Finished generating {n} initial samples")
        print(f"Initial latency: {initial_latency}")

        return train_x, train_obj, train_obj


    def get_fitted_model(train_x, train_obj, state_dict=None):
        model = SingleTaskGP(
            train_X=normalize(train_x, bounds),
            train_Y=train_obj,
            outcome_transform=Standardize(m=1)
        )
        if state_dict is not None:
            model.load_state_dict(state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(train_x)
        print(f"Train x: {train_x}")
        fit_gpytorch_mll(mll)

        return model

    def optimize_acqf_and_get_observation(acq_func):
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack(
                [
                    torch.zeros(d, dtype=dtype, device=device),
                    torch.ones(d, dtype=dtype, device=device),
                ]
            ),
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        print(f"Candidates: {candidates}")

        new_x = unnormalize(candidates.detach(), bounds=bounds)
        new_obj = objective_function(new_x).unsqueeze(-1)

        return new_x, new_obj

    torch.manual_seed(seed)

    if previous is None:
        is_initial_run = True
        best_observed = []
        train_x, train_obj, best_value = gen_initial_data()
        best_observed.append(best_value)
        state_dict = None
        model_results = []

        previous = LSBOResult(ML_model, None, model_results, tree, train_x, train_obj, state_dict, best_observed)

    for iteration in range(N_BATCH):

        model = get_fitted_model(
            train_x=previous.train_x,
            train_obj=previous.train_obj.double(),
            state_dict=previous.state_dict,
        )

        previous = LSBOResult(ML_model, model, previous.model_results, tree, previous.train_x, previous.train_obj, previous.state_dict, previous.best_values)

        print(f"Best f: {previous.train_obj.max()}")

        qmc_sampler = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]), seed=seed)
        qEI = qLogExpectedImprovement(
            model=model,
            sampler=qmc_sampler,
            best_f=previous.train_obj.max()
        )

        new_x, new_obj = optimize_acqf_and_get_observation(qEI)
        print(f"New_x: {new_x}")
        print(f"New_obj: {new_obj}")
        v_hat = [latent_vector + v for v in new_x]
        for v in v_hat:
            # Cast to float() because of warning from BoTorch
            decoded = ML_model.decoder(v.float(), indexes)
            #x = ML_model.softmax(decoded[0])
            previous.model_results.append([decoded[0].detach().numpy().tolist()[0], decoded[1].detach().numpy().tolist()[0]])

        if is_initial_run:
            previous.train_x = new_x
            previous.train_obj = new_obj
        else:
            previous.update(new_x, new_obj)

        print('Finish Bayesian Optimization for latent space', flush=True)

    return previous


def run_lsbo(input, args, previous: LSBOResult = None):
    print(f"Starting LSBO from python")

    # set some defaults, highly WIP
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path= f"{dir_path}/../Models/bvae.onnx"
    surrogate_path = f"{dir_path}/../Models/vae-surrogate.onnx"
    parameters_path = f"{dir_path}/../HyperparameterLogs/BVAE.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rather use a surrogate if it already exists
    if os.path.isfile(surrogate_path):
        print(f"Using existing surrogate model at {surrogate_path}", flush=True)
        model_path = surrogate_path

    data, in_dim, out_dim = load_autoencoder_data_from_str(
        device=device,
        data=input,
    )

    # find model parameters
    with open(parameters_path) as file:
        parameters = json.load(file)
        lr = parameters.get("lr", 0.001)
        gradient_norm = parameters.get("gradient_norm", 1.0)
        dropout = parameters.get("dropout", 0.1)
        z_dim = parameters.get("z_dim", 16)
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
        model.eval()

        dataloader = DataLoader(data, batch_size=1, drop_last=False, shuffle=True)
        lsbo_result = latent_space_BO(model, device, dataloader, args, previous)

    return lsbo_result

def get_plan_latency(args, sampled_plan) -> float:
    global TIMEOUT

    process = Popen([
        args.exec,
        args.namespace,
        args.args
    ], stdout=PIPE)

    """
    The first message received in the stdout should be the socket_port
    """

    socket_port = int(process.stdout.readline().rstrip().decode("utf-8"))
    print(socket_port)
    process.stdout.flush()

    sock_file, sock = open_connection(socket_port)

    plan = read_from_wayang(sock_file)

    print("Sending sampled plan back to Wayang")

    input_plan = [sampled_plan[0].detach().numpy().tolist()[0], sampled_plan[1].detach().numpy().tolist()[0]]
    dump_stream(iterator=[input_plan], stream=sock_file)

    sock_file.flush()

    print("Sent sampled plan back to Wayang")

    print(process.stdout.read())
    process.stdout.flush()

    try:
        if process.wait(TIMEOUT) != 0:
            print("Error closing Wayang process!")

            exec_time = int(TIMEOUT * 100000)
            return exec_time

    except Exception:
        print("Didnt finish fast enough")

        exec_time = int(TIMEOUT * 100000)

        return exec_time

    input, picked_plan, exec_time_str = read_from_wayang(sock_file).split(":")

    exec_time = int(exec_time_str)

    print(exec_time)

    return exec_time


def request_wayang_plan(args, lsbo_result: LSBOResult = None, index: int = 0, timeout: float = 3600) -> LSBOResult:
    global TIMEOUT
    TIMEOUT = timeout

    process = Popen([
        args.exec,
        args.namespace,
        args.args
    ], stdout=PIPE)

    """
    The first message received in the stdout should be the socket_port
    """

    socket_port = int(process.stdout.readline().rstrip().decode("utf-8"))
    print(socket_port)
    process.stdout.flush()

    sock_file, sock = open_connection(socket_port)

    plan = read_from_wayang(sock_file)

    # This holds plenty of metadata for multiple runs
    # and updating the actual latency of plans
    lsbo_result = run_lsbo([plan], args, lsbo_result)

    """
    print("Sending sampled plan back to Wayang")

    dump_stream(iterator=[lsbo_result.model_results[0]], stream=sock_file)

    for plan in lsbo_result.model_results:
        print(plan)

    sock_file.flush()

    print("Sent sampled plan back to Wayang")
    """

    print(process.stdout.read())
    process.stdout.flush()

    try:
        if process.wait() != 0:
            print("Error closing Wayang process!")

            exec_time = int(timeout * 100000)
            lsbo_result.train_obj[index][0] = exec_time

            return lsbo_result, ("", "", exec_time)
    except Exception:
        print("Didnt finish fast enough")

        exec_time = int(timeout * 100000)
        lsbo_result.train_obj[index][0] = exec_time

        return lsbo_result, ("", "", exec_time)

    """
    input, picked_plan, exec_time_str = read_from_wayang(sock_file).split(":")

    exec_time = int(exec_time_str)

    print(float(exec_time))

    lsbo_result.train_obj[index][0] = exec_time
    print(f"Train_obj: {lsbo_result.train_obj}")

    plan_data = (input, picked_plan, exec_time_str)
    """
    plan_data = (input, input, input)

    return lsbo_result, plan_data

def read_from_wayang(sock_file):
    udf_length = read_int(sock_file)
    serialized_udf = sock_file.read(udf_length)
    iterator = UTF8Deserializer().load_stream(sock_file)
    return next(iterator)


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
import datetime

from helper import get_weights_of_model_by_path, set_weights, load_autoencoder_data_from_str

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

# Set to 10min (600 seconds)
TIMEOUT = float(60 * 10)
PLAN_CACHE = set()
best_plan_data = None
z_dim = 31
distinct_choices = []
PLAN_SIZE = 0

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

    def update(self, new_x, new_obj, state_dict):
         # update training points
        self.train_x = torch.cat((self.train_x, new_x))
        self.train_obj = torch.cat((self.train_obj, new_obj))

        # update progress
        best_value = self.train_obj.max().item()
        self.best_values.append(best_value)

        self.state_dict = state_dict

def latent_space_BO(ML_model, device, plan, args, previous: LSBOResult = None):
    global initial_latency

    print('Running latent space Bayesian Optimization', flush=True)
    dtype = torch.float64
    latent_target = None
    with torch.no_grad():
        for tree,target in plan:
            encoded_plan = ML_model.encoder(tree)
            #softmaxed = ML_model.enc_softmax(encoded_plan[0])
            latent_target = target
        latent_vector = encoded_plan[0]
        print(f"Tree shape : {tree[0].shape}")
        indexes = encoded_plan[1]
        print(f"Indexes: {indexes.shape}")
        d = latent_vector.shape[1]
    #N_BATCH = 100
    BATCH_SIZE = 1
    NUM_RESTARTS = 10
    RAW_SAMPLES = 256
    MC_SAMPLES = 2048
    initial_latency = 0
    seed = 42
    latent_vector_sample = latent_vector[0].max().item()

    bounds = torch.tensor([[-6] * z_dim, [6] * z_dim], device=device, dtype=dtype)
    #bounds = torch.tensor([[-100] * z_dim, [100] * z_dim], device=device, dtype=dtype)
    #bounds = torch.tensor([[-6_000_000] * d, [6_000_000] * d], device=device, dtype=dtype)
    #bounds = torch.tensor([[-(latent_vector_sample)] * d, [latent_vector_sample] * d], device=device, dtype=dtype)
    #bounds = torch.stack([torch.zeros(d), torch.ones(d)]).to(device)

    def get_latencies(plans) -> list[torch.Tensor]:
        results = []
        for plan in plans:
            if plan is not None:
                latency = get_plan_latency(args, plan)
                #latency = random.randrange(1,100000)
                results.append(latency)
            else:
                results.append(initial_latency)

        return results

    def objective_function(X, initial = False):
        #with torch.no_grad():
        # Move the prediction made in latent_vector by some random v
        global distinct_choices
        global PLAN_SIZE

        """
        if not initial:
            v_hat = [torch.add(latent_vector, torch.tensor(v)) for v in X]
        else:
            v_hat = [latent_vector]
        """

        model_results = []

        no_distinct_plans_before = len(distinct_choices)

        for new_plan in X:
            decoded = ML_model.decoder(new_plan.float(), indexes)

            softmaxed = ML_model.softmax(decoded[0])
            #model_results.append([decoded[0].tolist()[0], decoded[1].tolist()[0]])

            platform_choices = list(
                map(
                    lambda x: [int(v == max(x)) for v in x],
                    softmaxed[0].detach().clone().transpose(0, 1)
                )
            )

            """
            if PLAN_SIZE > 0:
                platform_choices = platform_choices[0:PLAN_SIZE+1]
            """

            discovered_latent_vector = [softmaxed.tolist()[0], decoded[1].tolist()[0]]

            #print(f"neg log likelihood: {-F.cross_entropy(softmaxed, latent_target)}")

            # Only append and test new plans
            if platform_choices not in distinct_choices:
                distinct_choices.append(platform_choices)
                model_results.append(discovered_latent_vector)
            else:
                model_results.append(None)

        no_distinct_plans_after = len(distinct_choices)
        print(f"Generated {no_distinct_plans_after - no_distinct_plans_before} new plans")

        #assert no_distinct_plans_after > no_distinct_plans_before, f'No new plans generated, {len(distinct_choices)} total, aborting'

        latencies = get_latencies(model_results)

        improvements = get_improvements_from_latencies(latencies)
        print(f"Improvements: {improvements}")

        return torch.tensor(improvements, dtype=dtype)

    def get_improvements_from_latencies(latencies: list) -> list:
        global initial_latency
        if initial_latency == 0:
            print(f"Set initial latency: {latencies}")
            return latencies

        improvements = list(map(lambda latency: initial_latency - latency, latencies))

        def get_impr(improvement: float) -> float:
            if improvement > 0:

                return math.log(improvement)
            elif improvement < 0:

                return -1 * math.log(abs(improvement))
            else:

                return 0

        return list(
                map(lambda improvement: get_impr(improvement),
                    improvements
                )
        )

        #return list(map(lambda latency: math.log(max(initial_latency - latency, 1)), latencies))


    def gen_initial_data(plan, n: int = 3):
        global initial_latency
        initial_latency = objective_function([plan], True).unsqueeze(-1).min().item()

        train_x = unnormalize(
            torch.rand(n, d, device=device, dtype=dtype),
            bounds=bounds)

        # Move the prediction made in latent_vector by some random x
        candidates = [torch.add(latent_vector, x.clone().detach()) for x in train_x]
        #print(f"Initial train_x: {candidates}")

        train_obj = objective_function(candidates).unsqueeze(-1)
        #print(f"Train_obj: {train_obj}")
        best_observed_value = train_obj.max().item()
        #train_obj = torch.tensor([[0]])

        #initial_latency = best_observed_value
        print(f"Finished generating {n} initial samples")
        print(f"Initial latency: {initial_latency}")

        return train_x, train_obj, best_observed_value


    def get_fitted_model(train_x, train_obj, state_dict=None):
        model = SingleTaskGP(
            #train_X=normalize(train_x, bounds),
            train_X=train_x,
            train_Y=train_obj,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1)
        )
        if state_dict is not None:
            model.load_state_dict(state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(train_x)
        fit_gpytorch_mll(mll)

        return model

    def optimize_acqf_and_get_observation(acq_func):
        """
        Xinit = gen_batch_initial_conditions(
            acq_func, bounds, q=BATCH_SIZE, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES
        )

        batch_candidates, batch_acq_values = gen_candidates_torch(
            initial_conditions=Xinit,
            acquisition_function=qEI,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
        )

        candidates = get_best_candidates(batch_candidates, batch_acq_values)
        """

        x_center = previous.train_x[previous.train_obj.argmax(), :].clone()
        weights = torch.ones_like(x_center)*8 # less than 4 stdevs on either side max
        tr_lb = x_center - weights * 59 / 2.0
        tr_ub = x_center + weights * 59 / 2.0

        # optimize
        candidates, expected = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            #bounds=torch.stack([tr_lb, tr_ub]),
            #bounds = torch.tensor([[0.], [1.]]),
            #bounds=torch.stack(
            #    [
            #        torch.zeros(d, dtype=dtype, device=device),
            #        torch.ones(d, dtype=dtype, device=device),
            #    ]
            #),
            q=100,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            return_best_only=True,
        )

        new_x = unnormalize(candidates.detach(), bounds=bounds)
        #candidates = [torch.tensor(x).unsqueeze(0) for x in new_x]
        candidates = [torch.add(latent_vector, x.clone().detach()) for x in new_x]
        #new_x = candidates.detach()
        print(f"new_x: {candidates}")
        print(f"expected improvements: {expected}")
        new_obj = objective_function(candidates).unsqueeze(-1)

        """
        index, best_impr = max(enumerate(previous.train_obj), key=lambda x: x[1])
        best_plan = previous.train_x[index]
        # recenter the latent vector
        latent_vector = best_plan
        """

        return new_x, new_obj

    torch.manual_seed(seed)

    if previous is None:
        best_observed = []
        print(f"Intial encoded plan: {encoded_plan[0]}")
        print(f"Intial encoded plan dims: {len(encoded_plan[0][0])}")
        latent_space_vector = encoded_plan[0][0]
        assert len(latent_space_vector) == z_dim
        train_x, train_obj, best_value = gen_initial_data(encoded_plan[0])
        print(f"Train_x: {train_x}")
        best_observed.append(best_value)
        state_dict = None
        model_results = []

        previous = LSBOResult(ML_model, None, model_results, tree, train_x, train_obj, state_dict, best_observed)

    """
    t = threading.Timer(args.time * 60, set_time_limit_reached)
    t.start()
    """

    #for iteration in range(N_BATCH):
    criteria = StoppingCriteria(args.time * 60, args.improvement, initial_latency)

    criteria.start_timer()

    while not criteria.is_met():

        model = get_fitted_model(
            previous.train_x,
            previous.train_obj,
            previous.state_dict,
        )

        previous = LSBOResult(ML_model, model, previous.model_results, tree, previous.train_x, previous.train_obj, previous.state_dict, previous.best_values)

        print(f"Best f: {previous.train_obj.max()}")

        #sampler = StochasticSampler(sample_shape=torch.Size([MC_SAMPLES]))
        #sampler = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]))

        qEI = qLogExpectedImprovement(
            model=model,
            #sampler=sampler,
            #best_f=max(previous.train_obj.max(), 0)
            best_f=previous.train_obj.max()
        )

        new_x, new_obj = optimize_acqf_and_get_observation(qEI)
        previous.update(new_x, new_obj, model.state_dict())

        index, best_impr = max(enumerate(previous.train_obj), key=lambda x: x[1])
        criteria.step(best_impr.item())

    print('Finish Bayesian Optimization for latent space', flush=True)

    index, best_impr = max(enumerate(previous.train_obj), key=lambda x: x[1])

    best_plan = previous.train_x[index]
    best_latency = initial_latency - math.pow(math.e ,best_impr.item())
    print(f"{best_latency} = {initial_latency} - {math.pow(math.e, best_impr.item())}")

    criteria.stop_timer()

    return best_plan, best_latency


def run_lsbo(input, args, previous: LSBOResult = None):
    global z_dim
    print(f"Starting LSBO from python")

    # set some defaults, highly WIP
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path=args.model_path
    parameters_path=args.parameters
    z_dim = args.zdim
    #model_path= f"{dir_path}/../Models/bvae.onnx"
    #parameters_path = f"{dir_path}/../HyperparameterLogs/BVAE.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

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

        dataloader = DataLoader(data, batch_size=1, drop_last=False, shuffle=False)
        lsbo_result = latent_space_BO(model, device, dataloader, args, previous)

    return lsbo_result

def get_plan_latency(args, sampled_plan) -> float:
    global TIMEOUT
    global PLAN_CACHE
    global PLAN_SIZE
    global best_plan_data
    global initial_latency

    try:
        process = Popen([
            args.exec,
            args.memory,
            args.namespace,
            args.args,
            str(args.query)
        ], stdout=PIPE, stderr=PIPE)


        """
        The first message received in the stdout should be the socket_port
        """

        socket_port = int(process.stdout.readline().rstrip().decode("utf-8"))
        print(f"Socket port {socket_port}")
        process.stdout.flush()

        sock_file, sock = open_connection(socket_port)

        _ = read_from_wayang(sock_file)

        input_plan = sampled_plan
        dump_stream(iterator=[input_plan], stream=sock_file)

        sock_file.flush()

        plan_out = ""
        counter = 0
        for line in iter(process.stdout.readline, b''):
            line_str = line.rstrip().decode('utf-8')
            if line_str.startswith("Encoding while choices: "):
                plan_out += line_str
                counter += 1
            elif line_str.startswith("DECODED"):
                print(f"[{datetime.datetime.now()}] Decoded WayangPlan received on Java side")
                break
            elif plan_out != "":
                break

        PLAN_SIZE = counter

        if plan_out in PLAN_CACHE:
            print(f"[{datetime.datetime.now()}] Seen this plan before")
            os.system("pkill -TERM -P %s"%process.pid)
            sock_file.close()
            sock.close()

            #exec_time = int(TIMEOUT * 10_000)
            return initial_latency

        print(f"[{datetime.datetime.now()}] Sampling new plan")

        PLAN_CACHE.add(plan_out)
        process.stdout.flush()

        print(f"[{datetime.datetime.now()}] Starting execution with {TIMEOUT} seconds max.")
        out, err = process.communicate(timeout=TIMEOUT)
        #if process.wait(TIMEOUT) != 0:
        print(f"Out: {out}")
        print(f"Err: {err}")
        if err != b'' and err.decode('utf-8').split(' ', 1)[0] != 'WARNING:' and err.decode('utf-8').split(' ', 1)[0] != 'SLF4J:':
            print("Error closing Wayang process!")

            exec_time = int(TIMEOUT * 100000)
            return exec_time

        input, picked_plan, exec_time_str = read_from_wayang(sock_file).split(":")
        sock_file.close()
        sock.close()

        exec_time = int(exec_time_str)

        # Calculate the current set timeout in ms (convert from sec to ms)
        ms_timeout = TIMEOUT * 1000
        if ms_timeout > exec_time:
            TIMEOUT = int(exec_time / 1000)
            print(f"[{datetime.datetime.now()}] Found better plan, updating timeout: {TIMEOUT} sec")
            best_plan_data = input, picked_plan, exec_time_str

        print(exec_time)

        return exec_time

    except Exception as e:
        print(f"Exception: {e}")
        print("Didnt finish fast enough")
        sock_file.close()
        sock.close()
        process.kill()
        out, err = process.communicate()
        #os.system("pkill -TERM -P %s"%process.pid)
        #os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        #process.wait()

        exec_time = int(TIMEOUT * 100000)

        return exec_time


def request_wayang_plan(args, lsbo_result: LSBOResult = None, timeout: float = 3600):
    global TIMEOUT
    global best_plan_data
    #TIMEOUT = timeout

    print(f"Requesting plan for query: {args.query}")

    process = Popen([
        args.exec,
        args.memory,
        args.namespace,
        args.args,
        str(args.query)
    ], stdout=PIPE)

    """
    The first message received in the stdout should be the socket_port
    """

    socket_port = int(process.stdout.readline().rstrip().decode("utf-8"))
    process.stdout.flush()
    print(f"Socket on {socket_port}")

    sock_file, sock = open_connection(socket_port)

    plan = read_from_wayang(sock_file)

    # This holds plenty of metadata for multiple runs
    # and updating the actual latency of plans
    lsbo_result = run_lsbo([plan], args, lsbo_result)

    process.kill()

    return best_plan_data, initial_latency, PLAN_CACHE

def read_from_wayang(sock_file):
    udf_length = read_int(sock_file)
    serialized_udf = sock_file.read(udf_length)
    iterator = UTF8Deserializer().load_stream(sock_file)
    next_val = next(iterator, None)
    if next_val is None:
        print("None return from read_from_wayang")
    else:
        return next(iterator)


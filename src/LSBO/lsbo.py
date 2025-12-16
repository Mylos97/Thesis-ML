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
from subprocess import PIPE, Popen, TimeoutExpired
import signal
import threading
import random
import math
import datetime
import onnx
import onnxruntime

from helper import get_weights_of_model_by_path, set_weights, load_autoencoder_data_from_str, load_autoencoder_data
from .state import State

from torch.quasirandom import SobolEngine

from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize, Normalize
from botorch.models.transforms.input import InputStandardize
from botorch import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.optim import optimize_acqf, gen_batch_initial_conditions
from botorch.generation.gen import get_best_candidates, gen_candidates_torch
from OurModels.EncoderDecoder.model import VAE
from OurModels.EncoderDecoder.bvae import BVAE
from Util.communication import read_int, UTF8Deserializer, dump_stream, open_connection
from LSBO.criteria import StoppingCriteria

# Set to 30min (1800 seconds)
TIMEOUT = float(60 * 180)
PLAN_CACHE = set()
EXECUTABLE_PLANS = set()
VALID_X = set()
best_plan_data = None
z_dim = 31
distinct_choices = []
PLAN_SIZE = 0

seed = 42
torch.manual_seed(seed)

def latent_space_BO(ML_model, device, plan, args, state: State = None):
    global initial_latency
    global VALID_X

    print('Running latent space Bayesian Optimization', flush=True)
    dtype = torch.float64
    latent_target = None

    with torch.no_grad():
        for tree,target in plan:
            print(f"Input: {tree[0].shape}")
            torch.set_printoptions(profile="full")
            #print(f"Tree: {tree[0]}")
            encoded_plan = ML_model.encoder(tree)
            #softmaxed = ML_model.enc_softmax(encoded_plan[0])
            latent_target = target
        latent_vector = encoded_plan[0]
        print(f"Tree shape : {tree[0].shape}")
        indexes = encoded_plan[1]
        print(f"Indexes python: {indexes.shape}")
        d = latent_vector.shape[1]
    #N_BATCH = 100
    BATCH_SIZE = 1
    NUM_RESTARTS = 10
    RAW_SAMPLES = 256
    MC_SAMPLES = 2048
    initial_latency = 0
    latent_vector_sample = latent_vector[0].max().item()

    bounds = torch.tensor([[-6] * z_dim, [6] * z_dim], device=device, dtype=dtype)
    #bounds = torch.tensor([[-100] * z_dim, [100] * z_dim], device=device, dtype=dtype)
    #bounds = torch.tensor([[-6_000_000] * d, [6_000_000] * d], device=device, dtype=dtype)
    #bounds = torch.tensor([[-(latent_vector_sample)] * d, [latent_vector_sample] * d], device=device, dtype=dtype)
    #bounds = torch.stack([torch.zeros(d), torch.ones(d)]).to(device)

    def get_latencies(plans, duplicates: list) -> list[torch.Tensor]:
        global EXECUTABLE_PLANS
        global VALID_X
        results = []

        for i, plan in enumerate(plans):
            #if plan is not None:
            len_executables_before = len(EXECUTABLE_PLANS)
            if plan not in duplicates:
                latency = get_plan_latency(args, plan)
                if len_executables_before < len(EXECUTABLE_PLANS): # plan was valid
                    VALID_X.add(i)
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

        duplicate_plans = []

        if not initial:
            v_hat = [torch.add(latent_vector, v.clone().detach()) for v in X]
            #v_hat = X
        else:
            v_hat = [latent_vector]

        model_results = []

        no_distinct_plans_before = len(distinct_choices)

        for new_plan in v_hat:
            decoded = ML_model.decoder(new_plan.float(), indexes)

            #model_results.append([decoded[0].tolist()[0], decoded[1].tolist()[0]])

            debug_decoded = list(
                map(
                    lambda x: [float(v) for v in x],
                    decoded[0][0].detach().clone().transpose(0, 1)
                )
            )

            platform_choices = list(
                map(
                    lambda x: [int(v == max(x)) for v in x],
                    decoded[0][0].detach().clone().transpose(0, 1)
                )
            )

            """
            print(f"Decoded: {debug_decoded}")
            print(f"Indexes python: {indexes}")
            print(f"Platforms: {platform_choices}")
            """


            """
            if PLAN_SIZE > 0:
                platform_choices = platform_choices[0:PLAN_SIZE+1]
            """

            discovered_latent_vector = [decoded[0].tolist()[0], decoded[1].tolist()[0]]

            #print(f"neg log likelihood: {-F.cross_entropy(softmaxed, latent_target)}")

            """
            # Only append and test new plans
            if platform_choices not in distinct_choices:
                distinct_choices.append(platform_choices)
            else:
                model_results.append(None)
                duplicate_plans.append(discovered_latent_vector)
            """

            model_results.append(discovered_latent_vector)


        #no_distinct_plans_after = len(distinct_choices)
        #print(f"Generated {no_distinct_plans_after - no_distinct_plans_before} new plans")
        print(f"Generated {len(model_results)} new plans")

        #assert no_distinct_plans_after > no_distinct_plans_before, f'No new plans generated, {len(distinct_choices)} total, aborting'

        latencies = get_latencies(model_results, duplicate_plans)

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


    def gen_initial_data(plan, n: int = 10):
        global initial_latency
        initial_latency = objective_function([plan], True).unsqueeze(-1).min().item()

        #if init == "random":
        train_x = unnormalize(
            torch.rand(n, d, device=device, dtype=dtype),
            bounds=bounds)

        # Move the prediction made in latent_vector by some random x
        #candidates = [torch.add(latent_vector, x.clone().detach()) for x in train_x]
        candidates = train_x
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

        """
        model = SaasFullyBayesianSingleTaskGP(
            train_x.to(device),
            train_obj.to(device),
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1)
        )
        model.to(device)
        fit_fully_bayesian_model_nuts(model)
        """

        return model

    def optimize_acqf_and_get_observation(acq_func, args):
        global initial_latency

        x_center = state.train_x[state.train_obj.argmax(), :].clone()
        x_range = state.train_x.max().item() - state.train_x.min().item()
        x_range = max(x_range, 8.0)
        weights = torch.ones_like(x_center)
        weights = weights * x_range # less than 4 stdevs on either side max
        #tr_lb = x_center - weights # effectively the entire region
        #tr_ub = x_center + weights
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0

        new_bounds = torch.stack([tr_lb, tr_ub])

        if args.acqf == "ei":
            # optimize
            print(f"[{datetime.datetime.now()}] Starting gen candidates")
            candidates, expected = optimize_acqf(
                acq_function=acq_func,
                bounds=new_bounds,
                q=10,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
        elif args.acqf == "ts":
            sobol = SobolEngine(args.zdim, scramble=True)
            pert = sobol.draw(10).to(dtype=dtype).to(device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / args.zdim, 1.0)
            mask = torch.rand(10, args.zdim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, args.zdim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(10, args.zdim).clone()
            X_cand[mask] = pert[mask]
            try:
                with torch.no_grad():
                    candidates = acqf(X_cand, num_samples=10)
            except:  # noqa: E722
                # Sampling entirely failed, return first candidate
                candidates = X_cand[0].unsqueeze(0)
        elif args.acqf == "random":
            candidates = unnormalize(
                torch.rand(10, z_dim, device=device, dtype=dtype),
                bounds=new_bounds)

        print(f"[{datetime.datetime.now()}] Finished gen candidates")
        new_x = unnormalize(candidates.detach(), bounds=bounds)
        candidates = [torch.add(latent_vector, x.clone().detach()) for x in new_x]
        print(f"[{datetime.datetime.now()}] Starting objective_function on candidates")
        new_obj = objective_function(candidates).unsqueeze(-1)
        print(f"[{datetime.datetime.now()}] Finished objective_function on candidates")

        return new_x, new_obj

    if state is None:
        best_observed = []
        print(f"Intial encoded plan: {encoded_plan[0]}")
        print(f"Intial encoded plan dims: {len(encoded_plan[0][0])}")
        latent_space_vector = encoded_plan[0][0]
        assert len(latent_space_vector) == z_dim
        train_x, train_obj, best_value = gen_initial_data(encoded_plan[0])
        best_observed.append(best_value)
        state_dict = None
        model_results = []

        state = State(initial_latency, ML_model, None, model_results, tree, train_x, train_obj, state_dict, best_observed)
        VALID_X = set()

    criteria = StoppingCriteria(args.time * 60, args.improvement, initial_latency, args.steps)

    criteria.start_timer()

    while not criteria.is_met():

        model = get_fitted_model(
            state.train_x,
            state.train_obj,
            state.state_dict,
        )

        print("Overwriting state")
        state.update_model(model)

        print(f"Best f: {state.train_obj.max()}")

        if args.acqf == "ei":
            acqf = qLogExpectedImprovement(
                model=model,
                best_f=state.train_obj.max()
            )
        elif args.acqf == "ts":
            acqf = MaxPosteriorSampling(model=model, replacement=False,)
        elif args.acqf == "random":
            acqf = "random"


        new_x, new_obj = optimize_acqf_and_get_observation(acqf, args)
        state.update(new_x, new_obj, model.state_dict(), VALID_X)
        # reset the global set
        VALID_X = set()

        index, best_impr = max(enumerate(state.train_obj), key=lambda x: x[1])
        criteria.step(best_impr.item(), new_x.shape[0])

    print('Finish Bayesian Optimization for latent space', flush=True)

    index, best_impr = max(enumerate(state.train_obj), key=lambda x: x[1])

    best_plan = state.train_x[index]
    best_latency = initial_latency - math.pow(math.e ,best_impr.item())
    print(f"{best_latency} = {initial_latency} - {math.pow(math.e, best_impr.item())}")

    criteria.stop_timer()

    return best_plan, best_latency


def run_lsbo(input, args, state: State = None):
    global z_dim
    print(f"Starting LSBO from python")

    # set some defaults, highly WIP
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path=args.model_path
    parameters_path=args.parameters
    z_dim = args.zdim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        state = latent_space_BO(model, device, dataloader, args, state)

    return state

def get_plan_latency(args, sampled_plan) -> float:
    global TIMEOUT
    global PLAN_CACHE
    global PLAN_SIZE
    global EXECUTABLE_PLANS
    global best_plan_data
    global initial_latency

    try:
        process = Popen([
            args.exec,
            args.memory,
            args.namespace,
            args.args,
            str(args.query)
        ], stdout=PIPE, stderr=PIPE, start_new_session=True)


        """
        The first message received in the stdout should be the socket_port
        """

        socket_port = int(process.stdout.readline().rstrip().decode("utf-8"))
        print(f"Socket port {socket_port}")

        print(f"[{datetime.datetime.now()}] Opening socket connection")
        sock_file, sock = open_connection(socket_port)

        print(f"[{datetime.datetime.now()}] Reading from wayang")
        _ = read_from_wayang(sock_file)

        input_plan = sampled_plan
        dump_stream(iterator=[input_plan], stream=sock_file)
        print(f"[{datetime.datetime.now()}] Wrote to wayang")

        sock_file.flush()

        plan_out = ""
        counter = 0
        for line in iter(process.stdout.readline, b''):
            line_str = line.rstrip().decode('utf-8')
            if line_str.startswith("Nulling psql choice"):
                print(line_str)
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
            print(f"[{datetime.datetime.now()}] Closing sock_file")
            sock_file.close()
            print(f"[{datetime.datetime.now()}] Closing sock")
            sock.close()
            print(f"[{datetime.datetime.now()}] Killing process")
            process.stdout.close()
            process.stderr.close()
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.kill()
            print(f"[{datetime.datetime.now()}] Awaiting process.wait")
            process.wait(timeout=5)
            print(f"[{datetime.datetime.now()}] process.wait finished")

            return initial_latency

        print(f"[{datetime.datetime.now()}] Sampling new plan")

        PLAN_CACHE.add(plan_out)

        print(f"[{datetime.datetime.now()}] Starting execution with {TIMEOUT} seconds max.")

        if process.wait() != 0:
            print("Error closing Wayang process!")
            print(f"[{datetime.datetime.now()}] Closing sock_file")
            sock_file.close()
            print(f"[{datetime.datetime.now()}] Closing sock")
            sock.close()
            print(f"[{datetime.datetime.now()}] Killing process")
            process.stdout.close()
            process.stderr.close()
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.kill()
            print(f"[{datetime.datetime.now()}] Awaiting process.wait")
            process.wait(timeout=5)
            print(f"[{datetime.datetime.now()}] process.wait finished")

            exec_time = int(TIMEOUT * 100000)
            return exec_time

        input, picked_plan, exec_time_str = read_from_wayang(sock_file).split(":")
        sock_file.close()
        sock.close()


        exec_time = int(exec_time_str)

        if exec_time < sys.maxsize:
            print("Add an executable plan")
            EXECUTABLE_PLANS.add(plan_out)

        # Calculate the current set timeout in ms (convert from sec to ms)
        ms_timeout = TIMEOUT * 1000
        if ms_timeout > exec_time:
            TIMEOUT = int(exec_time / 1000)
            print(f"[{datetime.datetime.now()}] Found better plan, updating timeout: {TIMEOUT} sec")
            best_plan_data = input, picked_plan, exec_time_str

        print(exec_time)

        return exec_time

    except TimeoutExpired as e:
        print(f"Exception: {e}")
        print("Didnt finish fast enough")

        print(f"[{datetime.datetime.now()}] Seen this plan before")
        print(f"[{datetime.datetime.now()}] Closing sock_file")
        sock_file.close()
        print(f"[{datetime.datetime.now()}] Closing sock")
        sock.close()
        print(f"[{datetime.datetime.now()}] Killing process")
        process.stdout.close()
        process.stderr.close()
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        print(f"[{datetime.datetime.now()}] Awaiting process.wait")
        process.wait(timeout=5)
        print(f"[{datetime.datetime.now()}] process.wait finished")

        EXECUTABLE_PLANS.add(plan_out)

        exec_time = initial_latency

        return exec_time

    except Exception as e:
        # In case the underlying process died
        print(f"Exception: {e}")
        exec_time = int(TIMEOUT * 100000)

        return exec_time


def request_wayang_plan(args, state: State = None, timeout: float = 3600):
    global TIMEOUT
    global best_plan_data
    TIMEOUT = timeout

    print(f"Requesting plan for query: {args.query}")

    process = Popen([
        args.exec,
        args.memory,
        args.namespace,
        args.args,
        str(args.query)
    ], stdout=PIPE, stderr=PIPE, start_new_session=True)

    """
    The first message received in the stdout should be the socket_port
    """

    socket_port = int(process.stdout.readline().rstrip().decode("utf-8"))
    process.stdout.flush()
    print(f"Socket on {socket_port}")

    print(f"[{datetime.datetime.now()}] Opening socket connection")
    sock_file, sock = open_connection(socket_port)

    print(f"[{datetime.datetime.now()}] Reading plan from wayang")
    plan = read_from_wayang(sock_file)

    # This holds plenty of metadata for multiple runs
    # and updating the actual latency of plans
    state = run_lsbo([plan], args, state)

    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.kill()

    return best_plan_data, initial_latency, EXECUTABLE_PLANS

def read_from_wayang(sock_file):
    udf_length = read_int(sock_file)
    serialized_udf = sock_file.read(udf_length)
    iterator = UTF8Deserializer().load_stream(sock_file)
    next_val = next(iterator, None)
    if next_val is None:
        print("None return from read_from_wayang")
    else:
        return next_val


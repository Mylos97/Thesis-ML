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

from helper import get_weights_of_model_by_path, set_weights, load_autoencoder_data_from_str, load_autoencoder_data, load_autoencoder_carb_data
from .state import State

from torch.quasirandom import SobolEngine

from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize, Normalize
from botorch.models.transforms.input import InputStandardize
from botorch import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from botorch.acquisition import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.optim import optimize_acqf, gen_batch_initial_conditions
from botorch.generation.gen import get_best_candidates, gen_candidates_torch
from OurModels.EncoderDecoder.model import VAE
from OurModels.EncoderDecoder.bvae import BVAE
from OurModels.EncoderDecoder.betaCVAE.model import BetaCVAE
from OurModels.EncoderDecoder.carbVAE.model import CarbVAE
from Util.communication import read_int, UTF8Deserializer, dump_stream, open_connection
from LSBO.criteria import StoppingCriteria

global duplicates_counter
# Set to 30min (1800 seconds)
TIMEOUT = float(60 * 180)
PLAN_IMPROVEMENT_CACHE = {}
EXECUTABLE_PLANS = set()
best_plan_data = None
z_dim = 31
INVALID_PENALTY = 1e6
duplicates_counter = 0

#N_BATCH = 100
BATCH_SIZE = 25
NUM_RESTARTS = 20
RAW_SAMPLES = 1024
MC_SAMPLES = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


seed = 42
torch.manual_seed(seed)

def latent_space_BO(ML_model, device, plan, args, state: State = None):
    global duplicates_counter

    print('Running latent space Bayesian Optimization', flush=True)

    latent_bounds = torch.tensor([[-4] * z_dim, [4] * z_dim], device=device, dtype=dtype)
    norm_bounds = torch.stack([
        torch.zeros(z_dim),
        torch.ones(z_dim)
    ]).to(device=device, dtype=dtype)

    def get_latencies(plans) -> list[torch.Tensor]:
        results = []

        for i, plan in enumerate(plans):
            latency = get_plan_latency(args, plan)
            results.append(latency)

        return results

    def objective_function(logical_plan, candidate_zs):
        model_results = []

        for z in candidate_zs:
            node_feats, children = logical_plan
            phys_logits = ML_model.sample(logical_plan, z)

            model_results.append([phys_logits[0].transpose(0, 1).tolist(), logical_plan[1].tolist()[0]])

        print(f"Generated {len(model_results)} new plans")

        latencies = get_latencies(model_results)

        # objective if f(z) = -latency
        train_objs = list(map(lambda x: -x, latencies))
        print(f"f(z)'s: {train_objs}")

        return torch.tensor(train_objs, dtype=dtype)

    def gen_initial_data(
            logical_plan: torch.Tensor,
            model: CarbVAE,
            latent_bounds: torch.Tensor,
            initialization: str = 'random',
            latency_estimation_scale: float = 1.5
        ):
        global duplicates_counter
        global BATCH_SIZE

        if initialization == 'random':
            train_x = torch.randn(BATCH_SIZE, 1, z_dim, device=device, dtype=dtype)

            train_obj = objective_function(logical_plan, unnormalize(train_x, bounds=latent_bounds)).unsqueeze(-1)
            print(f"train_obj shape: {train_obj.shape}")
            best_observed_value = train_obj.max().item()

            # Remove all -1 and -2 (invalid or duplicate plans)
            valid_mask = (train_obj < 0).squeeze(-1)

            train_x_valid = train_x[valid_mask]
            train_x_invalid = train_x[~valid_mask]
            train_obj = train_obj[valid_mask]

            print(f"Valid #: {train_x_valid.shape}")
            print(f"Invalid #: {train_x_invalid.shape}")
            print(f"Valid labels #: {train_obj.shape}")
            print(f"Finished generating {BATCH_SIZE} initial samples")

            print(f"z[0] (latency dim) of valid: {train_x_valid[:, 0].mean():.3f} ± {train_x_valid[:, 0].std():.3f}")
            print(f"z[0] (latency dim) of invalid:   {train_x_invalid[:, 0].mean():.3f} ± {train_x_invalid[:, 0].std():.3f}")


            return train_x_valid, train_x_invalid, train_obj, best_observed_value, latent_bounds
        else:
            """
            Use topk (k=100) generated plans written down in .txt file as
            encoded vectors for initialization data
            """
            data, _, _, mean, std = load_autoencoder_carb_data(device=device, path=initialization, num_platfs=args.platforms, num_ops=args.operators)

            initialization_data = DataLoader(data, batch_size=len(data), drop_last=False, shuffle=False)

            all_mu = []
            train_x_valid = []
            all_latencies = []

            with torch.no_grad():
                for logical, physical, latency in initialization_data:
                    mu, logvar = model.encoder(logical, physical)
                    z = model.reparameterize(mu, logvar)
                    all_mu.append(mu)
                    train_x_valid.append(z)

                    # Unnormalize latency
                    raw_latency = torch.expm1(latency * std + mean)
                    all_latencies.append(raw_latency * latency_estimation_scale)

            all_mu = torch.cat(all_mu, dim=0)  # [n_total, z_dim]
            mu_std = all_mu.std(dim=0)         # [z_dim] — across entire dataset
            train_x_valid = torch.cat(train_x_valid, dim=0).to(device)
            train_obj = -1 * torch.cat(all_latencies, dim=0)   # [n_total] negated for maximization
            train_obj = train_obj.unsqueeze(-1).to(device)

            # Scale bounds by per-dimension std
            scale = mu_std / mu_std.mean()  # relative scale per dimension
            latent_bounds = torch.stack([
                -4.0 * scale,
                4.0 * scale
            ]).to(device=device, dtype=dtype)

            return train_x_valid, torch.empty(0, 1, z_dim).to(device), train_obj, train_obj.max().item(), latent_bounds

    def optimize_acqf_and_get_observation(acq_func, args):
        x_center = state.train_x_valid[state.train_obj.argmax(), :].clone().squeeze()
        """
        x_range = state.train_x.max().item() - state.train_x.min().item()
        x_range = max(x_range, 8.0)
        weights = torch.ones_like(x_center)
        weights = weights * x_range # less than 4 stdevs on either side max
        #tr_lb = x_center - weights # effectively the entire region
        #tr_ub = x_center + weights
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0
        new_bounds = torch.stack([tr_lb, tr_ub])
        """

        if args.acqf == "ei":
            # optimize
            print(f"[{datetime.datetime.now()}] Starting gen candidates")
            candidates, expected = optimize_acqf(
                acq_function=acq_func,
                bounds=norm_bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
            candidates = candidates
        elif args.acqf == "ts":
            print("Using thompson sampling")
            tr_lb = torch.tensor([0.0] * z_dim, device=device, dtype=dtype)
            tr_ub = torch.tensor([1.0] * z_dim, device=device, dtype=dtype)
            sobol = SobolEngine(args.zdim, scramble=True)
            pert = sobol.draw(state.batch_size).to(dtype=dtype).to(device)
            pert = tr_lb + (tr_ub - tr_lb) * pert
            # Create a perturbation mask
            prob_perturb = min(20.0 / args.zdim, 1.0)
            mask = torch.rand(state.batch_size, args.zdim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, args.zdim - 1, size=(len(ind),), device=device)] = 1
            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(state.batch_size, args.zdim).clone()
            X_cand[mask] = pert[mask]
            try:
                with torch.no_grad():
                    candidates = acqf(X_cand, num_samples=state.batch_size)  # shape: [batch_size, z_dim]
            except Exception as e:
                print(f"Failed sampling: {e}")
                candidates = X_cand[0].unsqueeze(0)  # shape: [1, z_dim]
        elif args.acqf == "random":
            print("Using random acqf")
            candidates = torch.randn(state.batch_size, 1, z_dim, device=device, dtype=dtype)

        print(f"[{datetime.datetime.now()}] Finished gen candidates")
        #new_x = unnormalize(candidates.detach().unsqueeze(1), bounds=latent_bounds)
        new_x = candidates
        candidates = unnormalize(candidates.detach().unsqueeze(1), bounds=latent_bounds)

        print(f"[{datetime.datetime.now()}] Starting objective_function on candidates")
        new_obj = objective_function(logical_plan, candidates).unsqueeze(-1)
        print(f"[{datetime.datetime.now()}] Finished objective_function on candidates")

        return new_x, new_obj

    criteria = StoppingCriteria(args.time * 60, args.improvement, args.steps)

    if state is None:
        best_observed = []

        for logical_plan, target in plan:
            train_x_valid, train_x_invalid, train_obj, best_value, latent_bounds = gen_initial_data(
                logical_plan,
                ML_model,
                latent_bounds,
                args.initialization,
            )
            best_observed.append(best_value)
            model_results = []

            print(f"Latent bounds: {latent_bounds}")

        state = State(
            args=args,
            ml_model=ML_model,
            model_results=model_results,
            tree=logical_plan,
            train_x_valid=normalize(train_x_valid.squeeze(1), bounds=norm_bounds),
            train_x_invalid=normalize(train_x_invalid.squeeze(1), bounds=norm_bounds),
            train_obj=train_obj,
            best_values=best_observed,
            batch_size=BATCH_SIZE,
            stopping_criteria=criteria,
        )

    criteria.start_timer()

    while not criteria.is_met():
        # Update the surrogate model
        state.update_surrogate_model()

        print(f"Best f: {state.train_obj.max()}")

        if args.acqf == "ei":
            """
            acqf = qLogExpectedImprovement(
                model=state.model,
                best_f=state.train_obj.max()
            )
            """

            acqf = qLogNoisyExpectedImprovement(
                model=state.model,
                X_baseline=state.train_x_valid,  # all previously observed points
            )
        elif args.acqf == "ts":
            acqf = MaxPosteriorSampling(model=state.model, replacement=False)
        elif args.acqf == "random":
            acqf = "random"


        new_x, new_obj = optimize_acqf_and_get_observation(acqf, args)

        # Remove -1 or -2 latencies (invalid or duplicates)
        valid_mask = (new_obj < 0).squeeze(-1)
        # Get all duplicates to count them
        duplicates_mask = (new_obj == 2).squeeze(-1)
        duplicates_counter += new_obj[duplicates_mask].shape[0]

        new_x_valid = new_x[valid_mask]
        new_x_invalid = new_x[~valid_mask]
        new_obj = new_obj[valid_mask]

        print(f"z[0] (latency dim) of valid: {new_x_valid[:, 0].mean():.3f} ± {new_x_valid[:, 0].std():.3f}")
        print(f"z[0] (latency dim) of invalid:   {new_x_invalid[:, 0].mean():.3f} ± {new_x_invalid[:, 0].std():.3f}")

        #state.update(normalize(new_x.squeeze(1), bounds=norm_domain_bounds), new_obj, VALID_X)
        state.update(new_x_valid.squeeze(1).to(device), new_x_invalid.squeeze(1).to(device), new_obj.to(device))

        index, best_impr = max(enumerate(state.train_obj), key=lambda x: x[1])
        criteria.step(best_impr.item(), new_x.shape[0])

        print(f"Duplicate rate: {duplicates_counter/criteria.steps_taken:.2%}")

    print('Finish Bayesian Optimization for latent space', flush=True)

    index, best_impr = max(enumerate(state.train_obj), key=lambda x: x[1])

    best_plan = state.train_x_valid[index]
    best_latency = best_impr.item() * -1
    print(f"Best latency: {best_latency}")

    criteria.stop_timer()

    return best_plan, best_latency, state


def run_lsbo(input, args, state: State = None):
    global z_dim
    print(f"Starting LSBO from python")

    # set some defaults, highly WIP
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path=args.model_path
    parameters_path=args.parameters
    z_dim = args.zdim

    data, in_dim, out_dim = load_autoencoder_data_from_str(
        device=device,
        data=input,
        num_platfs=args.platforms,
        num_ops=args.operators,
    )

    # find model parameters
    with open(parameters_path) as file:
        parameters = json.load(file)
        lr = parameters.get("lr", 0.001)
        gradient_norm = parameters.get("gradient_norm", 1.0)
        dropout = parameters.get("dropout", 0.1)
        weights = get_weights_of_model_by_path(model_path)


        if args.model == "carbvae":
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

            args.mean = parameters.get("mean", 1.0)
            args.std = parameters.get("std", 1.0)
        else:
            model = BetaCVAE(
                logical_dim=in_dim,
                physical_dim=out_dim,
                hidden_dim=128,
                latent_dim=z_dim,
                num_phys_ops=out_dim,
                beta=parameters.get('beta', 1.0)
            )

        if weights:
            set_weights(weights=weights, model=model, device=device)

        # load model
        model.to(device)
        model.eval()

        dataloader = DataLoader(data, batch_size=1, drop_last=False, shuffle=False)
        best_plan, best_latency, state = latent_space_BO(model, device, dataloader, args, state)

    return best_plan, best_latency, state

def get_plan_latency(args, sampled_plan) -> float:
    global TIMEOUT
    global PLAN_IMPROVEMENT_CACHE
    global EXECUTABLE_PLANS
    global best_plan_data

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

        if plan_out in PLAN_IMPROVEMENT_CACHE:
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

            #return PLAN_IMPROVEMENT_CACHE[plan_out]
            # Return -2, the code for being a duplicate
            return -2

        print(f"[{datetime.datetime.now()}] Sampling new plan")

        #PLAN_CACHE.add(plan_out)

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

            #exec_time = int(TIMEOUT * 100000)
            #Return -1, the code for being invalid
            return -1

        input, picked_plan, exec_time_str = read_from_wayang(sock_file).split(":")
        sock_file.close()
        sock.close()


        exec_time = int(exec_time_str)

        # Wayang gives -1 on non executable plans
        if exec_time > 0:
            print("Add an executable plan")
            EXECUTABLE_PLANS.add(plan_out)

            with open(args.experience, 'a') as exp_file:
                exp_file.write(f"{input}:{picked_plan}:{exec_time}\n")
                print(f"Successfully appended experience to {args.experience}")


            with open(args.stats, 'a') as stats_file:
                stats_file.write(f"{datetime.datetime.now()}: {exec_time}\n")
            print(f"Successfully appended statistics to {args.stats}")
        else:
            return -1

        # Calculate the current set timeout in ms (convert from sec to ms)
        ms_timeout = TIMEOUT * 1000
        if ms_timeout > exec_time:
            TIMEOUT = int(exec_time / 1000)
            print(f"[{datetime.datetime.now()}] Found better plan, updating timeout: {TIMEOUT} sec")
            best_plan_data = input, picked_plan, exec_time_str

        PLAN_IMPROVEMENT_CACHE[plan_out] = exec_time

        try:
            process.stdout.close()
            process.stderr.close()
            process.kill()
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already exited, nothing to kill

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

        exec_time = TIMEOUT

        return exec_time

    except Exception as e:
        # In case the underlying process died
        print(f"Exception: {e}")

        exec_time = -1
        try:
            process.stdout.close()
            process.stderr.close()
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already exited, nothing to kill

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
    best_plan, best_latency, state = run_lsbo([plan], args, state)

    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.kill()

    return best_plan_data, EXECUTABLE_PLANS, state

def read_from_wayang(sock_file):
    udf_length = read_int(sock_file)
    serialized_udf = sock_file.read(udf_length)
    iterator = UTF8Deserializer().load_stream(sock_file)
    next_val = next(iterator, None)
    if next_val is None:
        print("None return from read_from_wayang")
    else:
        return next_val


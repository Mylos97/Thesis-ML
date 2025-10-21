
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

import datetime
import json
import os
import sys
import umap
import matplotlib.pyplot as plt
import torch

from signal import signal
from subprocess import PIPE, Popen, TimeoutExpired
from typing import Callable, NamedTuple
from torch.utils.data import DataLoader

from LSBO.lsbo import read_from_wayang
from Util.communication import dump_stream, open_connection
from helper import get_weights_of_model_by_path, set_weights, load_autoencoder_data, get_relative_path
from OurModels.EncoderDecoder.bvae import BVAE
from OurModels.EncoderDecoder.decoder import TreeDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

class WayangArgs(NamedTuple):
    exec: str      = '/var/www/html/wayang-assembly/target/wayang-0.7.1/bin/wayang-submit'
    memory: str    = '-Xmx8g --illegal-access=permit'
    namespace: str = 'org.apache.wayang.ml.benchmarks.LSBORunner'
    args: str      = 'java,spark,flink,postgres file:///opt/data/'

def make_umap(plans: list, model: BVAE, label_func: Callable[[list, list, TreeDecoder], list], sample_amnt: 1000):
    """
        Creates umap embeddings & labels

        Args: 
            plans: list of input Wayang plans
            model: bvae model
            label_func: function that takes a list of latent vectors and returns a label
            via execution or other means
            sample_amnt: how many samples the model should sample per plan

        Returns: a tuple of embedding & labels
    """

    # Set training flag true so we get variance from BVAE encoder
    model.training = True
    
    # Create multiple latent vectors per plan & generate labels per latent vector
    latent_vectors: list = [model.encoder(plan) for plan in plans for _ in range(sample_amnt)]
    labels: list         = label_func(latent_vectors, [plan[0] for plan in plans], model.decoder)

    # Create umap
    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(latent_vectors)
    
    return embedding, labels

def plot(embedding, labels):
    """
        plots a umap and saves it to file /umap_plot.png

        Args:
            embedding: umap embeddings
            labels: umap labels 

        Returns:
            void
    """
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title("UMAP projection of VAE latent space")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.show()
    plt.savefig("./umap_plot.png", dpi=300)


def latency_to_label(latency: float):
    """
        Converts plan latency to three different classes, invalid, time-out & normal execution
    """
    if latency == 1000000:  # invalid
        return 2
    elif latency == 3000:  # time-out
        return 1
    else:  # default execution
        return 0

def make_validity_labels(latent_vectors: list[torch.Tensor], decoder: TreeDecoder) -> list[int]:
    """
        Decodes & executes latent vectors to get latency, then constructs labels from the score

        Args:
            latent_vectors: list of vectors sampled from the latent space
            decoder: BVAE decoder
    """

    results: list[float] = [execute_plan(decoder(latent_vector, index), WayangArgs()) for latent_vector, index in latent_vectors]
    labels: list[int]    = [latency_to_label(latency) for latency in results]

    return labels

def make_latency_labels(latent_vectors, decoder: TreeDecoder):
    pass

def execute_plan(plan, wayang_args: WayangArgs) -> float:
    """
        Executes a plan with a subprocess in Wayang and returns latency

        Args:
            plan: A plan decoded from a latent vector
            wayang_args: arguments to Wayang executable
    """
    TIMEOUT: int = 100000
    PLAN_CACHE: set = set()
    EXECUTABLE_PLANS: set = set()
    initial_latency: int = 0
    
    try:
        process = Popen([
            wayang_args.exec,
            wayang_args.memory,
            wayang_args.namespace,
            wayang_args.args,
            #str(args.query)
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

        dump_stream(iterator=[plan], stream=sock_file)
        print(f"[{datetime.datetime.now()}] Wrote to wayang")

        sock_file.flush()

        print(plan)
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

        # if plan has beeen executed before we return that plan's inital latency
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

        if process.wait(timeout=TIMEOUT) != 0:
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

        _, _, exec_time_str = read_from_wayang(sock_file).split(":")
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
            # best_plan_data = inp, picked_plan, exec_time_str

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

        exec_time = initial_latency

        return exec_time

    except Exception as e:
        # In case the underlying process died
        print(f"Exception: {e}")
        exec_time = int(TIMEOUT * 100000)

        return exec_time

def fetch_model(model_path: str, parameters_path: str, in_dim: int, out_dim: int) -> BVAE:
    with open(parameters_path) as file:
        parameters = json.load(file)
        dropout    = parameters.get("dropout", 0.1)
        z_dim      = parameters.get("z_dim", 0.1)
        weights    = get_weights_of_model_by_path(model_path)

        model = BVAE(
            in_dim=in_dim,
            out_dim=out_dim,
            dropout_prob=dropout,
            z_dim=z_dim,
        )

        set_weights(weights=weights, model=model, device=device)
        model.to(device)

        return model
    
def fetch_data(path: str) -> tuple[int, int, DataLoader]:
    """
        Fetches plan data from a given path

        Args:
            path to data .txt file

        Returns: 
            a tuple of (in_dim, out_dim, data)
    """
    data, in_dim, out_dim  = load_autoencoder_data(path=path, device=device, num_ops=43, num_platfs=9)
    dataloader: DataLoader = DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=0)

    return (in_dim, out_dim, [inp for inp, _ in dataloader])

# paths
data_path: str       = get_relative_path("train.txt", "Data/splits/imdb/training/")
model_path: str      = get_relative_path("bvae-1.onnx", "Models/imdb/")
parameters_path: str = get_relative_path("BVAE-B-1.json", "HyperparameterLogs/imdb/")

in_dim, out_dim, data = fetch_data(path=data_path)

model = fetch_model(
		model_path=model_path, 
		parameters_path=parameters_path, 
		in_dim=in_dim,
		out_dim=out_dim
	)

model.train(False)

embedding, labels = make_umap(
		model=model, 
		plans=data, 
		sample_amnt=1, 
		label_func=make_validity_labels
	)

plot(embedding=embedding, labels=labels)

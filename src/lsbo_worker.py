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
import socket
import struct
import base64
import re
import sys
import ast
import copy

import torch
import argparse
import torch.onnx
import ast
import json
from exporter import export_model
from OurModels.PairWise.model import Pairwise
from OurModels.CostModel.model import CostModel
from OurModels.EncoderDecoder.model import VAE
from OurModels.EncoderDecoder.bvae import BVAE

from helper import load_autoencoder_data, load_pairwise_data, load_costmodel_data, get_relative_path, get_weights_of_model_by_path, Beta_Vae_Loss, set_weights, load_autoencoder_data_from_str
from hyperparameterBO import do_hyperparameter_BO
from latentspaceBO import latent_space_BO
from torch.utils.data import DataLoader, Dataset


class SpecialLengths(object):
    END_OF_DATA_SECTION = -1
    PYTHON_EXCEPTION_THROWN = -2
    TIMING_DATA = -3
    END_OF_STREAM = -4
    NULL = -5
    START_ARROW_STREAM = -6

def read_int(stream):
    length = stream.read(4)
    if not length:
        raise EOFError
    res = struct.unpack("!i", length)[0]
    return res


class UTF8Deserializer:
    """
    Deserializes streams written by String.getBytes.
    """

    def __init__(self, use_unicode=True):
        self.use_unicode = use_unicode

    def loads(self, stream):
        length = read_int(stream)
        if length == SpecialLengths.END_OF_DATA_SECTION:
            raise EOFError
        elif length == SpecialLengths.NULL:
            return None
        s = stream.read(length)
        return s.decode("utf-8") if self.use_unicode else s

    def load_stream(self, stream):
        try:
            while True:
                yield self.loads(stream)
        except struct.error:
            return
        except EOFError:
            return

    def __repr__(self):
        return "UTF8Deserializer(%s)" % self.use_unicode


def write_int(p, outfile):
    outfile.write(struct.pack("!i", p))


def write_with_length(obj, stream):
    if type(obj) is list:
        arr = np.array(obj)
        serialized = arr.tobytes()
    else:
        serialized = str(obj).encode('utf-8')
    if serialized is None:
        raise ValueError("serialized value should not be None")
    if len(serialized) > (1 << 31):
        raise ValueError("can not serialize object larger than 2G")
    write_int(len(serialized), stream)
    stream.write(serialized)


def dump_stream(iterator, stream):
    if type(iterator) is bool:
        write_with_length(str(int(iterator == True)), stream)
    else:
        for obj in iterator:
            if type(obj) is str:
                write_with_length(obj, stream)
            if type(obj) is bool:
                write_with_length(int(obj == True), stream)
            else:
                write_with_length(obj, stream)
            ## elif type(obj) is list:
            ##    write_with_length(obj, stream)
    write_int(SpecialLengths.END_OF_DATA_SECTION, stream)

def lsbo(input):
    print(f"Starting LSBO from python")

    # set some defaults, highly WIP
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path= f"{dir_path}/Models/vae.onnx"
    parameters_path = f"{dir_path}/HyperparameterLogs/VAE.json"
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
        z_dim = parameters.get("z_dim", 16)
        weights = get_weights_of_model_by_path(model_path)

        #best_model, x = do_hyperparameter_BO(model_class=model_class, data=data, in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device, lr=lr, weights=weights, epochs=epochs, trials=trials, plots=args.plots)

        model = VAE(
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
        json_result = latent_space_BO(model, device, dataloader)

    return [json_result]

def process(infile, outfile):
    udf_length = read_int(infile)
    serialized_udf = infile.read(udf_length)
    iterator = UTF8Deserializer().load_stream(infile)
    output = lsbo(iterator)
    dump_stream(iterator=output, stream=outfile)

def local_connect(port):
    sock = None
    errors = []
    # Support for both IPv4 and IPv6.
    # On most of IPv6-ready systems, IPv6 will take precedence.
    for res in socket.getaddrinfo("127.0.0.1", port, socket.AF_UNSPEC, socket.SOCK_STREAM):
        af, socktype, proto, _, sa = res
        try:
            sock = socket.socket(af, socktype, proto)
            sock.settimeout(30)
            sock.connect(sa)
            sockfile = sock.makefile("rwb", 65536)
            return (sockfile, sock)
        except socket.error as e:
            emsg = str(e)
            errors.append("tried to connect to %s, but an error occurred: %s" % (sa, emsg))
            sock.close()
            sock = None
    raise Exception("could not open socket: %s" % errors)


if __name__ == '__main__':
    java_port = int(os.environ["PYTHON_WORKER_FACTORY_PORT"])
    sock_file, sock = local_connect(java_port)
    process(sock_file, sock_file)
    sock_file.flush()
    exit()

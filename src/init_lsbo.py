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
import argparse
import torch
from subprocess import PIPE, Popen
from torch.utils.data import DataLoader, Dataset

from Util.communication import read_int, SpecialLengths, UTF8Deserializer, write_int, write_with_length, dump_stream, open_connection
from lsbo_worker import lsbo
from latentspaceBO import LSBOResult
from helper import load_autoencoder_data_from_str
from main import main as retrain

def init_model():
    pass

def request_wayang_plan(exec: str, namespace: str, args: str, lsbo_result: LSBOResult = None, index: int = 0) -> LSBOResult:
    process = Popen([
        exec,
        namespace,
        args
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
    lsbo_result = lsbo([plan], lsbo_result)

    print("Sending sampled plan back to Wayang")

    dump_stream(iterator=lsbo_result.model_results, stream=sock_file)

    sock_file.flush()

    print("Sent sampled plan back to Wayang")

    #print(process.stdout.read())
    process.stdout.flush()

    input, picked_plan, exec_time_str = read_from_wayang(sock_file).split(":")

    exec_time = int(exec_time_str)

    print(float(exec_time))

    lsbo_result.train_obj[index][0] = exec_time
    print(f"Train_obj: {lsbo_result.train_obj}")

    if process.wait() != 0:
        print("Error closing Wayang process!")

    plan_data = (input, picked_plan, exec_time_str)

    return lsbo_result, plan_data

def read_from_wayang(sock_file):
    udf_length = read_int(sock_file)
    serialized_udf = sock_file.read(udf_length)
    iterator = UTF8Deserializer().load_stream(sock_file)

    return next(iterator)


def main(args) -> None:
    #model, surrogate_model = init_model()
    lsbo_result = None
    best_plan = None
    for i in range(10):
        lsbo_result, plan_data = request_wayang_plan(args.exec, args.namespace, args.args, lsbo_result, i)

        if best_plan is None:
            best_plan = plan_data
        elif float(best_plan[2]) > float(plan_data[2]):
            print("Found better plan latency")
            best_plan = plan_data

    print(best_plan)

    # add best plan to trainset
    with open(args.trainset, 'a') as training_file:
        training_file.write(f"{best_plan[0]}:{best_plan[1]}:{best_plan[2]}\n")
        print(f"Successfully appended best sampled plan to {args.trainset}")

    args.retrain = args.trainset
    retrain(args)
    #lsbo()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vae')
    parser.add_argument('--exec', type=str, default='/var/www/html/wayang-assembly/target/wayang-0.7.1/bin/wayang-submit')
    parser.add_argument('--namespace', type=str, default='org.apache.wayang.ml.benchmarks.LSBOSampler')
    parser.add_argument('--args', type=str, default='java,spark,flink file:///var/www/html/data')
    parser.add_argument('--trainset', type=str, default='./src/Data/new-encodings.txt')
    parser.add_argument('--model-path', default='./src/Models/vae.onnx')
    parser.add_argument('--parameters', default='./src/HyperparameterLogs/BVAE.json')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--lr', type=str, default='[1e-6, 0.1]')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--plots', type=bool, default=False)
    parser.add_argument('--platforms', type=int, default=9)
    parser.add_argument('--operators', type=int, default=43)
    args = parser.parse_args()

    main(args)

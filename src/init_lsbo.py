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

from LSBO.lsbo import LSBOResult, request_wayang_plan
from main import main as retrain

# Default should be 10 min
TIMEOUT = 60 * 10
TOLERANCE = 1.25
time_limit_reached = False

def main(args) -> None:
    lsbo_result = None
    timeout = float(60 * 10)

    plan_data, initial_latency, plan_cache = request_wayang_plan(args, lsbo_result, timeout)
    print(f"Best plan data: {plan_data}")

    # add best plan to trainset
    with open(args.trainset, 'a') as training_file:
        #training_file.write(f"{plan_data[0]}:{plan_data[1]}:{plan_data[2]}\n")
        training_file.write(f"{plan_data[1]}:{plan_data[0]}:{plan_data[2]}\n")
        print(f"Successfully appended best sampled plan to {args.trainset}")

    with open(args.stats, 'a') as stats_file:
        #training_file.write(f"{plan_data[0]}:{plan_data[1]}:{plan_data[2]}\n")
        stats_file.write(f"{args.query}:{len(plan_cache)}:{initial_latency}:{plan_data[2]}\n")
        print(f"Successfully appended statistics to {args.stats}")

    """
    args.retrain = args.trainset
    retrain(args)
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bvae')
    parser.add_argument('--exec', type=str, default='/var/www/html/wayang-assembly/target/wayang-0.7.1/bin/wayang-submit')
    parser.add_argument('--memory', type=str, default='-Xmx8g --illegal-access=permit')
    parser.add_argument('--namespace', type=str, default='org.apache.wayang.ml.benchmarks.LSBORunner')
    parser.add_argument('--args', type=str, default='java,spark,flink,postgres file:///opt/data/')
    parser.add_argument('--query', type=str, default="1")
    parser.add_argument('--trainset', type=str, default='./src/Data/splits/tpch/bvae/retrain-25.txt')
    parser.add_argument('--stats', type=str, default='./src/Data/splits/tpch/bvae/stats.txt')
    parser.add_argument('--model-path', default='./src/Models/bvae.onnx')
    parser.add_argument('--parameters', default='./src/HyperparameterLogs/BVAE.json')
    parser.add_argument('--zdim', type=int, default=31)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--lr', type=str, default='[1e-6, 0.1]')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--plots', type=bool, default=False)
    parser.add_argument('--platforms', type=int, default=9)
    parser.add_argument('--operators', type=int, default=43)
    # add a time in minutes for this process to run, otherwise stop it
    parser.add_argument('--time', type=int, default=1)
    parser.add_argument('--improvement', type=float, default=25)
    args = parser.parse_args()

    main(args)

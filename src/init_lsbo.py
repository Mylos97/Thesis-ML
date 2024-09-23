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
import threading

from LSBO.lsbo import LSBOResult, request_wayang_plan
from main import main as retrain

# Default should be an hour
TIMEOUT = 60 * 60 * 60
TOLERANCE = 1.25
time_limit_reached = False

def main(args) -> None:
    lsbo_result = None
    best_plan = None
    timeout = float(60 * 60 * 60)
    index = -1

    def set_time_limit_reached():
        global time_limit_reached
        time_limit_reached = True
        print("Time limit reached, stopping LSBO loop")

    """
    t = threading.Timer(args.time * 60, set_time_limit_reached)
    t.start()

    while not time_limit_reached:
    """
    index += 1
    lsbo_result, plan_data = request_wayang_plan(args, lsbo_result, index, timeout)

    if best_plan is None:
        best_plan = plan_data
    elif float(best_plan[2]) > float(plan_data[2]):
        print("Found better plan latency")
        best_plan = plan_data

    timeout = (float(best_plan[2]) * 1.5) / 1000

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
    parser.add_argument('--model', default='bvae')
    parser.add_argument('--exec', type=str, default='/var/www/html/wayang-assembly/target/wayang-0.7.1/bin/wayang-submit')
    parser.add_argument('--namespace', type=str, default='org.apache.wayang.ml.benchmarks.LSBORunner')
    parser.add_argument('--args', type=str, default='java,spark,flink,giraph file:///var/www/html/data')
    parser.add_argument('--trainset', type=str, default='./src/Data/naive-lsbo.txt')
    parser.add_argument('--model-path', default='./src/Models/bvae.onnx')
    parser.add_argument('--parameters', default='./src/HyperparameterLogs/BVAE.json')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--lr', type=str, default='[1e-6, 0.1]')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--plots', type=bool, default=False)
    parser.add_argument('--platforms', type=int, default=9)
    parser.add_argument('--operators', type=int, default=43)
    # add a time in minutes for this process to run, otherwise stop it
    parser.add_argument('--time', type=int, default=1)
    args = parser.parse_args()

    main(args)

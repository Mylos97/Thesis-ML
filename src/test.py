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
import argparse
import datetime

def main():
    timeout = 5

    try:
        print(f"[{datetime.datetime.now()}] Starting sleep for {timeout * 2} seconds")
        process = Popen([
            "sleep",
            str(timeout * 10),
        ],
        stdout=PIPE, stderr=PIPE, start_new_session=True)

        #print(f"[stdout] {process.stdout.readline()}")
        #out, err = process.communicate(timeout=timeout)
        process.wait(timeout=timeout)
    except TimeoutExpired:
        print(f"[{datetime.datetime.now()}] Timeout after {timeout} seconds")
        #os.system("pkill -TERM -P %s"%process.pid)
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.kill()
        process.wait(timeout=1)


        try:
            print(f"[{datetime.datetime.now()}] Starting 2nd sleep for {timeout * 2} seconds")
            process = Popen([
                "sleep",
                str(timeout * 10),
            ],
            stdout=PIPE, stderr=PIPE, start_new_session=True)

            process.stdout.flush()
            #out, err= process.communicate(timeout=timeout)
            print(process.wait(timeout=timeout))
        except TimeoutExpired as e:
            print(f"Exception: {e}")
            print(f"[{datetime.datetime.now()}] Timeout after {timeout} seconds")
            #os.system("pkill -TERM -P %s"%process.pid)
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.kill()
            process.wait(timeout=1)

if __name__ == "__main__":
    main()

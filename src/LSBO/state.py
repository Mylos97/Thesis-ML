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
import torch
import math

class State:

    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False
    valid_values: list = list()

    def __init__(
        self,
        initial_latency,
        ml_model,
        model,
        model_results,
        tree,
        train_x,
        train_obj,
        state_dict,
        best_values
    ):
        self.initial_latency = initial_latency
        self.ml_model = ml_model
        self.model = model
        self.model_results = model_results
        self.tree = tree
        self.train_x = train_x
        self.train_obj = train_obj
        self.state_dict = state_dict
        self.best_values = best_values

    def initialize_tr_state(self):
        self.length = 0.8
        self.failure_counter = 0
        self.success_counter = 0
        self.restart_triggered = False

    def update_tr_length(self):
        # Update the length of the trust region according to
        # success and failure counters
        # (Just as in original TuRBO paper)
        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

        if self.length < self.length_min:  # Restart when trust region becomes too small
            self.restart_triggered = True
            print("state restart_triggered")
            self.initialize_tr_state()

        print("====STATE====")
        print(f"length: {self.length}")
        print(f"success_counter: {self.success_counter}")
        print(f"failure_counter: {self.failure_counter}")
        print(f"# of valid_x: {len(self.valid_values)}")
        print("====STATE====")

    def update_opt_state(self, new_x, new_obj, valid_x):
        """Success and failure counters are updated accoding to
        the objective values (new_obj) of the batch of candidate
        points evaluated on the optimization step.
        """

        #Determine which candidates produced valid plans
        valid_new_x = [t for i, t in enumerate(new_x) if i in valid_x]
        print(f"Valid_x: {valid_x}")
        print(f"Valid new_x: {valid_new_x}")
        if len(valid_new_x) == 0: # no valid candidates found
            # count a failure
            self.success_counter = 0
            self.failure_counter += 1
        else: # at least one valid candidate was found
            # Case 1: improvement was found - count success
            # Case 2: No improvement was found, but a first valid_x was found  - count a success
            max_prev = max(self.best_values)
            improved_obj = max(new_obj) > max_prev
            obtained_validity = len(self.valid_values) == 0 # no previous valid values

            if improved_obj or obtained_validity:
                print(f"new impr: {improved_obj} or new valid: {obtained_validity}")
                # add all valid x's to state
                for x in valid_x:
                    self.valid_values.append(x)
                    print(f"valid_values: {self.valid_values}")

                self.success_counter += 1
                self.failure_counter = 0
            else:
                # count a failure
                self.success_counter = 0
                self.failure_counter += 1

        # Update the length of the trust region
        self.update_tr_length()


    def update(self, new_x, new_obj, state_dict, valid_x):
        # update optimization state
        self.update_opt_state(new_x, new_obj, valid_x)

         # update training points
        self.train_x = torch.cat((self.train_x, new_x))
        self.train_obj = torch.cat((self.train_obj, new_obj))

        # update progress
        best_value = self.train_obj.max().item()
        self.best_values.append(best_value)

        self.state_dict = state_dict


    def update_model(self, model):
        self.model = model

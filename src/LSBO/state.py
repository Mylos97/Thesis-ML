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
from torch.utils.data import DataLoader, TensorDataset
import math
import gpytorch
import datetime
from .sampling.gaussian import GPModel
from gpytorch.mlls import PredictiveLogLikelihood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    batch_size: int = 10
    initial_model_training_done: bool = False
    learning_rate = 0.01

    def __init__(
        self,
        args,
        ml_model,
        model_results,
        tree,
        train_x_valid,
        train_x_invalid,
        train_obj,
        best_values,
        batch_size,
        stopping_criteria
    ):
        self.args = args
        self.ml_model = ml_model
        self.model_results = model_results
        self.tree = tree
        self.train_x_valid = train_x_valid
        self.train_x_invalid = train_x_invalid
        self.train_obj = train_obj
        self.best_values = best_values
        self.batch_size = batch_size
        self.stopping_criteria = stopping_criteria
        self.failure_tolerance = int((stopping_criteria.max_steps / 4) / batch_size)
        self.initialize_surrogate_model()

        print(f"unique plans: {self.train_x_valid.shape[0]}")
        print(f"state.patience: {self.failure_tolerance}")

    def initialize_tr_state(self):
        self.length = 0.8
        self.failure_counter = 0
        self.success_counter = 0
        self.restart_triggered = False

    def initialize_surrogate_model(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        self.model = GPModel(
            inducing_points=self.train_x_valid.float(),
            likelihood=likelihood
        ).to(device)

        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.train_x_valid.size(-2))
        #self.mll = gpytorch.mlls.VariationalELBO(self.model.likelihood, self.model, num_data=self.train_x.size(-2))

    def update_surrogate_model(self):
        # GP model has not been trained - use all data
        if not self.initial_model_training_done:
            train_x = self.train_x_valid
            train_y = self.train_obj
            # TODO: pick reasonable nr of epochs
            n_epochs = 20

        # Otherwise, train only on recently obtained data
        else:
            train_x = self.train_x_valid[-self.batch_size :]
            train_y = self.train_obj[-self.batch_size :].squeeze(-1)

            # TODO: pick reasonable nr of epochs
            n_epochs = 20

        try:
            # Actual training loop of surrogate model
            self.model = self.model.train()

            train_batch_size = min(len(train_y), 128)
            optimizer = torch.optim.Adam([{"params": self.model.parameters(), "lr": self.learning_rate}], lr=self.learning_rate)

            # Dataloading
            train_dataset = TensorDataset(train_x, train_y)
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

            for e in range(n_epochs):
                batch_n = 0
                for data in train_loader:
                    batch_n += 1
                    inputs = data[0]
                    scores = data[1]
                    if len(inputs) == 1:
                        # NOTE: hack to handle case with only one training data point
                        # concatenate to repeat the same point twice
                        # needed bc single piont causes bug in mll w/ censored GP
                        # todo later: fix this bug so hack isn't needed
                        inputs = torch.cat((inputs, inputs))
                        scores = torch.cat((scores, scores))
                    output = model(inputs.to(device))
                    loss = -mll(output, scores.to(device))
                    if loss.isfinite().item():
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    else:
                        pass

            self.model = self.model.eval()
            self.model = self.model.to(device)
            self.initial_model_training_done = True
        except:
                  # Sometimes due to unstable training/ inf loss, we get
            #   errors where model params become nan, in this case we want
            #   to re-init the model on all data
            self.initialize_surrogate_model()
            self.initial_model_training_done = False
            self.learning_rate = self.learning_rate / 2


    def update_tr_length(self):
        # Update the length of the trust region according to
        # success and failure counters
        # (Just as in original TuRBO paper)
        """
        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0
        """

        if self.success_counter > 1:
            self.failure_counter = 0

        if self.failure_counter > 0:
            self.success_counter = 0

            if self.failure_counter == self.failure_tolerance:
                self.stopping_criteria.force_stop()

        if self.length < self.length_min:  # Restart when trust region becomes too small
            self.restart_triggered = True
            print("state restart_triggered")
            self.initialize_tr_state()

        print("====STATE====")
        print(f"length: {self.length}")
        print(f"success_counter: {self.success_counter}")
        print(f"failure_counter: {self.failure_counter}")
        print(f"# of valid_x: {self.train_x_valid.shape[0]}")
        print("====STATE====")

    def update_opt_state(self, new_x_valid, new_x_invalid, new_obj):
        """Success and failure counters are updated accoding to
        the objective values (new_obj) of the batch of candidate
        points evaluated on the optimization step.
        """

        self.train_x_valid = torch.cat([self.train_x_valid, new_x_valid], dim=0)
        self.train_x_invalid = torch.cat([self.train_x_invalid, new_x_invalid], dim=0)

        #Determine which candidates produced valid plans
        if (new_x_valid.shape[0]) == 0: # no valid candidates found
            # count a failure
            self.success_counter = 0
            self.failure_counter += 1
        else: # at least one valid candidate was found
            # Case 1: improvement was found - count success
            # Case 2: No improvement was found, but a first valid_x was found  - count a success
            max_prev = max(self.best_values)
            improved_obj = max(new_obj) > max_prev
            obtained_validity = self.train_x_valid.shape[0] == 0 # no previous valid values

            if improved_obj or obtained_validity:
                print(f"new impr: {improved_obj} or new valid: {obtained_validity}")

                with open(self.args.stats, 'a') as stats_file:
                    stats_file.write(f"{datetime.datetime.now()}: {max(new_obj).item() * -1} at {self.stopping_criteria.steps_taken}\n")

                self.success_counter += 1
                self.failure_counter = 0
            else:
                # count a failure
                self.success_counter = 0
                self.failure_counter += 1

        # Update the length of the trust region
        self.update_tr_length()


    def update(self, new_x_valid, new_x_invalid, new_obj):
        # update optimization state
        self.update_opt_state(new_x_valid, new_x_invalid, new_obj)

         # update training points
        self.train_obj = torch.cat((self.train_obj, new_obj))

        # update progress
        best_value = self.train_obj.max().item()
        self.best_values.append(best_value)


    def update_model(self, model):
        self.model = model

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
import threading
import math

# Set to 30min (1800 seconds)
TIMEOUT = float(60 * 30)

class StoppingCriteria:
    # Time limit as primary stopping criteria
    time_limit: int
    __time_limit_reached: bool = False

    # When a certain % improvement is hit, we can also stop
    improvement_threshhold: int
    __improvement_threshhold_reached: bool = False

    # Iterations left after improvement is hit
    __iterations_left: int = 2

    # Initial latency to be given in the start of the loop
    __initial_latency: int

    # Timer needs to be stopped eventually
    __timer: threading.Timer = None

    # Total number of steps taken in lsbo process
    steps_taken: int = 0

    """
    Maximum number of steps to be taken in general (default 0)
    Setting this to > 0 will make the time_limit useless,
    as we only care about steps then
    """
    max_steps: int = 0

    def __init__(
        self,
        time_limit: int,
        improvement_threshhold: float,
        initial_latency: int,
        max_steps: int,
    ):
        self.time_limit = time_limit
        self.improvement_threshhold = improvement_threshhold
        self.__initial_latency = initial_latency
        self.max_steps = max_steps

    # Global timer that fulfills the criteria after some time
    def start_timer(self):
        if self.max_steps == 0:
            def set_reached():
                self.__time_limit_reached = True
                print("Time limit reached, stopping LSBO loop")

            self.__timer = threading.Timer(self.time_limit, set_reached)
            self.__timer.start()

    def stop_timer(self):
        if self.max_steps == 0:
            if not self.__time_limit_reached:
                self.__timer.cancel()

    def __improvement_rate(self, improvement: float) -> float:
        return (math.pow(math.e, improvement) / self.__initial_latency) * 100

    def step(self, improvement: float, num_steps: int):
        self.steps_taken += num_steps
        print(f"{self.steps_taken}/{self.max_steps} taken")

        if self.__iterations_left <= 0 and self.max_steps == 0:
            print("Canceling timer, no iterations left")
            self.__timer.cancel()
            return

        if self.__improvement_threshhold_reached:
            self.__iterations_left -= 1
            print(f"Improvement threshhold hit, {self.__iterations_left} steps left, {self.steps_taken} steps taken")

            return

        if self.__improvement_rate(improvement) >= self.improvement_threshhold:
            self.__improvement_threshhold_reached = True
            print("Improvement threshhold hit")

        if self.max_steps > 0 and self.steps_taken >= self.max_steps:
            print(f"{self.steps_taken}/{self.max_steps} taken, stopping")


    def is_met(self) -> bool:
        # If the initial plan yielded a valid execution, prioritize improvement threshhold
        if self.__initial_latency <= TIMEOUT:
            return self.__time_limit_reached or (self.__improvement_threshhold_reached and self.__iterations_left <= 0) or (self.max_steps > 0 and self.steps_taken >= self.max_steps)
        else:
            # If otherwise, improvement can't be a stopping criteria, as MAX_VALUE - real_latency will most like fulfill criteria instantly
            return self.__time_limit_reached or (self.max_steps > 0 and self.steps_taken >= self.max_steps)


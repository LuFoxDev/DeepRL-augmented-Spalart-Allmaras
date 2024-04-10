from modules.utilities.json_config import json_config 
from scipy.optimize import *
import time
import numpy as np

class Optimizer():

    def __init__(self, path_to_config):

        self.config = json_config(path_to_config)
        self.bounds = (None, None)  # by default set bounds to None
        self.evaluated_points = None
        self.results = None
        self.function_calls = 0

        print(f"imported optimizer settings: {self.config.optimizer_settings}")

        try:
            self.optimizer = eval(self.config.optimizer_name)
        except Exception:
            print(f"optimizer {self.config.optimizer_name} not recognized")

    def attach_function(self, function_in):
        """
        
        """
        def function(x):
            """This function will be passed to a minimize method"""
            try:
                input_point = x.reshape((1,self.config.number_of_parameters))  # TODO: use dynamic dimension 
                result = function_in(input_point)
                self.function_calls += 1
                print(f"function call {self.function_calls}: {x} : value: {result}")

                # save results
                if self.evaluated_points is None:
                    self.evaluated_points = input_point
                else:
                    self.evaluated_points = np.append(self.evaluated_points, input_point, axis=0)
                
                if self.results is None:
                    self.results = result
                else:
                    self.results = np.append(self.results, result, axis=0)

                np.savetxt("evaluated_points.txt", self.evaluated_points)
                np.savetxt("results.txt", self.results)
            except:
                print(f"function call {self.function_calls}: {x} :FAILED! returning 1.1111 ")
                result = 1.1111
            return result

        self.function = function
    
    def set_bounds(self):
        """
        """
        bounds_list = self.config.number_of_parameters * [self.config.bounds]
        self.bounds = bounds_list

    def run_optimization(self):
        """
        """

        print("starting optimizer")
        _start = time.time()
        self.optimization_results = self.optimizer(func=self.function,
                                    bounds=self.bounds,
                                    **self.config.optimizer_settings)
        _duration = time.time() - _start
        print(f"duration: {_duration:.3f}")
        print(self.optimization_results)

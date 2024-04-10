 
import numpy as np


def Ackley(input):

    shift_x = -0.2
    shift_y = -0.3
    x = (5*input[0]) + shift_x
    y = (5*input[1]) + shift_y
    result = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 * np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20
    result = np.array(result).reshape((1,))
    result += 1e-4*np.random.random(size=(1,))
    return result


def generate_random_two_dim_gaussian_fields(n_fields, size, seed=0):
    """
    """

    print("done")
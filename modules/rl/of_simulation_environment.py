import numpy as np


class Simulation():

    def __init__(self, n_mesh_batches, cells_per_batch, n_parameters):

        self.n_mesh_batches = n_mesh_batches
        self.cells_per_batch = cells_per_batch
        self.n_parameters = n_parameters
        self.target = np.random.normal(0, 0.2, size=(self.cells_per_batch, self.n_mesh_batches))



    def run(self, actions):
        """
        """

        gaussian = np.random.normal(0, 0.1, self.cells_per_batch)
        next_state = actions + gaussian

        return next_state

    def calculate_reward(self):
        """
        """

        rmse = 0.1

        reward = 1. / rmse
    
        return reward

import numpy as np
from tensorforce import Environment

import numpy as np
from tensorforce import Environment
from modules.rl.of_simulation_environment import Simulation


class MARL():

    def __init__(self, n_batches, cells_per_batch, n_parameters):

        self.n_batches = n_batches
        self.cells_per_batch = cells_per_batch
        self.n_parameters= n_parameters
        
        env = CustomEnvironment(n_batches, cells_per_batch, n_parameters, self)
        
        self.environments = [Environment.create(
                            environment=env, max_episode_timesteps=10
                            ) for i in range(n_batches)]
        self.test = 0
        self.target_state = [np.random.normal(0.6, 0.01, size=(self.cells_per_batch, self.n_parameters)) for i in range(self.n_batches)]

    def get_environments(self):

        return self.environments

    def run_simulation(self, batch_actions):

        batch_states, batch_rewards = self._run_simulation(batch_actions)

        return batch_states, batch_rewards

    def _run_simulation(self, batch_actions):

        #print("_run_simulation")
        batch_states = []
        for no_batch in range(self.n_batches):
            #values = (1-np.random.normal(size=(self.cells_per_batch, self.n_parameters))) * np.array([batch_actions[no_batch], batch_actions[no_batch]]).T
            values = (np.array([batch_actions[no_batch]+0.1, batch_actions[no_batch]+0.2])).T
            #values = #(values - (np.min(values))) / (np.max(values)-np.min(values))
            values = (values) / (np.max(values))
            batch_states.append(values)

        batch_rmse =   [np.sum(np.abs(state - target))/state.shape[0] for state, target in zip(batch_states, self.target_state)]
        batch_reward = [1-rmse for rmse in batch_rmse] #[1/rmse for rmse in batch_rmse]

        return batch_states, batch_reward

    


class CustomEnvironment(Environment):

    def __init__(self, n_batches, cells_per_batch, n_parameters, main_simulation):
        self.n_batches = n_batches
        self.cells_per_batch = cells_per_batch
        self.n_parameters = n_parameters
        self.simulation = main_simulation
        super().__init__()

    def states(self):
        return dict(type='float', shape=(self.cells_per_batch, self.n_parameters), min_value=0.0, max_value=1.0)

    def actions(self):
        return dict(type='float', shape=(self.cells_per_batch,), min_value=0.0, max_value=1.0)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.zeros((self.cells_per_batch, self.n_parameters))
        return state

    def execute(self, actions):
        next_state = self.simulation.run(actions)
        terminal = False  # Always False if no "natural" terminal state
        reward = self.simulation.calculate_reward()
        return next_state, terminal, reward

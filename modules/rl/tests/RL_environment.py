import numpy as np
from tensorforce import Environment
from modules.rl.of_simulation_environment import Simulation


class CustomEnvironment(Environment):

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='float', shape=(1,), min_value=0.0, max_value=15.0)

    def actions(self):
        return dict(type='float', shape=(2,), min_value=-1.0, max_value=1.0)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        #state = Ackley((.5,.5))
        return state

    def execute(self, actions):
        #next_state = Ackley(actions)
        terminal = False  # Always False if no "natural" terminal state
        reward = np.array(1.0 / (next_state + 1e-6)).reshape((1,))
        return next_state, terminal, reward
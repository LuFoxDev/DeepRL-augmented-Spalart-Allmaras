from typing import List
import numpy as np
from modules.config import Config
import os
import matplotlib.pyplot as plt



class DemoRLController():

    def __init__(self, ref_error : float, action_shape : tuple, config : Config):

        self.ref_error = ref_error
        self.action_shape = action_shape
        self.config = config

        self.path_to_postprocessing_folder = os.path.join(
            self.config["export_folder"], self.config["name_of_run"], "post-processing", "")

    def get_demo_actions(self, states : List, rewards : List) -> List:

        print("now computing actions")

        x = np.linspace(-1, 1, self.action_shape[0])
        y = np.linspace(-1, 1, self.action_shape[1])
        xx_batch, yy_batch = np.meshgrid(x, y)
        z = [1e-2*(1-reward)*(2-(xx_batch**2 + yy_batch**2))/2 for reward in rewards]
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(z[0], cmap="RdBu_r", origin="lower", interpolation='none')
        fig.savefig(self.path_to_postprocessing_folder + "/single_action.png", bbox_inches="tight", dpi=200)
        actions = len(states) * z
        
        return actions

    def compute_reward(self, cell_batch_inputs : List):
        print("now computing rewards")
        errors_per_batch = [cell_batch_inputs[batch_id][2][:,:,2] for batch_id in range(len(cell_batch_inputs))]
        rewards = [max(min((self.ref_error-np.mean(e))/self.ref_error, 1), 0) for e in errors_per_batch]
        return rewards

    def compute_states(self, cell_batch_inputs):
        print("now computing states")
        # the state should be compiled of the y center of the batch and the Ux and Uy
        states = []
        for batch in cell_batch_inputs:
            y_center = (np.max(batch[1])-np.min(batch[1]))/2
            Ux = batch[2][:,:,0].flatten()
            Uy = batch[2][:,:,1].flatten()
            state = np.hstack((y_center, Ux, Uy))
            states.append(state)

        return states



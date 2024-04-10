from distutils.command import config
from genericpath import exists
from typing import List
import numpy as np
from modules.config import Config
import os
import pickle
import glob
import matplotlib.pyplot as plt
from tensorforce import Agent
from logging import Logger
import shutil
import time
from modules.plotting.plotting_functions import plot_scalar_field_on_mesh, plot_stencil_vectors
from modules.utilities.units import get_unit




class RLController():

    def __init__(self, state_scaling_margin : list, action_space : tuple, state_space : tuple, max_episode_timesteps : int, config : Config, path_to_tensorforce_config : str, logger : Logger, ref_error=None, path_to_saved_agent : str = None):

        self.state_scaling_margin = state_scaling_margin
        self.ref_error = ref_error
        self.path_to_saved_agent = path_to_saved_agent
        self.logger = logger
        self.config = config
        self.path_to_postprocessing_folder = os.path.join(
            self.config["export_folder"], self.config["name_of_run"], "post-processing", "")

        if self.path_to_saved_agent:
            # if agent shall be imported the action and state spaced
            self.load_agent(self.path_to_saved_agent)


        else:
            self.action_space = action_space
            self.state_space = state_space
            self.path_to_tensorforce_config = path_to_tensorforce_config
            self.max_episode_timesteps = max_episode_timesteps
            # init agent
            self.agent = Agent.create(agent=self.path_to_tensorforce_config,
                                    states=self.state_space,
                                    actions=self.action_space,
                                    max_episode_timesteps=self.max_episode_timesteps,
                                    summarizer=dict(directory='summaries', summaries='all', filename=self.config["name_of_run"], max_summaries=100)) # summarizer=dict(directory='summaries', summaries='all')
            self.logger.info(self.agent.get_architecture())
            # copy used settings to postprocessing folder 
            _target = os.path.join(self.path_to_postprocessing_folder, "used_tensorforce_config.json")
            shutil.copyfile(self.path_to_tensorforce_config, _target)

        self.internals = None
        self.min_rewards =  []
        self.mean_rewards =  []
        self.max_rewards =  []
        self.local_errors = []
        self.global_error_normalized = []
        self.min_actions =  []
        self.mean_actions =  []
        self.max_actions =  []
        self.min_states =  []
        self.mean_states =  []
        self.max_states =  []
        self.agent_updates = 0
        self.experiences = 0
        self.state_scaling = None
        self.global_ref_error = None
        self.all_actions_as_array = None

        self.all_selected_states = []
        self.all_selected_actions = []
        self.all_selected_episode_terminals = []
        self.all_selected_rewards = []

        self.get_unit = get_unit

    def load_agent(self, path : str):
        """
        import agent from file
        """
        self.logger.info(f"now loading agent from file: {path}")
        self.agent = Agent.load(path)
        self.logger.info(f"finished loading agent from file")

    def test_agent(self):
        """
        This function tests the RL pipeline
        """

        self.logger.info("now testing agent")
        test_counter = 100
        try:
            test_states = np.random.random(size=self.state_space["shape"]) 
            test_internals = [self.agent.initial_internals() for i in range(len(test_states))]
            #acts = [self.agent.act(states=state, internals=self.internals, independent=True) for state in states]  # old statement - was very slow
            acts = self.agent.act(states=test_states, independent=True, internals=test_internals) 
            selected_states = [np.random.random(size=self.state_space["shape"])*self.state_space["max_value"] for i in range(test_counter)]
            selected_actions = [np.random.random(size=self.action_space["shape"])*self.action_space["max_value"] for i in range(test_counter)]
            selected_episode_terminals = test_counter * [0]
            selected_episode_terminals[-1] = 2 
            selected_rewards = test_counter * [-1]

            self.agent.experience(
                    states=selected_states, actions=selected_actions,
                    terminal=selected_episode_terminals, reward=selected_rewards
            )

            self.agent.update()
            self.logger.info("agent passed tests without errors")
        except:
            self.logger.error("testing agent failed")





    def feed_recorded_experience_to_agent(self, states : list, last_actions : list, rewards : list, indexes_to_use_for_experience = None):
        """
        """
        self.logger.info("now feeding recorded experience to agent")

        experience_replay = False
        indexes_to_use_for_experience = indexes_to_use_for_experience[0]  # the data input is (array([0,1,...]),)
        #episode_internals = self.internals
        episode_terminal = len(states) * [0]
        #episode_terminal[-1] = 1

        reverse_update = False
        loop_experience = False
        set_all_states_to_terminal = True
        instant_update = False

        # if reverse_update:
        #     states = states[::-1]
        #     last_actions = last_actions[::-1]
        #     episode_terminal = episode_terminal[::-1]
        #     rewards = rewards[::-1]
        #     indexes_to_use_for_experience = indexes_to_use_for_experience[::-1]

        # Eigentlich will ich an dieser Stelle nur einen reward und ein terminal übermitteln, weil die 77800 states ja einer episode entspricht und ich nur ein reward für eine episode brauceh
        # Ich sollte mal checken, wie das im act_exp-Demo interface aussieht mit den shapes, weil das hier einen fehler wirft
        #feed_resolution = 10
        #experience_batches_count = 100
        #experience_batches =  np.array_split(np.arange(len(states), step=feed_resolution), experience_batches_count)
        if indexes_to_use_for_experience is None:
            selected_states = states
            selected_actions = last_actions
            selected_episode_terminals = episode_terminal
            selected_episode_terminals[-1] = 2
            selected_rewards = rewards
        else:
            selected_states = [states[i] for i in indexes_to_use_for_experience if i < len(states)]
            selected_actions = [last_actions[i] for i in indexes_to_use_for_experience if i < len(states)]
            selected_episode_terminals = [episode_terminal[i] for i in indexes_to_use_for_experience if i < len(states)]
            selected_episode_terminals[-1] = 2
            selected_rewards = [rewards[i] for i in indexes_to_use_for_experience if i < len(states)]
        
        self.all_selected_states.extend(selected_states)
        self.all_selected_actions.extend(selected_actions)
        self.all_selected_episode_terminals.extend(selected_episode_terminals)
        self.all_selected_rewards.extend(selected_rewards)

        if experience_replay:
            self.logger.info(f"now feeding {len(self.all_selected_states)} experiences to the agent")
            _start = time.time()
            self.experiences += 1 
            self.agent.experience(
                    states=self.all_selected_states, actions=self.all_selected_actions,
                    terminal=self.all_selected_episode_terminals, reward=self.all_selected_rewards
            )
            _end = time.time()
            self.logger.info(f"TIMER: agent experience took {(_end-_start):.1f} sec")

        else:
            self.logger.info(f"now performing agent.experience()")
            _start = time.time()
            if set_all_states_to_terminal:
                selected_episode_terminals = [1] * len(selected_episode_terminals)
            counter = -1 if reverse_update else 1
            if loop_experience:
                self.logger.info(f"LOOPING EXPERIENCES")
                for selected_state, selected_action, selected_episode_terminal, selected_reward in zip(selected_states[::counter], selected_actions[::counter], selected_episode_terminals[::counter], selected_rewards[::counter]):
                    self.agent.experience(
                        states=[selected_state], actions=[selected_action],
                        terminal=[selected_episode_terminal], reward=[selected_reward])
                    if instant_update:
                        self.agent.update()
                
            else:
                self.agent.experience(
                        states=selected_states[::counter], actions=selected_actions[::counter],
                        terminal=selected_episode_terminals[::counter], reward=selected_rewards[::counter]
                )

            _end = time.time()
            self.logger.info(f"TIMER: agent experience took {(_end-_start):.1f} sec")
            # _start = time.time()
            # for selected_state, selected_action, selected_reward in zip(selected_states, selected_actions, selected_rewards):
            #     self.agent.experience(
            #         states=[selected_state], actions=[selected_action],
            #         terminal=[1], reward=[selected_reward]
            #     )
            # _end = time.time()
            # self.logger.info(f"TIMER: agent experience took {(_end-_start):.1f} sec")
            

        # save experience
        objects = [self.all_selected_states, self.all_selected_actions, self.all_selected_episode_terminals, self.all_selected_rewards]
        names = ["states", "actions", "epsiode_terminals", "rewards"]
        self.experience_folder = os.path.join(self.config["export_folder"],self.config["name_of_run"], f"experience_{self.config['name_of_run']}")
        os.makedirs(self.experience_folder, exist_ok=True)
        for obejct, name in zip(objects, names):
            try:
                file_path = os.path.join(self.experience_folder, f'{name}.pickle')
                pickle_file = open(file_path, 'wb')
                pickle.dump(obejct, pickle_file)
                pickle_file.close()
                self.logger.info(f"saved {name}")
            except Exception as e:
                self.logger.error(f"ERROR: failed to save {file_path}")
                self.logger.error(e)




        # for batch_coutner, experience_batch in enumerate(experience_batches):
        #     self.logger.info(f"feeding experiences to agent: {(batch_coutner+1)*experience_batch.shape[0]} from {experience_batch.shape[0]*experience_batches_count}")
        #     states_batch = [states[i] for i in experience_batch]
        #     last_actions_batch = [last_actions[i] for i in experience_batch]
        #     episode_terminal_batch = [episode_terminal[i] for i in experience_batch]
        #     rewards_batch = [rewards[i] for i in experience_batch]
        #     self.agent.experience(
        #             states=states_batch, actions=last_actions_batch,
        #             terminal=episode_terminal_batch, reward=rewards_batch
        #     )
        # _end = time.time()
        # self.logger.info(f"TIMER: agent experience took {(_end-_start)/60:.1f} minutes")
        # for state, last_action, reward in zip(states, last_actions, rewards):
        #     self.agent.experience(
        #         states=state, actions=dict(last_action),
        #         terminal=episode_terminal, reward=reward
        #     )

    def import_experiences(self):     
        # load experience
        folder_to_import_experiences = self.config["import_experiences"]
        if isinstance(folder_to_import_experiences, str):
            self.logger.info("now importing experiences from path")
            names = ["states", "actions", "epsiode_terminals", "rewards"]
            self.imported_experiences = {}
            _experience_folders = glob.glob(folder_to_import_experiences + "*")
            for _experience_folder in _experience_folders:
                self.logger.info(f"importing experience from {_experience_folder}")
                for name in names:
                    try:
                        file_path = os.path.join(_experience_folder, f'{name}.pickle')
                        pickle_file = open(file_path, 'rb')
                        self.imported_experiences[name] = pickle.load(pickle_file)
                        pickle_file.close()
                    except Exception as e:
                        self.logger.error(f"ERROR: failed to load {name}: {file_path}")
                        self.logger.error(e)

                self.logger.info(f"now feeding {len(self.imported_experiences)} imported experiences to the agent")
                _start = time.time()
                self.agent.experience(
                            states=self.imported_experiences["states"], actions=self.imported_experiences["actions"],
                            terminal=self.imported_experiences["epsiode_terminals"], reward=self.imported_experiences["rewards"]
                    )
                _end = time.time()
                self.logger.info(f"TIMER: agent experience took {(_end-_start):.1f} sec")

                self.logger.info("now updating agent")
                _start = time.time()
                self.agent.update()
                self.agent_updates += 1
                _end = time.time()
                self.logger.info(f"TIMER: updating agent took: {_end - _start:.2f} s")
        else:
            self.logger.info("importing experience from path was deactivated")


    def update_agent(self): 
        """
        """
        self.logger.info("now updating agent")
        _start = time.time()
        self.agent.update()
        self.agent_updates += 1
        #self.logger.info("finished updating agent, now saving agent status")
        # save agent
        _end = time.time()
        self.logger.info(f"TIMER: updating agent took: {_end - _start:.2f} s")


    def get_actions(self, states):
        """
        """
        self.logger.info("now computing actions")   
        _timer = time.time()
        if self.internals is None:
            self.internals = [self.agent.initial_internals() for i in range(len(states))]
        #self.internals = [self.agent.initial_internals() for i in range(len(states))]

        #acts = [self.agent.act(states=state, internals=self.internals, independent=True) for state in states]  # old statement - was very slow
        acts, self.internals = self.agent.act(states=states, independent=True, internals=self.internals)  # REVERSE
        
        # HIER GEHTS WEITER MIT DEM TESTING
        def compare_actions(acts1, acts2):
            reversed_actions_are_equal = True
            for acts1_, acts2_ in zip(acts1, acts2):
                if not (acts1_ == acts2_):
                    reversed_actions_are_equal = False
            if not reversed_actions_are_equal:
                print("ACTIONS NOT EQUAL")
            else:
                print("actions equal")
        # test action reversal
        acts_listcomp = [self.agent.act(states=state, independent=True) for state in states][0]
        acts_reversed = self.agent.act(states=states[::-1] , independent=True, internals=self.internals[::-1] )[0][::-1]  # REVERSE
        compare_actions(acts, acts_listcomp)
        compare_actions(acts, acts_reversed)
        # results: 
        # - performing actions on list of states produces same results as executing as list comprehension
        # - reversing produces same results 


        #actions = [acts[i][0] for i in range(len(acts))]
        actions = acts # [::-1] # REVERSE
        #self.internals = [acts[i][1] for i in range(len(acts))]
        #self.internals = acts[1]

        self.min_actions.append(np.min(actions))
        self.mean_actions.append(np.mean(actions))
        self.max_actions.append(np.max(actions))

        actions_as_array = np.array(actions)
        if self.all_actions_as_array is not None:
            self.all_actions_as_array = np.hstack((self.all_actions_as_array, actions_as_array))
        else:
            self.all_actions_as_array = np.hstack((actions_as_array, actions_as_array))

        plt.close()
        fig = plt.figure(figsize=(12,6))
        plt.imshow(self.all_actions_as_array, aspect="auto")
        plt.xlabel("episode")
        plt.ylabel("action values per cell")
        plt.title("actions")
        plt.colorbar()
        plt.savefig(self.path_to_postprocessing_folder + "/actions_image.png", bbox_inches="tight", dpi=200)

        fig = plt.figure()
        plt.plot(self.min_actions, label="min")
        plt.plot(self.mean_actions, label="mean")
        plt.plot(self.max_actions, label="max")
        plt.xlabel("episode")
        plt.ylabel("action value")
        plt.title("actions")
        plt.legend()
        #plt.yscale("log")
        plt.savefig(self.path_to_postprocessing_folder + "/actions_statistics.png", bbox_inches="tight", dpi=200)

        self.logger.info(f"TIMER: computing and plotting actions took:  {(time.time()-_timer):.1f} seconds")

        return actions

    # def observer_environment(self, reward):


    def get_demo_actions(self, states : List, rewards : List) -> List:

        self.logger.info("now computing actions")

        x = np.linspace(-1, 1, self.action_space[0])
        y = np.linspace(-1, 1, self.action_space[1])
        xx_batch, yy_batch = np.meshgrid(x, y)
        z = [1e-2*(1-reward)*(2-(xx_batch**2 + yy_batch**2))/2 for reward in rewards]
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(z[0], cmap="RdBu_r", origin="lower", interpolation='none')
        fig.savefig(self.path_to_postprocessing_folder + "/single_action.png", bbox_inches="tight", dpi=200)
        actions = len(states) * z
        
        return actions
    
    def _extract_reference_error(self, errors_per_batch : list):

        self.ref_error = np.max([np.mean(e) for e in errors_per_batch])
        self.logger.info(f"determined reference error from data: {self.ref_error:.3f}")

    def compute_reward(self, cell_batch_inputs : List, global_error : float):
        """
        compute reward
        """
        self.logger.info("now computing rewards")
        _start = time.time()

        if global_error is None:
            errors_per_batch = [cell_batch_inputs[batch_id][2][:,:,2] for batch_id in range(len(cell_batch_inputs))]
            if self.ref_error is None:
                # extract reference error
                self._extract_reference_error(errors_per_batch)
        
        if self.global_ref_error is None:
            # set current global error as global ref error
            # this should occur at the first call of this method
            self.global_ref_error = global_error

        global_error_normalized = global_error / self.global_ref_error # next try: (global_error / self.global_ref_error)**4
        #local_errors = [np.mean(e) for e in errors_per_batch] / self.ref_error
        #rewards = (-local_errors -global_error_normalized + 3)/3 -1
        # theoretical max error case: (-inf -inf +3)/3 -1 = -inf
        # largest plauible error case: (-3 -3 +3)/3 -1 = -2
        # theoretical best error: (-0 -0 +3)/3 -1 = 0
        
        # VERY SIMPLIEFIED RMSE ONLY REWARD
        rewards = len(cell_batch_inputs) * [-global_error_normalized]
        #rewards = len(cell_batch_inputs) * [-global_error_normalized+6]
        
        # NEW VERSION SCALED between -1000 and zero
        #global_error_scaled = 2000* (-global_error_normalized) + 1000
        #rewards = len(cell_batch_inputs) * [global_error_scaled]

        #self.local_errors.append(local_errors)
        #self.global_error_normalized.append(global_error_normalized)

        self.min_rewards.append(np.min(rewards))
        self.mean_rewards.append(np.mean(rewards))
        self.max_rewards.append(np.max(rewards))

        self.logger.info(f"computed rewards: min: {np.min(rewards):.3f}, mean: {np.mean(rewards):.3f}, max: {np.max(rewards):.3f}")

        # check if its the last reward is the highest (based on mean)
        if self.mean_rewards[-1] == np.max(self.mean_rewards):
            overwrite_best_agent = True
        else:
            overwrite_best_agent = False

        self.save_agent(overwrite_best_agent=overwrite_best_agent)

        fig = plt.figure()
        plt.plot(self.min_rewards, label="min")
        plt.plot(self.mean_rewards, label="mean")
        plt.plot(self.max_rewards, label="max")
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.title("rewards")
        plt.legend()
        #plt.yscale("log")
        plt.savefig(self.path_to_postprocessing_folder + "/rewards.png", bbox_inches="tight", dpi=200)

        # TAKES AGES TO PLOT
        # fig = plt.figure()
        # local_errors_as_array = np.array(self.local_errors).T
        # for i in range(local_errors_as_array.shape[0]):
        #     if i==0:
        #         lbl = "local error"
        #     else:
        #         lbl = '_nolegend_'

        #     plt.plot(local_errors_as_array[i], color="grey", alpha=0.2, label=lbl)
        # plt.plot(self.global_error_normalized, color="red", label = "normalized global error")
        # plt.xlabel("episode")
        # plt.ylabel("error")
        # plt.title("local errors and normalized global error")
        # plt.legend()
        # #plt.yscale("log")
        # plt.savefig(self.path_to_postprocessing_folder + "/errors.png", bbox_inches="tight", dpi=200)

        _end = time.time()
        self.logger.info(f"TIMER: computing rewards took {_end-_start:.2f} s")

        
        return rewards
    
    def _save_agent_to_target_folder(self, folder_name):
        """
        """
        agent_file_name = os.path.join(self.path_to_postprocessing_folder, folder_name)
        try:
            if os.path.exists(agent_file_name):
                shutil.rmtree(agent_file_name)  # delete folder because otherwise every agent would be saved (77 MB per agent)
            self.agent.save(agent_file_name)
        except Exception as e:
            self.logger.error(f"saving agent {self.agent_updates} failed! Error message: {e}")

    def save_agent(self, overwrite_best_agent=False):
        """
        save latest agent and the one with the highest reward
        """

        if overwrite_best_agent:
            self.logger.info("now saving new best agent")
            self._save_agent_to_target_folder("best_agent")
        else:
            self.logger.info("now saving latest agent")
            self._save_agent_to_target_folder("latest_agent")


    def _extract_state_scaling(self, cell_batch_inputs):
        # get values
        y_centers = [np.min(batch[1])+(np.max(batch[1])-np.min(batch[1]))/2 for batch in cell_batch_inputs]
        Ux_values = [batch[2][:,:,0] for batch in cell_batch_inputs]
        Uy_values = [batch[2][:,:,1] for batch in cell_batch_inputs]
        nut_values = [batch[2][:,:,2] for batch in cell_batch_inputs]
        # get max
        max_y = np.max(y_centers)*self.state_scaling_margin[0]
        max_Ux = np.max(np.abs(Ux_values))
        max_Uy = np.max(np.abs(Uy_values))
        max_velocity = max(max_Ux, max_Uy)*self.state_scaling_margin[1]
        max_nut = np.max(np.abs(nut_values))*self.state_scaling_margin[2]
        # define state scaling
        self.state_scaling = [1/max_y, 1/max_velocity, 1/max_velocity, 1/max_nut]

        self.logger.info(f"extracted state scaling: {self.state_scaling}")






    def compute_states(self, cell_batch_inputs, use_nut=True, field_data=None, mesh_coordinates=None, current_time=None):
        self.logger.info("now computing states")
        _start = time.time()

        # the state should be compiled of the y center of the batch and the Ux and Uy
        if self.state_scaling is None:
            self._extract_state_scaling(cell_batch_inputs)

        states = []
        for batch in cell_batch_inputs:
            y_center = (np.max(batch[1])-np.min(batch[1]))/2 * self.state_scaling[0]
            Ux = batch[2][:,:,0].flatten() * self.state_scaling[1]
            Uy = batch[2][:,:,1].flatten() * self.state_scaling[2]
            nut = batch[2][:,:,2].flatten() * self.state_scaling[3]
            # IMPORTANT: if normalization is changed here, changes have to be applied to the entire field too 
            if use_nut:
                state = np.hstack((y_center, Ux, Uy, nut))
            else:
                state = np.hstack((y_center, Ux, Uy))
            states.append(state)
        
        # apply normalization to entire field data
        if field_data:
            y_center_on_mesh = mesh_coordinates[:,1] * self.state_scaling[0]
            Ux_on_mesh = field_data["U"][:,0] * self.state_scaling[1]
            Uy_on_mesh = field_data["U"][:,1] * self.state_scaling[2]

            fields = [y_center_on_mesh, Ux_on_mesh, Uy_on_mesh]
            names = ["normalized radial position", "normalized Ux", "normalized Uy"]

            for field, name in zip(fields, names):
                plot_scalar_field_on_mesh(mesh_coordinates[:,0], mesh_coordinates[:,1],
                        field,
                        f"{name} on mesh at t = {current_time}",
                        "x", "y", "-", f'{name.replace(" ", "_")}_at_t_{current_time}',
                        self.path_to_postprocessing_folder,
                        close_up_limits=None, cmap="jet", center_cbar=False, aspect_ratio=1)

            if use_nut:
                nut_on_mesh = field_data["nut"] * self.state_scaling[3]
                plot_scalar_field_on_mesh(mesh_coordinates[:,0], mesh_coordinates[:,1],
                            nut_on_mesh,
                            f"normalized nut on mesh at t = {current_time}",
                            "x", "y", self.get_unit("nut"), f"normalized_nut_at_t_{current_time}",
                            self.path_to_postprocessing_folder,
                            close_up_limits=None, cmap="jet", center_cbar=False, aspect_ratio=1)


        if np.max(states) > 1.0:
            self.logger.warning(f"states maximum exceeds 1.0: max(states)= {np.max(states)}")
            states = np.clip(states, -1, 1)

        if np.min(states) < -1.0:
            self.logger.warning(f"states minimum exceeds -1.0: min(states)= {np.min(states)}")
            states = np.clip(states, -1, 1)


        self.min_states.append(np.min(states))
        self.mean_states.append(np.mean(states))
        self.max_states.append(np.max(states))

        self.logger.info(f"computed states: min: {np.min(states):.3f}, mean: {np.mean(states):.3f}, max: {np.max(states):.3f}")

        ###########################################################
        states_as_array = np.array(states)
        with open(self.path_to_postprocessing_folder + f"/{self.config['name_of_run']}_states_as_array.npy", 'wb') as f:
            np.save(f, states_as_array, allow_pickle=True)

        #states_as_array2  = np.load(self.path_to_postprocessing_folder + "/states_as_array.npy", allow_pickle=True)
        plot_stencil_vectors(states_as_array, self.path_to_postprocessing_folder + f"/{self.config['name_of_run']}_states_matrix.png")

        fig = plt.figure()
        plt.plot(self.min_states, label="min")
        plt.plot(self.mean_states, label="mean")
        plt.plot(self.max_states, label="max")
        plt.xlabel("episode")
        plt.ylabel("states")
        plt.title("states")
        plt.legend()
        #plt.yscale("log")
        plt.savefig(self.path_to_postprocessing_folder + "/states.png", bbox_inches="tight", dpi=200)
       
        _end = time.time()
        self.logger.info(f"TIMER: computing states took {_end-_start:.2f} s")

        return states



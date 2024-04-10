from modules.LES_results.LES_results_worker import LESresultsWorker
from modules.openfoam.OpenFoamCase import OpenFoamCase
from modules.openfoamCore.openfoamCore import OpenFoamCore
from modules.utilities.logging import init_logger, close_logger
from modules.utilities.mailing import send_attachment_via_email
from modules.rl.rl_controller import RLController
from modules.config import Config
import numpy as np
import os
import pickle
import time
import json
import argparse


_timer = time.time()
#############################################################
# SETTINGS
path_to_config = "config/run_pretrained_agent.conf"
#############################################################

# init logging and import config
logger = init_logger("openfoam_case")
config = Config()
config.load(path_to_config)

# get settings
path_to_agent = config["path_to_agent"] 
max_episodes = config["max_episodes"] 
initial_run_episodes = config["initial_run_episodes"] 
writeInterval = config["writeinterval"] 
points_per_batch_side = config["points_per_batch_side"] 
max_episode_timesteps_for_agent = config["max_episode_timesteps_for_agent"]
use_nut = config["use_nut"] 
# tensorforce_config_dict = config["rl_settings"]
epochs_per_episode = config["epochs_per_episode"] 


#action_min = float(config["action_min"])
#action_max = float(config["action_max"])
cell_batch_rectangle_length = float(config["cell_batch_rectangle_length"])

# define epochs per episode ramping
epochs_per_episode_list = [30000, 60000, 60000, 60000]

# init Open Foam Case
open_foam_case = OpenFoamCase(logger, config)
open_foam_case.setup_new_run(delete_old_files=True)

# init Open Foam Core
open_foam_core = OpenFoamCore(logger, config)

# compile turbulence model
open_foam_core.compile_turbulence_model(force_compilation=False)

# # init Reinforcment Learnign
# tensorforce_config = os.path.join(config["export_folder"], config["name_of_run"], "ppo_temp.json") 
# with open(tensorforce_config, 'w') as fp:
#     json.dump(tensorforce_config_dict, fp)

# init Reinforcement Learnign
rl_controller = RLController(state_scaling_margin = [1, 1.2, 1.2],
                            action_space = None,
                            state_space = None,
                            max_episode_timesteps = None,
                            config=config,
                            path_to_tensorforce_config=None,
                            path_to_saved_agent=path_to_agent,
                            logger=logger)
                            
# rl_controller.import_experiences()

# rl_controller.test_agent()

# import LES results
les = LESresultsWorker(logger, config)
les.load_data()
les.perform_azimutal_averaging(open_foam_case.BlockMesh_coordinates_2D)
open_foam_case.init_target_mean_field(les.mean_field)
logger.info(f"TIMER: setup took {time.time() - _timer:.2f} s")
_timer = time.time()

endTime = 0

#initial run
endTime += initial_run_episodes
open_foam_case.decompose()
open_foam_case.run_solver(endTime=endTime, writeInterval=int(initial_run_episodes/2))
logger.info(f"TIMER: initial run took {time.time() - _timer:.2f} s")
open_foam_case.reconstruct()
open_foam_case.import_results()
open_foam_case.import_log()
open_foam_case.calculate_error()
open_foam_case.plot_error(enforce_plotting=True)
open_foam_case.plot_results(enforce_plotting=True)
open_foam_case.init_neighborhood_batches_OLD_VERSION(rectangle_length=cell_batch_rectangle_length, points_per_batch_side=points_per_batch_side)
# get currents states
initial_states = rl_controller.compute_states(open_foam_case.cell_batches, use_nut, 
                                    field_data=open_foam_case.results,
                                    mesh_coordinates=open_foam_case.BlockMesh_coordinates_2D,
                                    current_time=endTime)

for epoch_id in range(max_episodes):
    episode_start_time = time.time()

    # get all actions
    action_batches = rl_controller.get_actions(initial_states)
    open_foam_case.actions_on_grid = open_foam_case.process_actions(action_batches)

    # execute environment to get reward
    open_foam_case.copy_results_folder(source_time=1, target_time=endTime+1)
    open_foam_case.set_nutSource(time_folder=endTime+1, data=open_foam_case.actions_on_grid)
    open_foam_case.decompose()
    endTime += epochs_per_episode
    open_foam_case.run_solver(endTime=endTime, writeInterval=writeInterval)
    open_foam_case.reconstruct()
    open_foam_case.import_results()
    open_foam_case.import_log()
    open_foam_case.calculate_error()
    open_foam_case.plot_error(enforce_plotting=True)
    open_foam_case.plot_results(enforce_plotting=True)
    logger.info(f"TIMER: run post processing took {(time.time()-episode_start_time)/60:.2f} min")
    # I just neeed to perform this once
    #open_foam_case.init_neighborhood_batches_OLD_VERSION(rectangle_length=cell_batch_rectangle_length, points_per_batch_side=points_per_batch_side)
    rewards = rl_controller.compute_reward(cell_batch_inputs=open_foam_case.cell_batches, global_error=open_foam_case.rmse)

    episode_end_time = time.time()
    hours_left = (max_episodes-epoch_id-1)*(episode_end_time-episode_start_time)*(1/(60*60))
    logger.info(f"\nTIMER: episode {epoch_id+1}/{max_episodes} took {(episode_end_time-episode_start_time)/60:.1f} minutes. Run finished in approximatly {hours_left:.1f} hours\n")


send_attachment_via_email(open_foam_case.path_to_postprocessing_folder+"/rmse_by_time.png", subject=config["name_of_run"], body= f"relative rmse\n(rmse at start: {open_foam_case.rmse_by_time[0]:.4f}, min: {open_foam_case.min_rmse:.4f} at t={open_foam_case.time_of_min_rmse})")

# dump all relevant files
objects = [open_foam_case, les, open_foam_core, rl_controller]
object_names = ["open_foam_case", "les", "open_foam_core", "rl_controller"]
for obj_name, obj in zip(object_names, objects):
    try:
        file_path = os.path.join(config["export_folder"],config["name_of_run"], f'{obj_name}.pickle')
        pickle_file = open(file_path, 'wb')
        pickle.dump(obj, pickle_file)
        pickle_file.close()
        print(f"saved {file_path}")
    except Exception as e:
        print(f"ERROR: failed to save {file_path}")
        print(e)


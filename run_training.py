from modules.LES_results.LES_results_worker import LESresultsWorker
from modules.openfoam.OpenFoamCase import OpenFoamCase
from modules.openfoamCore.openfoamCore import OpenFoamCore
from modules.utilities.logging import init_logger, close_logger
from modules.rl.rl_controller import RLController
from modules.config import Config
import numpy as np
import os
import pickle
import time
import json
import argparse


description = ""
epilog = ""
parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("config_path", type=str, metavar="path_to_config_file", help="specify the path to config file (absolute by default)")
parser.add_argument('-c', '--config', type=str, nargs='*', help='overwrite standard config file settings')

args = parser.parse_args()



def main():
    _timer = time.time()
    #############################################################
    # SETTINGS
    path_to_config = args.config_path

    #############################################################
    # init logging and import config
    logger = init_logger("openfoam_case")
    config = Config()
    config.load(path_to_config)
    tensorforce_config_dict = config["rl_settings"]
    #config.save("fff")

    # get settings
    max_episodes = config["max_episodes"] 
    initial_run_episodes = config["initial_run_episodes"] 
    epochs_per_episode = config["epochs_per_episode"] 
    writeInterval = config["writeinterval"] 
    points_per_batch_side = config["points_per_batch_side"] 
    max_episode_timesteps_for_agent = config["max_episode_timesteps_for_agent"] 
    action_min = float(config["action_min"])
    action_max = float(config["action_max"])
    cell_batch_rectangle_length = float(config["cell_batch_rectangle_length"])
    use_nut = config["use_nut"] 

    # init Open Foam Case
    open_foam_case = OpenFoamCase(logger, config)
    open_foam_case.setup_new_run(delete_old_files=True)

    # init Open Foam Core
    open_foam_core = OpenFoamCore(logger, config)

    # compile turbulence model
    open_foam_core.compile_turbulence_model(force_compilation=False)

    # determine action and state space
    if use_nut:
        state_space = dict(type='float', shape=(3*points_per_batch_side*points_per_batch_side+1,), min_value=-1.0, max_value=1.0)
    else:
        state_space = dict(type='float', shape=(2*points_per_batch_side*points_per_batch_side+1,), min_value=-1.0, max_value=1.0)
    action_space = dict(type='float', shape=(1, 1), min_value=action_min, max_value=action_max)

    # init Reinforcment Learnign
    tensorforce_config = os.path.join(config["export_folder"], config["name_of_run"], "ppo_temp.json") 
    with open(tensorforce_config, 'w') as fp:
        json.dump(tensorforce_config_dict, fp)

    rl_controller = RLController(state_scaling_margin = [1, 1.2, 1.2],
                                action_space = action_space,
                                state_space = state_space,
                                max_episode_timesteps = max_episode_timesteps_for_agent,
                                config=config,
                                path_to_tensorforce_config=tensorforce_config,
                                logger=logger)
                                
    rl_controller.import_experiences()

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
        rewards = rl_controller.compute_reward(cell_batch_inputs=open_foam_case.cell_batches, global_error=open_foam_case.rmse)

        # Feed recorded experience to agent
        rl_controller.feed_recorded_experience_to_agent(initial_states, action_batches, rewards, open_foam_case.coordinate_filter_indexes)
        rl_controller.update_agent()

        episode_end_time = time.time()
        hours_left = (max_episodes-epoch_id-1)*(episode_end_time-episode_start_time)*(1/(60*60))
        logger.info(f"\nTIMER: episode {epoch_id+1}/{max_episodes} took {(episode_end_time-episode_start_time)/60:.1f} minutes. Run finished in approximatly {hours_left:.1f} hours\n")

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

if __name__ == "__main__":
    main()
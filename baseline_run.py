
from modules.LES_results.LES_results_worker import LESresultsWorker
from modules.openfoam.OpenFoamCase import OpenFoamCase
from modules.openfoamCore.openfoamCore import OpenFoamCore
from modules.utilities.logging import init_logger, close_logger
from modules.rl.tests.utils import generate_random_two_dim_gaussian_fields
from modules.config import Config
import numpy as np
import os
import pickle

# init logging and import config
path_to_config = "config/baseline_run.conf"
logger = init_logger("openfoam_case")
config = Config()
config.load(path_to_config)

# init Open Foam Case
OpenFoamCase = OpenFoamCase(logger, config)

# init Open Foam Core
OpenFoamCore = OpenFoamCore(logger, config)

# compile turbulence model
#OpenFoamCore.compile_turbulence_model(force_compilation=True)
OpenFoamCase.setup_new_run(delete_old_files=True)


nutSource = np.zeros(shape=(OpenFoamCase.BlockMesh_coordinates_2D.shape[0],1))
OpenFoamCase.set_nutSource(time_folder=0, data=nutSource)

LES = LESresultsWorker(logger, config)
LES.load_data()
LES.perform_azimutal_averaging(OpenFoamCase.BlockMesh_coordinates_2D)
mean_field_indexes_of_fuel_injector, fuel_injector_velocities = LES.extract_fuel_injector_velocity(OpenFoamCase.BlockMesh_coordinates_2D)
mean_field_indexes_of_inlet, inlet_velocities = LES.extract_inlet_velocity(OpenFoamCase.BlockMesh_coordinates_2D)
OpenFoamCase.set_inlet_and_fuel_injector_boundary_condition(inlet_velocities, fuel_injector_velocities)

nutSource = np.zeros(shape=(OpenFoamCase.BlockMesh_coordinates_2D.shape[0],1))
OpenFoamCase.set_nutSource(time_folder=0, data=nutSource)

#OpenFoamCase.run()
OpenFoamCase.decompose()
OpenFoamCase.run_solver(endTime=60000, writeInterval=2000)
OpenFoamCase.reconstruct()

OpenFoamCase.import_results()
OpenFoamCase.import_log()

OpenFoamCase.init_target_mean_field(LES.mean_field)
OpenFoamCase.calculate_error()
OpenFoamCase.plot_results(enforce_plotting=True)
OpenFoamCase.plot_error(enforce_plotting=True)
#OpenFoamCase.create_interpolated_cell_batches(rectangle_length=0.01, points_per_batch_side=5)
   
# dump all relevant files
objects = [OpenFoamCase, LES, OpenFoamCore]
object_names = ["OpenFoamCase", "LES", "OpenFoamCore"]
for obj_name, obj in zip(object_names, objects):
    try:
        file_path = os.path.join(config["export_folder"],config["name_of_run"], f'{obj_name}.pickle')
        pickle_file = open(file_path, 'wb')
        pickle.dump(obj, pickle_file)
        pickle_file.close()
        print(f"saved {file_path}")
    except Exception:
        print(f"ERROR: failed to save {file_path}")
